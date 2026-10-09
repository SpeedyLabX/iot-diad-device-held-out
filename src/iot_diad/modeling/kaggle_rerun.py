"""Approved BASE-002 LR/MLP convergence rerun on a self-auditing Kaggle bundle."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import polars as pl
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

from iot_diad.modeling.baselines import (
    CAP,
    CLASSES,
    CONTRACT_PATHS,
    SEED,
    estimators,
    guard_manifest,
    make_transform_contract,
    metric_rows,
    object_hash,
    sha256,
    summarize,
    validate_probabilities,
    versions,
    write_json,
    write_parquet,
)
from iot_diad.modeling.data import METADATA_COLUMNS
from iot_diad.modeling.features import load_packet_feature_spec
from iot_diad.modeling.preprocessing import TrainOnlyPreprocessor

BASELINE_RELATIVE = Path("artifacts/experiments/baselines")
AMENDMENT_RELATIVE = Path("config/base002_kaggle_rerun.json")
BUNDLE_MANIFEST = "kaggle_bundle_manifest.json"
MODEL_ORDER = ("LR", "MLP")


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def normalized(value: Any) -> Any:
    return json.loads(json.dumps(value))


def validate_environment(amendment: dict[str, Any]) -> None:
    current_python = ".".join(platform.python_version_tuple()[:2])
    if current_python != amendment["required_python_major_minor"]:
        raise ValueError(
            f"Python {current_python} does not match the approved "
            f"{amendment['required_python_major_minor']} runtime"
        )
    actual = {
        name: importlib.metadata.version(name) for name in amendment["required_package_versions"]
    }
    if actual != amendment["required_package_versions"]:
        raise ValueError(
            "Package versions differ from the approved rerun environment: "
            f"actual={actual}, expected={amendment['required_package_versions']}"
        )


def validate_amendment(amendment: dict[str, Any]) -> None:
    if amendment["seed"] != SEED or amendment["training_cap_per_device_class"] != CAP:
        raise ValueError("The approved seed or training cap changed")
    if tuple(amendment["class_order"]) != CLASSES:
        raise ValueError("The approved class order changed")
    if amendment["minimum_output_free_bytes"] != 5 * 1024**3:
        raise ValueError("The approved minimum output disk safeguard changed")
    expected = {
        "LR": {"original_max_iter": 300, "amended_max_iter": 1000},
        "MLP": {"original_max_iter": 50, "amended_max_iter": 200},
    }
    for name, values in expected.items():
        record = amendment["models"].get(name, {})
        if any(record.get(key) != value for key, value in values.items()):
            raise ValueError(f"Unexpected {name} convergence amendment")
    if set(amendment["models"]) != set(MODEL_ORDER):
        raise ValueError("Only LR and MLP may be rerun under this amendment")


def validate_bundle(bundle_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    marker = read_json(bundle_root / BUNDLE_MANIFEST)
    amendment = read_json(bundle_root / AMENDMENT_RELATIVE)
    validate_amendment(amendment)
    validate_environment(amendment)
    files = marker.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Bundle manifest has no file inventory")
    for record in files:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe bundle path: {relative}")
        path = bundle_root / relative
        if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Bundle file integrity mismatch: {relative}")

    baseline = bundle_root / BASELINE_RELATIVE
    prepared = read_json(baseline / "prepared.json")
    config = read_json(baseline / "config.json")
    if sha256(baseline / "config.json") != prepared["config_sha256"]:
        raise ValueError("Prepared configuration hash mismatch")
    if sha256(baseline / "training_sample.parquet") != prepared["training_sample_sha256"]:
        raise ValueError("Training sample hash mismatch")
    if sha256(baseline / "training_sample_manifest.csv") != prepared["sample_manifest_sha256"]:
        raise ValueError("Training sample manifest hash mismatch")
    if prepared["sample_manifest_sha256"] != amendment["expected_sample_manifest_sha256"]:
        raise ValueError("Unexpected training sample manifest")
    if prepared["feature_contract_sha256"] != amendment["expected_feature_contract_sha256"]:
        raise ValueError("Unexpected packet feature contract")
    if prepared["training_rows"] != amendment["expected_training_rows"]:
        raise ValueError("Unexpected training row count")
    if prepared["rows_by_partition"] != amendment["expected_rows_by_partition"]:
        raise ValueError("Unexpected Gate 2 population")
    if config["seed"] != SEED or config["cap"] != CAP:
        raise ValueError("Original BASE-002 seed or cap changed")
    if tuple(config["class_order"]) != CLASSES:
        raise ValueError("Original BASE-002 class order changed")
    if object_hash(config["features"]) != prepared["feature_contract_sha256"]:
        raise ValueError("Feature contract object hash mismatch")
    for relative, digest in config["contracts"].items():
        current_digest = sha256(bundle_root / relative)
        if relative == "docs/reports/gate3_protocol.md":
            if digest != amendment["original_gate3_protocol_sha256"]:
                raise ValueError("Archived Gate 3 protocol hash changed")
            if current_digest != amendment["amended_gate3_protocol_sha256"]:
                raise ValueError("Approved amended Gate 3 protocol hash changed")
        elif current_digest != digest:
            raise ValueError(f"Source contract changed: {relative}")
    for name in MODEL_ORDER:
        original = normalized(estimators()[name].get_params())
        if original != config["estimators"][name]:
            raise ValueError(f"Archived {name} estimator contract changed")
        if original["max_iter"] != amendment["models"][name]["original_max_iter"]:
            raise ValueError(f"Archived {name} max_iter does not match the amendment")
    evaluation_rows = 0
    for source in prepared["evaluation_chunks"]:
        path = baseline / source["path"]
        if path.stat().st_size <= 0 or sha256(path) != source["sha256"]:
            raise ValueError(f"Evaluation shard integrity mismatch: {source['path']}")
        evaluation_rows += source["rows"]
    expected_evaluation = sum(
        amendment["expected_rows_by_partition"][key] for key in ("calibration", "test")
    )
    if evaluation_rows != expected_evaluation:
        raise ValueError("Evaluation shard row total mismatch")
    return amendment, prepared, config


def amended_estimator(
    name: str, amendment: dict[str, Any], config: dict[str, Any]
) -> tuple[Any, dict[str, Any]]:
    estimator = estimators()[name]
    estimator.set_params(max_iter=amendment["models"][name]["amended_max_iter"])
    final = normalized(estimator.get_params())
    original = config["estimators"][name]
    changed = {key for key in final if final[key] != original[key]}
    if changed != {"max_iter"}:
        raise ValueError(f"{name} changed fields beyond max_iter: {sorted(changed)}")
    return estimator, final


def execution_code_hashes(bundle_root: Path) -> dict[str, str]:
    source_root = bundle_root / "src/iot_diad/modeling"
    return {
        str(path.relative_to(bundle_root)): sha256(path)
        for path in sorted(source_root.glob("*.py"))
    }


def fit_model(
    bundle_root: Path,
    output_root: Path,
    name: str,
    amendment: dict[str, Any],
    prepared: dict[str, Any],
    config: dict[str, Any],
) -> tuple[TrainOnlyPreprocessor, Any, Path]:
    run_id = amendment["models"][name]["run_id"]
    directory = output_root / run_id
    directory.mkdir(parents=True, exist_ok=True)
    estimator, final_params = amended_estimator(name, amendment, config)
    code_hashes = execution_code_hashes(bundle_root)
    run_config = {
        "amendment_id": amendment["amendment_id"],
        "amendment_sha256": sha256(bundle_root / AMENDMENT_RELATIVE),
        "bundle_manifest_sha256": sha256(bundle_root / BUNDLE_MANIFEST),
        "cap": CAP,
        "class_order": list(CLASSES),
        "config_sha256": prepared["config_sha256"],
        "estimator": final_params,
        "execution_code_sha256": code_hashes,
        "feature_contract_sha256": prepared["feature_contract_sha256"],
        "model_id": name,
        "run_id": run_id,
        "sample_manifest_sha256": prepared["sample_manifest_sha256"],
        "seed": SEED,
        "software_versions": versions(),
    }
    guard_manifest(directory / "run_config.json", run_config)
    fit_path = directory / "fit.json"
    model_path = directory / "model.joblib"
    if fit_path.exists():
        fit = read_json(fit_path)
        if sha256(model_path) != fit["model_sha256"]:
            raise ValueError(f"Saved {name} model integrity mismatch")
        processor, estimator = joblib.load(model_path)
        return processor, estimator, directory

    sample_path = bundle_root / BASELINE_RELATIVE / "training_sample.parquet"
    sample = pl.read_parquet(sample_path)
    if sample.height != prepared["training_rows"] or set(sample["_partition"]) != {"train"}:
        raise ValueError("Training sample population changed")
    encoder = LabelEncoder().fit(sample["_label"].to_numpy())
    if tuple(encoder.classes_) != CLASSES:
        raise ValueError("Training sample class order changed")
    processor = TrainOnlyPreprocessor(load_packet_feature_spec(bundle_root / CONTRACT_PATHS[0]))
    started = time.monotonic()
    print(f"{run_id}: preprocessing {sample.height} training rows", flush=True)
    features = processor.fit_transform(sample)
    print(f"{run_id}: fitting amended estimator on {features.shape}", flush=True)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=8):
        warnings.simplefilter("always")
        estimator.fit(features, encoder.transform(sample["_label"].to_numpy()))
    messages = [{"category": w.category.__name__, "message": str(w.message)} for w in caught]
    convergence_failure = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    temporary = model_path.with_suffix(".joblib.tmp")
    joblib.dump((processor, estimator), temporary, compress=3)
    temporary.replace(model_path)
    write_json(
        fit_path,
        {
            "execution_code_sha256": code_hashes,
            "fit_seconds": time.monotonic() - started,
            "iterations": np.asarray(getattr(estimator, "n_iter_", [])).tolist(),
            "model_sha256": sha256(model_path),
            "preprocessor": processor.audit(),
            "status": "convergence_failure" if convergence_failure else "passed",
            "training_only_validation": name == "MLP",
            "training_rows": sample.height,
            "warnings": messages,
        },
    )
    del features, sample
    print(f"{run_id}: fit saved; convergence_failure={convergence_failure}", flush=True)
    return processor, estimator, directory


def export_models(
    bundle_root: Path,
    output_root: Path,
    amendment: dict[str, Any],
    prepared: dict[str, Any],
    fitted: dict[str, tuple[TrainOnlyPreprocessor, Any, Path]],
) -> None:
    expected_classes = np.arange(len(CLASSES))
    contracts = {}
    for name, (processor, estimator, directory) in fitted.items():
        if not np.array_equal(estimator.classes_, expected_classes):
            raise ValueError(f"{name} estimator class order mismatch")
        contract = make_transform_contract(
            bundle_root,
            processor,
            prepared["sample_manifest_sha256"],
            prepared["feature_contract_sha256"],
        )
        guard_manifest(directory / "transform_contract.json", contract)
        contracts[name] = contract
    if len({object_hash(contract) for contract in contracts.values()}) != 1:
        raise ValueError("LR and MLP fitted preprocessing contracts differ")

    amendment_sha = sha256(bundle_root / AMENDMENT_RELATIVE)
    bundle_sha = sha256(bundle_root / BUNDLE_MANIFEST)
    software = json.dumps(versions(), sort_keys=True)
    chunks: dict[str, list[dict[str, Any]]] = {name: [] for name in MODEL_ORDER}
    started = time.monotonic()
    for index, source in enumerate(prepared["evaluation_chunks"]):
        records: dict[str, dict[str, Any]] = {}
        missing = []
        for name in MODEL_ORDER:
            directory = fitted[name][2]
            destination = directory / "probabilities" / f"{index:05d}.parquet"
            receipt = destination.with_suffix(".json")
            if receipt.exists():
                record = read_json(receipt)
                if (
                    record["sha256"] != sha256(destination)
                    or record["input_sha256"] != source["sha256"]
                    or record["run_config_sha256"] != sha256(directory / "run_config.json")
                ):
                    raise ValueError(f"Resume chunk integrity mismatch: {destination}")
                records[name] = record
            else:
                missing.append(name)
        if missing:
            source_path = bundle_root / BASELINE_RELATIVE / source["path"]
            if sha256(source_path) != source["sha256"]:
                raise ValueError(f"Evaluation shard changed during execution: {source['path']}")
            frame = pl.read_parquet(source_path)
            metadata = frame.select(METADATA_COLUMNS)
            features = fitted[MODEL_ORDER[0]][0].transform(frame)
            if features.dtype != np.float64 or not np.isfinite(features).all():
                raise ValueError("Streaming preprocessing produced invalid features")
        for name in missing:
            estimator = fitted[name][1]
            directory = fitted[name][2]
            destination = directory / "probabilities" / f"{index:05d}.parquet"
            receipt = destination.with_suffix(".json")
            with threadpool_limits(limits=8):
                probabilities = np.asarray(estimator.predict_proba(features), dtype=np.float64)
            validate_probabilities(probabilities, metadata.height)
            exported = metadata.with_columns(
                **{f"prob__{label}": probabilities[:, i] for i, label in enumerate(CLASSES)},
                predicted_label=pl.Series(np.asarray(CLASSES)[probabilities.argmax(axis=1)]),
                model_id=pl.lit(name),
                run_id=pl.lit(amendment["models"][name]["run_id"]),
                seed=pl.lit(SEED),
                feature_contract_sha256=pl.lit(prepared["feature_contract_sha256"]),
                sample_manifest_sha256=pl.lit(prepared["sample_manifest_sha256"]),
                protocol_amendment_sha256=pl.lit(amendment_sha),
                input_bundle_manifest_sha256=pl.lit(bundle_sha),
                software_versions=pl.lit(software),
            )
            write_parquet(destination, exported)
            record = {
                "input_sha256": source["sha256"],
                "max_probability_sum_error": float(np.abs(probabilities.sum(axis=1) - 1).max()),
                "path": str(destination.relative_to(output_root)),
                "rows": metadata.height,
                "run_config_sha256": sha256(directory / "run_config.json"),
                "sha256": sha256(destination),
            }
            write_json(receipt, record)
            records[name] = record
        for name in MODEL_ORDER:
            chunks[name].append(records[name])
        if index % 25 == 0:
            print(
                f"LR/MLP amended export: {index + 1}/{len(prepared['evaluation_chunks'])}",
                flush=True,
            )
    for name in MODEL_ORDER:
        directory = fitted[name][2]
        write_json(
            directory / "exports.json",
            {
                "chunks": chunks[name],
                "fit_status": read_json(directory / "fit.json")["status"],
                "prediction_seconds_this_invocation": time.monotonic() - started,
                "rows": sum(record["rows"] for record in chunks[name]),
                "status": "complete",
            },
        )
        summarize(directory)


def metrics_from_matrices(
    matrices: dict[tuple[str, str], np.ndarray[Any, np.dtype[np.int64]]],
) -> dict[str, Any]:
    devices = [
        {"partition": key[0], "device": key[1], **metric_rows(matrix)}
        for key, matrix in sorted(matrices.items())
    ]
    aggregates = {}
    for partition in ("calibration", "test"):
        entries = [device for device in devices if device["partition"] == partition]
        scores = np.asarray([device["macro_f1"] for device in entries])
        total = sum(
            (matrix for key, matrix in matrices.items() if key[0] == partition),
            start=np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64),
        )
        aggregates[partition] = {
            "device_count": len(entries),
            "mean_device_macro_f1": float(scores.mean()),
            "median_device_macro_f1": float(np.median(scores)),
            "q25_device_macro_f1": float(np.quantile(scores, 0.25)),
            "q75_device_macro_f1": float(np.quantile(scores, 0.75)),
            "iqr_device_macro_f1": float(np.quantile(scores, 0.75) - np.quantile(scores, 0.25)),
            "min_device_macro_f1": float(scores.min()),
            "max_device_macro_f1": float(scores.max()),
            "global_row_weighted": metric_rows(total),
            "device_balanced_per_class_recall": {
                label: {
                    "recall": float(
                        np.mean(
                            [
                                device["per_class"][label]["recall"]
                                for device in entries
                                if device["per_class"][label]["support"] > 0
                            ]
                        )
                    ),
                    "devices": sum(device["per_class"][label]["support"] > 0 for device in entries),
                }
                for label in CLASSES
            },
            "leave_one_device_out": {
                device["device"]: (
                    float(np.delete(scores, index).mean()) if len(scores) > 1 else None
                )
                for index, device in enumerate(entries)
            },
        }
    return {
        "aggregates": aggregates,
        "class_order": list(CLASSES),
        "devices": devices,
        "metric_scope": "descriptive point estimates only; no conformal calibration or inference",
    }


def verify_outputs(
    bundle_root: Path,
    output_root: Path,
    amendment: dict[str, Any],
    prepared: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    expected_input_contract = {
        "amendment_sha256": sha256(bundle_root / AMENDMENT_RELATIVE),
        "bundle_manifest_sha256": sha256(bundle_root / BUNDLE_MANIFEST),
        "minimum_output_free_bytes": amendment["minimum_output_free_bytes"],
        "prepared_sha256": sha256(bundle_root / BASELINE_RELATIVE / "prepared.json"),
    }
    if read_json(output_root / "input_contract.json") != expected_input_contract:
        raise ValueError("Rerun input contract mismatch")
    partition_rows: Counter[str] = Counter()
    model_records = []
    expected_amendment_sha = sha256(bundle_root / AMENDMENT_RELATIVE)
    expected_bundle_sha = sha256(bundle_root / BUNDLE_MANIFEST)
    expected_rows = sum(
        amendment["expected_rows_by_partition"][key] for key in ("calibration", "test")
    )
    for name in MODEL_ORDER:
        directory = output_root / amendment["models"][name]["run_id"]
        run_config = read_json(directory / "run_config.json")
        fit = read_json(directory / "fit.json")
        exports = read_json(directory / "exports.json")
        _, expected_params = amended_estimator(name, amendment, config)
        expected_run_values = {
            "amendment_id": amendment["amendment_id"],
            "amendment_sha256": expected_amendment_sha,
            "bundle_manifest_sha256": expected_bundle_sha,
            "cap": CAP,
            "class_order": list(CLASSES),
            "config_sha256": prepared["config_sha256"],
            "estimator": expected_params,
            "execution_code_sha256": execution_code_hashes(bundle_root),
            "feature_contract_sha256": prepared["feature_contract_sha256"],
            "model_id": name,
            "run_id": amendment["models"][name]["run_id"],
            "sample_manifest_sha256": prepared["sample_manifest_sha256"],
            "seed": SEED,
        }
        for key, value in expected_run_values.items():
            if run_config.get(key) != value:
                raise ValueError(f"{name} run configuration mismatch: {key}")
        producing_versions = run_config.get("software_versions", {})
        producing_python = str(producing_versions.get("python", ""))
        if ".".join(producing_python.split(".")[:2]) != amendment["required_python_major_minor"]:
            raise ValueError(f"{name} producing Python version mismatch")
        producing_packages = {
            key: producing_versions.get(key) for key in amendment["required_package_versions"]
        }
        if producing_packages != amendment["required_package_versions"]:
            raise ValueError(f"{name} producing package versions mismatch")
        expected_software = json.dumps(producing_versions, sort_keys=True)
        if sha256(directory / "model.joblib") != fit["model_sha256"]:
            raise ValueError(f"{name} model hash mismatch")
        processor, estimator = joblib.load(directory / "model.joblib")
        if normalized(estimator.get_params()) != expected_params:
            raise ValueError(f"{name} saved estimator parameters mismatch")
        if not np.array_equal(estimator.classes_, np.arange(len(CLASSES))):
            raise ValueError(f"{name} saved estimator class order mismatch")
        if processor.audit() != fit["preprocessor"]:
            raise ValueError(f"{name} saved preprocessor audit mismatch")
        if fit["execution_code_sha256"] != execution_code_hashes(bundle_root):
            raise ValueError(f"{name} fit execution code mismatch")
        if fit["training_rows"] != prepared["training_rows"]:
            raise ValueError(f"{name} fit training row count mismatch")
        if fit["training_only_validation"] != (name == "MLP"):
            raise ValueError(f"{name} training-only validation record mismatch")
        recorded_iterations = fit["iterations"]
        actual_iterations = np.asarray(getattr(estimator, "n_iter_", [])).tolist()
        if recorded_iterations != actual_iterations:
            raise ValueError(f"{name} iteration record mismatch")
        convergence_warning = any(
            warning.get("category") == "ConvergenceWarning" for warning in fit["warnings"]
        )
        expected_status = "convergence_failure" if convergence_warning else "passed"
        if fit["status"] != expected_status:
            raise ValueError(f"{name} fit status does not match its warnings")
        transform_contract = read_json(directory / "transform_contract.json")
        expected_transform = {
            "feature_contract_sha256": prepared["feature_contract_sha256"],
            "preprocessing_source_sha256": sha256(
                bundle_root / "src/iot_diad/modeling/preprocessing.py"
            ),
            "preprocessor_audit": processor.audit(),
            "sample_manifest_sha256": prepared["sample_manifest_sha256"],
            "software_versions": producing_versions,
        }
        if transform_contract != expected_transform:
            raise ValueError(f"{name} transform contract mismatch")
        if exports["status"] != "complete" or exports["rows"] != expected_rows:
            raise ValueError(f"{name} export row count mismatch")
        if exports["fit_status"] != fit["status"]:
            raise ValueError(f"{name} export fit status mismatch")
        if len(exports["chunks"]) != len(prepared["evaluation_chunks"]):
            raise ValueError(f"{name} export chunk count mismatch")
        current_rows: Counter[str] = Counter()
        matrices: dict[tuple[str, str], np.ndarray[Any, np.dtype[np.int64]]] = defaultdict(
            lambda: np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
        )
        label_indices = {label: index for index, label in enumerate(CLASSES)}
        for index, (source, record) in enumerate(
            zip(prepared["evaluation_chunks"], exports["chunks"], strict=True)
        ):
            expected_path = str(
                (directory / "probabilities" / f"{index:05d}.parquet").relative_to(output_root)
            )
            if record["path"] != expected_path or record["rows"] != source["rows"]:
                raise ValueError(f"{name} output chunk path or row count mismatch: {index}")
            path = output_root / record["path"]
            if read_json(path.with_suffix(".json")) != record:
                raise ValueError(f"{name} output receipt mismatch: {record['path']}")
            if (
                record["input_sha256"] != source["sha256"]
                or record["sha256"] != sha256(path)
                or record["run_config_sha256"] != sha256(directory / "run_config.json")
            ):
                raise ValueError(f"{name} output chunk lineage mismatch: {record['path']}")
            frame = pl.read_parquet(path)
            source_frame = pl.read_parquet(
                bundle_root / BASELINE_RELATIVE / source["path"],
                columns=list(METADATA_COLUMNS),
            )
            if frame.height != source["rows"] or not frame.select(METADATA_COLUMNS).equals(
                source_frame
            ):
                raise ValueError(f"{name} output row identity mismatch: {record['path']}")
            probabilities = frame.select([f"prob__{label}" for label in CLASSES]).to_numpy()
            validate_probabilities(probabilities, frame.height)
            probability_error = float(np.abs(probabilities.sum(axis=1) - 1).max())
            if not np.isclose(
                record["max_probability_sum_error"], probability_error, atol=1e-15, rtol=0
            ):
                raise ValueError(f"{name} probability audit mismatch: {record['path']}")
            expected_prediction = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
            if not np.array_equal(frame["predicted_label"].to_numpy(), expected_prediction):
                raise ValueError(f"{name} predicted labels do not match probabilities")
            for partition, count in frame.group_by("_partition").len().rows():
                current_rows[str(partition)] += int(count)
            grouped = frame.group_by(
                "_partition", "_canonical_device_id", "_label", "predicted_label"
            ).len()
            for partition, device, true, predicted, count in grouped.rows():
                matrices[(str(partition), str(device))][
                    label_indices[str(true)], label_indices[str(predicted)]
                ] += int(count)
            constants = {
                "feature_contract_sha256": prepared["feature_contract_sha256"],
                "sample_manifest_sha256": prepared["sample_manifest_sha256"],
                "model_id": name,
                "run_id": amendment["models"][name]["run_id"],
                "seed": SEED,
                "protocol_amendment_sha256": expected_amendment_sha,
                "input_bundle_manifest_sha256": expected_bundle_sha,
                "software_versions": expected_software,
            }
            for column, value in constants.items():
                if frame[column].n_unique() != 1 or frame[column][0] != value:
                    raise ValueError(f"{name} output metadata mismatch: {column}")
        expected_partitions = {
            key: amendment["expected_rows_by_partition"][key] for key in ("calibration", "test")
        }
        if dict(current_rows) != expected_partitions:
            raise ValueError(f"{name} partition row count mismatch")
        if read_json(directory / "metrics.json") != metrics_from_matrices(dict(matrices)):
            raise ValueError(f"{name} point metrics mismatch")
        if not partition_rows:
            partition_rows.update(current_rows)
        model_records.append(
            {
                "fit_status": fit["status"],
                "iterations": fit["iterations"],
                "model_id": name,
                "model_sha256": fit["model_sha256"],
                "rows": exports["rows"],
                "run_config_sha256": sha256(directory / "run_config.json"),
                "run_id": run_config["run_id"],
            }
        )
    result = {
        "amendment_id": amendment["amendment_id"],
        "models": model_records,
        "rows_by_partition": dict(partition_rows),
        "status": "passed",
    }
    write_json(output_root / "rerun_verification.json", result)
    return result


def run(bundle_root: Path, output_root: Path) -> dict[str, Any]:
    amendment, prepared, config = validate_bundle(bundle_root)
    output_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_root).free
    if free_bytes < amendment["minimum_output_free_bytes"]:
        raise ValueError(
            f"Insufficient output disk: {free_bytes} free bytes; "
            f"{amendment['minimum_output_free_bytes']} required"
        )
    guard_manifest(
        output_root / "input_contract.json",
        {
            "amendment_sha256": sha256(bundle_root / AMENDMENT_RELATIVE),
            "bundle_manifest_sha256": sha256(bundle_root / BUNDLE_MANIFEST),
            "minimum_output_free_bytes": amendment["minimum_output_free_bytes"],
            "prepared_sha256": sha256(bundle_root / BASELINE_RELATIVE / "prepared.json"),
        },
    )
    fitted = {}
    for name in MODEL_ORDER:
        fitted[name] = fit_model(bundle_root, output_root, name, amendment, prepared, config)
    export_models(bundle_root, output_root, amendment, prepared, fitted)
    verification = verify_outputs(bundle_root, output_root, amendment, prepared, config)
    statuses = {record["model_id"]: record["fit_status"] for record in verification["models"]}
    summary = {
        "acceptance_status": (
            "passed" if all(status == "passed" for status in statuses.values()) else "blocked"
        ),
        "amendment_id": amendment["amendment_id"],
        "fit_statuses": statuses,
        "next_action": (
            "retain this output directory for independent verification"
            if all(status == "passed" for status in statuses.values())
            else "preserve outputs and reopen Gate 3 because a convergence warning remains"
        ),
        "verification_status": verification["status"],
    }
    write_json(output_root / "rerun_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        amendment, prepared, config = validate_bundle(args.bundle_root)
        result = verify_outputs(args.bundle_root, args.output_root, amendment, prepared, config)
    else:
        result = run(args.bundle_root, args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
