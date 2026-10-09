"""Final approved BASE-002 LR solver amendment with training-first acceptance."""

from __future__ import annotations

import argparse
import json
import shutil
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

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
    sha256,
    summarize,
    validate_probabilities,
    versions,
    write_json,
    write_parquet,
)
from iot_diad.modeling.data import METADATA_COLUMNS
from iot_diad.modeling.features import load_packet_feature_spec
from iot_diad.modeling.kaggle_rerun import (
    BASELINE_RELATIVE,
    BUNDLE_MANIFEST,
    metrics_from_matrices,
    normalized,
    read_json,
    validate_bundle,
)
from iot_diad.modeling.preprocessing import TrainOnlyPreprocessor

AMENDMENT_RELATIVE = Path("config/base002_lr_solver_amendment.json")
AMENDMENT_REPORT_RELATIVE = Path("docs/reports/base002_lr_solver_amendment.md")
SOURCE_RELATIVE = Path("src/iot_diad/modeling/lr_solver_rerun.py")


def validate_solver_amendment(
    repo_root: Path, bundle_root: Path, amendment: dict[str, Any], prepared: dict[str, Any]
) -> None:
    expected = {
        "model_id": "LR",
        "solver": "newton-cholesky",
        "max_iter": 200,
        "seed": SEED,
        "training_cap_per_device_class": CAP,
        "expected_training_rows": prepared["training_rows"],
        "expected_sample_manifest_sha256": prepared["sample_manifest_sha256"],
        "expected_feature_contract_sha256": prepared["feature_contract_sha256"],
        "expected_rows_by_partition": prepared["rows_by_partition"],
        "base_bundle_manifest_sha256": sha256(bundle_root / BUNDLE_MANIFEST),
        "amendment_report_sha256": sha256(repo_root / AMENDMENT_REPORT_RELATIVE),
    }
    for key, value in expected.items():
        if amendment.get(key) != value:
            raise ValueError(f"LR solver amendment mismatch: {key}")
    if tuple(amendment["class_order"]) != CLASSES:
        raise ValueError("LR solver amendment class order changed")
    if amendment["minimum_output_free_bytes"] != 2_684_354_560:
        raise ValueError("LR solver amendment disk safeguard changed")


def amended_lr(amendment: dict[str, Any], config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    estimator = estimators()["LR"]
    estimator.set_params(solver=amendment["solver"], max_iter=amendment["max_iter"])
    params = normalized(estimator.get_params())
    original = config["estimators"]["LR"]
    changed = {key for key in params if params[key] != original[key]}
    if changed != {"max_iter", "solver"}:
        raise ValueError(f"Unexpected LR parameter changes: {sorted(changed)}")
    return estimator, params


def source_hash(repo_root: Path) -> str:
    return sha256(repo_root / SOURCE_RELATIVE)


def fit_lr(
    repo_root: Path,
    bundle_root: Path,
    output_root: Path,
    amendment: dict[str, Any],
    prepared: dict[str, Any],
    config: dict[str, Any],
) -> tuple[TrainOnlyPreprocessor, Any, Path, bool]:
    directory = output_root / amendment["run_id"]
    directory.mkdir(parents=True, exist_ok=True)
    estimator, params = amended_lr(amendment, config)
    run_config = {
        "amendment_id": amendment["amendment_id"],
        "amendment_sha256": sha256(repo_root / AMENDMENT_RELATIVE),
        "amendment_report_sha256": amendment["amendment_report_sha256"],
        "base_bundle_manifest_sha256": amendment["base_bundle_manifest_sha256"],
        "cap": CAP,
        "class_order": list(CLASSES),
        "config_sha256": prepared["config_sha256"],
        "estimator": params,
        "feature_contract_sha256": prepared["feature_contract_sha256"],
        "model_id": "LR",
        "run_id": amendment["run_id"],
        "runner_source_sha256": source_hash(repo_root),
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
            raise ValueError("Saved amended LR model integrity mismatch")
        processor, estimator = joblib.load(model_path)
        return processor, estimator, directory, fit["status"] == "passed"

    sample = pl.read_parquet(bundle_root / BASELINE_RELATIVE / "training_sample.parquet")
    if sample.height != prepared["training_rows"] or set(sample["_partition"]) != {"train"}:
        raise ValueError("Training sample population changed")
    encoder = LabelEncoder().fit(sample["_label"].to_numpy())
    if tuple(encoder.classes_) != CLASSES:
        raise ValueError("Training sample class order changed")
    processor = TrainOnlyPreprocessor(load_packet_feature_spec(bundle_root / CONTRACT_PATHS[0]))
    started = time.monotonic()
    print(f"{amendment['run_id']}: preprocessing {sample.height} training rows", flush=True)
    features = processor.fit_transform(sample)
    print(f"{amendment['run_id']}: fitting on {features.shape}", flush=True)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=8):
        warnings.simplefilter("always")
        estimator.fit(features, encoder.transform(sample["_label"].to_numpy()))
        training_probabilities = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    messages = [{"category": w.category.__name__, "message": str(w.message)} for w in caught]
    validate_probabilities(training_probabilities, sample.height)
    iterations = np.asarray(getattr(estimator, "n_iter_", [])).tolist()
    warning_free = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    stopped_before_limit = bool(iterations) and all(
        int(value) < amendment["max_iter"] for value in iterations
    )
    finite_parameters = bool(
        np.isfinite(estimator.coef_).all() and np.isfinite(estimator.intercept_).all()
    )
    accepted = warning_free and stopped_before_limit and finite_parameters
    temporary = model_path.with_suffix(".joblib.tmp")
    joblib.dump((processor, estimator), temporary, compress=3)
    temporary.replace(model_path)
    write_json(
        fit_path,
        {
            "acceptance_checks": {
                "finite_coefficients_and_intercepts": finite_parameters,
                "finite_normalized_training_probabilities": True,
                "no_convergence_warning": warning_free,
                "stopped_before_iteration_limit": stopped_before_limit,
            },
            "execution_seconds": time.monotonic() - started,
            "iterations": iterations,
            "model_sha256": sha256(model_path),
            "preprocessor": processor.audit(),
            "runner_source_sha256": source_hash(repo_root),
            "status": "passed" if accepted else "training_acceptance_failure",
            "training_probability_max_sum_error": float(
                np.abs(training_probabilities.sum(axis=1) - 1).max()
            ),
            "training_rows": sample.height,
            "warnings": messages,
        },
    )
    del features, training_probabilities, sample
    print(f"{amendment['run_id']}: fit saved; accepted={accepted}", flush=True)
    return processor, estimator, directory, accepted


def export_lr(
    repo_root: Path,
    bundle_root: Path,
    output_root: Path,
    amendment: dict[str, Any],
    prepared: dict[str, Any],
    processor: TrainOnlyPreprocessor,
    estimator: Any,
    directory: Path,
) -> None:
    if not np.array_equal(estimator.classes_, np.arange(len(CLASSES))):
        raise ValueError("Amended LR class order mismatch")
    guard_manifest(
        directory / "transform_contract.json",
        make_transform_contract(
            bundle_root,
            processor,
            prepared["sample_manifest_sha256"],
            prepared["feature_contract_sha256"],
        ),
    )
    amendment_sha = sha256(repo_root / AMENDMENT_RELATIVE)
    bundle_sha = sha256(bundle_root / BUNDLE_MANIFEST)
    software = json.dumps(versions(), sort_keys=True)
    chunks: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, source in enumerate(prepared["evaluation_chunks"]):
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
        else:
            source_path = bundle_root / BASELINE_RELATIVE / source["path"]
            if sha256(source_path) != source["sha256"]:
                raise ValueError(f"Evaluation input changed: {source['path']}")
            frame = pl.read_parquet(source_path)
            metadata = frame.select(METADATA_COLUMNS)
            features = processor.transform(frame)
            if features.dtype != np.float64 or not np.isfinite(features).all():
                raise ValueError("Streaming preprocessing produced invalid features")
            with threadpool_limits(limits=8):
                probabilities = np.asarray(estimator.predict_proba(features), dtype=np.float64)
            validate_probabilities(probabilities, metadata.height)
            exported = metadata.with_columns(
                **{f"prob__{label}": probabilities[:, i] for i, label in enumerate(CLASSES)},
                predicted_label=pl.Series(np.asarray(CLASSES)[probabilities.argmax(axis=1)]),
                model_id=pl.lit("LR"),
                run_id=pl.lit(amendment["run_id"]),
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
        chunks.append(record)
        if index % 25 == 0:
            print(
                f"{amendment['run_id']}: exported {index + 1}/{len(prepared['evaluation_chunks'])}",
                flush=True,
            )
    write_json(
        directory / "exports.json",
        {
            "chunks": chunks,
            "fit_status": "passed",
            "prediction_seconds_this_invocation": time.monotonic() - started,
            "rows": sum(record["rows"] for record in chunks),
            "status": "complete",
        },
    )
    summarize(directory)


def verify(
    repo_root: Path,
    bundle_root: Path,
    output_root: Path,
    amendment: dict[str, Any],
    prepared: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    directory = output_root / amendment["run_id"]
    expected_input_contract = {
        "amendment_sha256": sha256(repo_root / AMENDMENT_RELATIVE),
        "amendment_report_sha256": amendment["amendment_report_sha256"],
        "base_bundle_manifest_sha256": amendment["base_bundle_manifest_sha256"],
        "prepared_sha256": sha256(bundle_root / BASELINE_RELATIVE / "prepared.json"),
    }
    if read_json(output_root / "input_contract.json") != expected_input_contract:
        raise ValueError("Amended LR input contract mismatch")
    run_config = read_json(directory / "run_config.json")
    fit = read_json(directory / "fit.json")
    _, expected_params = amended_lr(amendment, config)
    expected_run = {
        "amendment_id": amendment["amendment_id"],
        "amendment_sha256": sha256(repo_root / AMENDMENT_RELATIVE),
        "amendment_report_sha256": amendment["amendment_report_sha256"],
        "base_bundle_manifest_sha256": amendment["base_bundle_manifest_sha256"],
        "cap": CAP,
        "class_order": list(CLASSES),
        "config_sha256": prepared["config_sha256"],
        "estimator": expected_params,
        "feature_contract_sha256": prepared["feature_contract_sha256"],
        "model_id": "LR",
        "run_id": amendment["run_id"],
        "runner_source_sha256": source_hash(repo_root),
        "sample_manifest_sha256": prepared["sample_manifest_sha256"],
        "seed": SEED,
        "software_versions": versions(),
    }
    if run_config != expected_run:
        raise ValueError("Amended LR run configuration mismatch")
    if sha256(directory / "model.joblib") != fit["model_sha256"]:
        raise ValueError("Amended LR model hash mismatch")
    processor, estimator = joblib.load(directory / "model.joblib")
    if normalized(estimator.get_params()) != expected_params:
        raise ValueError("Saved amended LR parameters mismatch")
    if processor.audit() != fit["preprocessor"]:
        raise ValueError("Saved amended LR preprocessor mismatch")
    if fit["runner_source_sha256"] != source_hash(repo_root):
        raise ValueError("Amended LR runner source changed")
    checks = fit["acceptance_checks"]
    if fit["status"] != "passed" or not all(checks.values()):
        raise ValueError("Amended LR did not pass the training-side acceptance rule")
    if not fit["iterations"] or not all(
        int(value) < amendment["max_iter"] for value in fit["iterations"]
    ):
        raise ValueError("Amended LR iteration criterion mismatch")
    if fit["iterations"] != np.asarray(estimator.n_iter_).tolist():
        raise ValueError("Amended LR saved iteration record mismatch")
    if any(warning["category"] == "ConvergenceWarning" for warning in fit["warnings"]):
        raise ValueError("Amended LR retained a convergence warning")
    if not np.isfinite(estimator.coef_).all() or not np.isfinite(estimator.intercept_).all():
        raise ValueError("Amended LR has non-finite parameters")
    if not np.array_equal(estimator.classes_, np.arange(len(CLASSES))):
        raise ValueError("Amended LR saved class order mismatch")
    training = pl.read_parquet(bundle_root / BASELINE_RELATIVE / "training_sample.parquet")
    if training.height != fit["training_rows"] or training.height != prepared["training_rows"]:
        raise ValueError("Amended LR training row count mismatch")
    training_features = processor.transform(training)
    with threadpool_limits(limits=8):
        training_probabilities = np.asarray(
            estimator.predict_proba(training_features), dtype=np.float64
        )
    validate_probabilities(training_probabilities, training.height)
    training_error = float(np.abs(training_probabilities.sum(axis=1) - 1).max())
    if not np.isclose(
        fit["training_probability_max_sum_error"], training_error, atol=1e-15, rtol=0
    ):
        raise ValueError("Amended LR training probability audit mismatch")
    del training, training_features, training_probabilities
    expected_transform = make_transform_contract(
        bundle_root,
        processor,
        prepared["sample_manifest_sha256"],
        prepared["feature_contract_sha256"],
    )
    if read_json(directory / "transform_contract.json") != expected_transform:
        raise ValueError("Amended LR transform contract mismatch")
    exports = read_json(directory / "exports.json")
    expected_rows = sum(
        amendment["expected_rows_by_partition"][key] for key in ("calibration", "test")
    )
    if (
        exports["status"] != "complete"
        or exports["fit_status"] != "passed"
        or exports["rows"] != expected_rows
        or len(exports["chunks"]) != len(prepared["evaluation_chunks"])
    ):
        raise ValueError("Amended LR export manifest mismatch")
    partition_rows: Counter[str] = Counter()
    matrices: dict[tuple[str, str], np.ndarray[Any, np.dtype[np.int64]]] = defaultdict(
        lambda: np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    )
    labels = {label: index for index, label in enumerate(CLASSES)}
    amendment_sha = sha256(repo_root / AMENDMENT_RELATIVE)
    bundle_sha = sha256(bundle_root / BUNDLE_MANIFEST)
    software = json.dumps(versions(), sort_keys=True)
    for index, (source, record) in enumerate(
        zip(prepared["evaluation_chunks"], exports["chunks"], strict=True)
    ):
        path = directory / "probabilities" / f"{index:05d}.parquet"
        if record != read_json(path.with_suffix(".json")):
            raise ValueError(f"Amended LR receipt mismatch: {index}")
        if (
            record["path"] != str(path.relative_to(output_root))
            or record["rows"] != source["rows"]
            or record["input_sha256"] != source["sha256"]
            or record["sha256"] != sha256(path)
            or record["run_config_sha256"] != sha256(directory / "run_config.json")
        ):
            raise ValueError(f"Amended LR output lineage mismatch: {index}")
        frame = pl.read_parquet(path)
        source_metadata = pl.read_parquet(
            bundle_root / BASELINE_RELATIVE / source["path"], columns=list(METADATA_COLUMNS)
        )
        if frame.height != source["rows"] or not frame.select(METADATA_COLUMNS).equals(
            source_metadata
        ):
            raise ValueError(f"Amended LR row identity mismatch: {index}")
        probabilities = frame.select([f"prob__{label}" for label in CLASSES]).to_numpy()
        validate_probabilities(probabilities, frame.height)
        error = float(np.abs(probabilities.sum(axis=1) - 1).max())
        if not np.isclose(record["max_probability_sum_error"], error, atol=1e-15, rtol=0):
            raise ValueError(f"Amended LR probability audit mismatch: {index}")
        predicted = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
        if not np.array_equal(frame["predicted_label"].to_numpy(), predicted):
            raise ValueError(f"Amended LR predicted-label mismatch: {index}")
        constants = {
            "feature_contract_sha256": prepared["feature_contract_sha256"],
            "sample_manifest_sha256": prepared["sample_manifest_sha256"],
            "model_id": "LR",
            "run_id": amendment["run_id"],
            "seed": SEED,
            "protocol_amendment_sha256": amendment_sha,
            "input_bundle_manifest_sha256": bundle_sha,
            "software_versions": software,
        }
        for column, value in constants.items():
            if frame[column].n_unique() != 1 or frame[column][0] != value:
                raise ValueError(f"Amended LR metadata mismatch: {column}")
        for partition, count in frame.group_by("_partition").len().rows():
            partition_rows[str(partition)] += int(count)
        grouped = frame.group_by(
            "_partition", "_canonical_device_id", "_label", "predicted_label"
        ).len()
        for partition, device, true, prediction, count in grouped.rows():
            matrices[(str(partition), str(device))][labels[str(true)], labels[str(prediction)]] += (
                int(count)
            )
    expected_partitions = {
        key: amendment["expected_rows_by_partition"][key] for key in ("calibration", "test")
    }
    if dict(partition_rows) != expected_partitions:
        raise ValueError("Amended LR partition row count mismatch")
    if read_json(directory / "metrics.json") != metrics_from_matrices(dict(matrices)):
        raise ValueError("Amended LR point metrics mismatch")
    result = {
        "amendment_id": amendment["amendment_id"],
        "fit_status": fit["status"],
        "iterations": fit["iterations"],
        "model_sha256": fit["model_sha256"],
        "rows_by_partition": dict(partition_rows),
        "run_config_sha256": sha256(directory / "run_config.json"),
        "status": "passed",
    }
    write_json(output_root / "verification.json", result)
    return result


def run(repo_root: Path, bundle_root: Path, output_root: Path) -> dict[str, Any]:
    _, prepared, config = validate_bundle(bundle_root)
    amendment = read_json(repo_root / AMENDMENT_RELATIVE)
    validate_solver_amendment(repo_root, bundle_root, amendment, prepared)
    output_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_root).free < amendment["minimum_output_free_bytes"]:
        raise ValueError("Insufficient free disk for amended LR outputs")
    guard_manifest(
        output_root / "input_contract.json",
        {
            "amendment_sha256": sha256(repo_root / AMENDMENT_RELATIVE),
            "amendment_report_sha256": amendment["amendment_report_sha256"],
            "base_bundle_manifest_sha256": amendment["base_bundle_manifest_sha256"],
            "prepared_sha256": sha256(bundle_root / BASELINE_RELATIVE / "prepared.json"),
        },
    )
    processor, estimator, directory, accepted = fit_lr(
        repo_root, bundle_root, output_root, amendment, prepared, config
    )
    if not accepted:
        result = {
            "acceptance_status": "blocked",
            "amendment_id": amendment["amendment_id"],
            "next_action": "preserve the failure and return to Human Gate 3",
        }
        write_json(output_root / "summary.json", result)
        return result
    export_lr(
        repo_root, bundle_root, output_root, amendment, prepared, processor, estimator, directory
    )
    verification = verify(repo_root, bundle_root, output_root, amendment, prepared, config)
    result = {
        "acceptance_status": "passed",
        "amendment_id": amendment["amendment_id"],
        "fit_status": verification["fit_status"],
        "verification_status": verification["status"],
    }
    write_json(output_root / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    bundle_root = args.bundle_root.resolve()
    output_root = args.output_root.resolve()
    _, prepared, config = validate_bundle(bundle_root)
    amendment = read_json(repo_root / AMENDMENT_RELATIVE)
    validate_solver_amendment(repo_root, bundle_root, amendment, prepared)
    result = (
        verify(repo_root, bundle_root, output_root, amendment, prepared, config)
        if args.verify_only
        else run(repo_root, bundle_root, output_root)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
