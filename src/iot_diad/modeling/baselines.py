"""Frozen BASE-002 execution: bounded raw reads, deterministic sampling, atomic resume."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

from iot_diad.metadata import discover_sources
from iot_diad.modeling.data import (
    METADATA_COLUMNS,
    load_partition_contract,
    select_modeling_rows,
)
from iot_diad.modeling.features import load_packet_feature_spec
from iot_diad.modeling.preprocessing import TrainOnlyPreprocessor

SEED = 2027
CAP = 2000
CHUNK = 50_000
MODELS = ("LR", "ET", "HGB", "RF", "MLP")
CLASSES = ("Benign", "Brute Force", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web-based")
CONTRACT_PATHS = (
    "artifacts/eda/column_roles.csv",
    "artifacts/splits/device_split_manifest.csv",
    "artifacts/splits/cross_partition_duplicate_exclusions.csv",
    "artifacts/splits/gate2_decision_summary.json",
    "docs/reports/gate3_protocol.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_parquet(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.write_parquet(temporary, compression="zstd")
    temporary.replace(path)


def versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        **{
            name: importlib.metadata.version(name)
            for name in ("numpy", "polars", "scikit-learn", "scipy", "joblib", "threadpoolctl")
        },
    }


def estimators() -> dict[str, Any]:
    return {
        "LR": LogisticRegression(C=1, solver="lbfgs", max_iter=300, random_state=SEED),
        "ET": ExtraTreesClassifier(
            n_estimators=300,
            max_features="sqrt",
            max_leaf_nodes=256,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED,
        ),
        "HGB": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            max_leaf_nodes=31,
            l2_regularization=1,
            class_weight="balanced",
            early_stopping=False,
            random_state=SEED,
        ),
        "RF": RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            max_leaf_nodes=256,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=2048,
            max_iter=50,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            random_state=SEED,
        ),
        "DUMMY": DummyClassifier(strategy="prior", random_state=SEED),
    }


def priorities(capture: str, indices: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """SplitMix64 permutation of unique row index XOR a seed/capture SHA256 salt."""
    salt = int.from_bytes(hashlib.sha256(f"{SEED}\0{capture}".encode()).digest()[:8], "little")
    with np.errstate(over="ignore"):
        values = (indices ^ np.uint64(salt)) + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return values ^ (values >> np.uint64(31))


def bottom_k(frame: pl.DataFrame, cap: int) -> pl.DataFrame:
    return (
        frame.sort(["_canonical_device_id", "_label", "_priority", "_capture_id", "_row_index"])
        .group_by(["_canonical_device_id", "_label"], maintain_order=True)
        .head(cap)
        .select(frame.columns)
    )


def guard_manifest(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Resume contract changed: {path}")
    else:
        write_json(path, value)


def prepare(root: Path, output: Path) -> None:
    start = time.monotonic()
    sources = [
        s
        for s in discover_sources(root / "data/raw/CIC-IoT-IDAD-Dataset-2024")
        if s.feature_view == "packet"
    ]
    spec = load_packet_feature_spec(root / CONTRACT_PATHS[0])
    contract = load_partition_contract(root / CONTRACT_PATHS[1], root / CONTRACT_PATHS[2])
    config = {
        "seed": SEED,
        "cap": CAP,
        "chunk_rows": CHUNK,
        "class_order": CLASSES,
        "sampling": (
            "bottom-k SplitMix64(seed/capture SHA256 salt XOR raw row index); ties by row key"
        ),
        "contracts": {p: sha256(root / p) for p in CONTRACT_PATHS},
        "features": {"numeric": spec.numeric, "categorical": spec.categorical},
        "software_versions": versions(),
        "estimators": {name: estimator.get_params() for name, estimator in estimators().items()},
        "sources": [
            {
                "capture_id": s.capture_id,
                "size_bytes": s.size_bytes,
                "mtime_ns": s.path.stat().st_mtime_ns,
            }
            for s in sources
        ],
    }
    # JSON round trip normalizes tuples before immutable resume comparison.
    guard_manifest(output / "config.json", json.loads(json.dumps(config)))
    if (output / "prepared.json").exists():
        return
    for source_index, source in enumerate(sources):
        directory = output / "cache" / f"{source_index:03d}"
        completion = directory / "complete.json"
        if completion.exists():
            continue
        directory.mkdir(parents=True, exist_ok=True)
        candidates: pl.DataFrame | None = None
        counts: Counter[str] = Counter()
        cell_counts: Counter[tuple[str, str, str]] = Counter()
        chunks = []
        offset = 0
        scan = pl.scan_csv(source.path, infer_schema=False).select("device_mac", *spec.columns)
        for chunk_index, raw in enumerate(scan.collect_batches(chunk_size=CHUNK)):
            frame = select_modeling_rows(
                raw, source.capture_id, source.coarse_label, spec, contract, row_offset=offset
            )
            offset += raw.height
            for row in frame.group_by("_partition", "_canonical_device_id", "_label").len().rows():
                cell_counts[(str(row[0]), str(row[1]), str(row[2]))] += int(row[3])
                counts[str(row[0])] += int(row[3])
            train = frame.filter(pl.col("_partition") == "train")
            if train.height:
                train = train.with_columns(
                    pl.Series(
                        "_priority",
                        priorities(
                            source.capture_id, train["_row_index"].to_numpy().astype(np.uint64)
                        ),
                    )
                )
                candidates = bottom_k(
                    pl.concat([candidates, train]) if candidates is not None else train, CAP
                )
            evaluation = frame.filter(pl.col("_partition") != "train")
            if evaluation.height:
                path = directory / f"eval-{chunk_index:05d}.parquet"
                write_parquet(path, evaluation)
                chunks.append(
                    {
                        "path": str(path.relative_to(output)),
                        "rows": evaluation.height,
                        "sha256": sha256(path),
                    }
                )
        if candidates is None:
            raise ValueError(f"No training rows in capture {source.capture_id}")
        candidate_path = directory / "candidates.parquet"
        write_parquet(candidate_path, candidates)
        write_json(
            completion,
            {
                "capture_id": source.capture_id,
                "source_sha256": sha256(source.path),
                "raw_rows": offset,
                "counts": counts,
                "evaluation_chunks": chunks,
                "candidates_sha256": sha256(candidate_path),
                "cells": [
                    {"partition": k[0], "device": k[1], "label": k[2], "rows": v}
                    for k, v in sorted(cell_counts.items())
                ],
            },
        )
        print(f"prepared {source_index + 1}/{len(sources)}: {offset} raw rows", flush=True)
    candidates = None
    totals: Counter[str] = Counter()
    cells: Counter[tuple[str, str, str]] = Counter()
    chunks = []
    sources_audit = []
    for source_index in range(len(sources)):
        directory = output / "cache" / f"{source_index:03d}"
        audit = json.loads((directory / "complete.json").read_text())
        assert sha256(directory / "candidates.parquet") == audit["candidates_sha256"]
        frame = pl.read_parquet(directory / "candidates.parquet")
        candidates = bottom_k(
            pl.concat([candidates, frame]) if candidates is not None else frame, CAP
        )
        totals.update(audit["counts"])
        for cell in audit["cells"]:
            cells[(cell["partition"], cell["device"], cell["label"])] += cell["rows"]
        chunks.extend(audit["evaluation_chunks"])
        sources_audit.append({k: audit[k] for k in ("capture_id", "source_sha256", "raw_rows")})
    expected = json.loads((root / CONTRACT_PATHS[3]).read_text())["filtered_rows_by_partition"]
    if dict(totals) != expected:
        raise ValueError(f"Population mismatch: {totals} versus {expected}")
    assert candidates is not None
    sample = candidates.sort(["_capture_id", "_row_index"])
    write_parquet(output / "training_sample.parquet", sample)
    manifest = sample.select(*METADATA_COLUMNS, "_priority")
    manifest.write_csv(output / "training_sample_manifest.csv")
    sample_cells = {
        (r[0], r[1]): r[2] for r in sample.group_by("_canonical_device_id", "_label").len().rows()
    }
    for (partition, device, label), count in cells.items():
        if partition == "train" and sample_cells[(device, label)] != min(CAP, count):
            raise ValueError("Training cell cap mismatch")
    write_json(
        output / "prepared.json",
        {
            "status": "passed",
            "config_sha256": sha256(output / "config.json"),
            "sample_manifest_sha256": sha256(output / "training_sample_manifest.csv"),
            "training_sample_sha256": sha256(output / "training_sample.parquet"),
            "feature_contract_sha256": object_hash(config["features"]),
            "training_rows": sample.height,
            "rows_by_partition": dict(totals),
            "evaluation_chunks": chunks,
            "sources": sources_audit,
            "cells": [
                {"partition": k[0], "device": k[1], "label": k[2], "rows": v}
                for k, v in sorted(cells.items())
            ],
            "preparation_seconds": time.monotonic() - start,
        },
    )


def cached_evaluation_features(
    output: Path,
    source: dict[str, Any],
    processor: TrainOnlyPreprocessor,
    cache_key: str,
) -> tuple[pl.DataFrame, NDArray[np.float64]]:
    """Share exactly identical fitted transforms, never fit on evaluation rows."""
    source_path = output / source["path"]
    if sha256(source_path) != source["sha256"]:
        raise ValueError(f"Evaluation cache hash mismatch: {source_path}")
    path = output / "transformed" / cache_key / f"{source['sha256']}.npy"
    receipt = path.with_suffix(".json")
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if record["sha256"] != sha256(path) or record["input_sha256"] != source["sha256"]:
            raise ValueError(f"Transformed cache hash mismatch: {path}")
        frame = pl.read_parquet(source_path, columns=list(METADATA_COLUMNS))
        features = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        frame = pl.read_parquet(source_path)
        features = processor.transform(frame)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, features, allow_pickle=False)
        temporary.replace(path)
        write_json(
            receipt,
            {
                "sha256": sha256(path),
                "input_sha256": source["sha256"],
                "processor_cache_key": cache_key,
                "rows": frame.height,
                "columns": features.shape[1],
                "dtype": str(features.dtype),
            },
        )
    if features.shape != (source["rows"], processor.audit()["output_features"]):
        raise ValueError("Transformed cache shape mismatch")
    if features.dtype != np.float64 or not np.isfinite(features).all():
        raise ValueError("Transformed cache is not finite float64")
    return frame.select(METADATA_COLUMNS), features


def validate_probabilities(probabilities: NDArray[np.float64], rows: int) -> None:
    if probabilities.shape != (rows, len(CLASSES)):
        raise ValueError("Probability shape/class order mismatch")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("Invalid probability values")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-8, rtol=0):
        raise ValueError("Probability sums differ from one")


def make_transform_contract(
    root: Path,
    processor: TrainOnlyPreprocessor,
    sample_manifest_sha256: str,
    feature_contract_sha256: str,
) -> dict[str, Any]:
    preprocessing_path = root / "src/iot_diad/modeling/preprocessing.py"
    return {
        "sample_manifest_sha256": sample_manifest_sha256,
        "feature_contract_sha256": feature_contract_sha256,
        "preprocessor_audit": processor.audit(),
        "preprocessing_source_sha256": sha256(preprocessing_path)
        if preprocessing_path.exists()
        else "synthetic-test",
        "software_versions": versions(),
    }


def run_model(root: Path, output: Path, name: str, cap: int = CAP) -> None:
    prepared = json.loads((output / "prepared.json").read_text())
    config = json.loads((output / "config.json").read_text())
    if config["software_versions"] != versions():
        raise ValueError("Software changed since preparation")
    if sha256(output / "config.json") != prepared["config_sha256"]:
        raise ValueError("Configuration hash mismatch")
    for path, digest in config["contracts"].items():
        if sha256(root / path) != digest:
            raise ValueError(f"Changed source contract: {path}")
    current_params = json.loads(json.dumps(estimators()[name].get_params()))
    if current_params != config["estimators"][name]:
        raise ValueError("Estimator configuration changed since protocol freeze")
    run_id = name if cap == CAP else f"{name}-cap{cap}"
    directory = output / run_id
    directory.mkdir(exist_ok=True)
    if sha256(output / "training_sample.parquet") != prepared["training_sample_sha256"]:
        raise ValueError("Training sample hash mismatch")
    sample = pl.read_parquet(output / "training_sample.parquet")
    if cap != CAP:
        sample = bottom_k(sample, cap).sort(["_capture_id", "_row_index"])
    manifest_path = directory / "training_sample_manifest.csv"
    sample.select(*METADATA_COLUMNS, "_priority").write_csv(manifest_path)
    sample_hash = sha256(manifest_path)
    execution_code = {
        str(path.relative_to(root)): sha256(path)
        for path in sorted((root / "src/iot_diad/modeling").glob("*.py"))
    }
    run_config = {
        "model_id": name,
        "execution_code_sha256": execution_code,
        "run_id": run_id,
        "cap": cap,
        "seed": SEED,
        "sample_manifest_sha256": sample_hash,
        "feature_contract_sha256": prepared["feature_contract_sha256"],
        "config_sha256": prepared["config_sha256"],
        "software_versions": versions(),
        "class_order": list(CLASSES),
        "estimator": config["estimators"][name],
    }
    guard_manifest(directory / "run_config.json", run_config)
    fit_path = directory / "fit.json"
    model_path = directory / "model.joblib"
    if not fit_path.exists():
        started = time.monotonic()
        processor = TrainOnlyPreprocessor(load_packet_feature_spec(root / CONTRACT_PATHS[0]))
        print(f"{run_id}: preprocessing {sample.height} training rows", flush=True)
        features = processor.fit_transform(sample)
        encoder = LabelEncoder().fit(sample["_label"].to_numpy())
        if tuple(encoder.classes_) != CLASSES:
            raise ValueError("Unexpected fitted class order")
        estimator = estimators()[name]
        print(f"{run_id}: fitting {features.shape}", flush=True)
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
                "status": "convergence_failure" if convergence_failure else "passed",
                "warnings": messages,
                "training_rows": sample.height,
                "preprocessor": processor.audit(),
                "model_sha256": sha256(model_path),
                "fit_seconds": time.monotonic() - started,
                "iterations": np.asarray(getattr(estimator, "n_iter_", [])).tolist(),
                "training_only_validation": name == "MLP",
                "execution_code_sha256": {
                    str(path.relative_to(root)): sha256(path)
                    for path in sorted((root / "src/iot_diad/modeling").glob("*.py"))
                },
            },
        )
        del features
        print(f"{run_id}: fit saved; convergence_failure={convergence_failure}", flush=True)
    else:
        fit = json.loads(fit_path.read_text())
        if sha256(model_path) != fit["model_sha256"]:
            raise ValueError("Saved model hash mismatch")
        processor, estimator = joblib.load(model_path)
    del sample
    expected_classes = np.arange(len(CLASSES))
    if not np.array_equal(estimator.classes_, expected_classes):
        raise ValueError("Estimator class order mismatch")
    transform_contract = make_transform_contract(
        root, processor, sample_hash, prepared["feature_contract_sha256"]
    )
    cache_key = object_hash(transform_contract)
    guard_manifest(
        directory / "transform_cache.json",
        {
            "cache_key": cache_key,
            **transform_contract,
        },
    )
    started = time.monotonic()
    chunks = []
    for index, source in enumerate(prepared["evaluation_chunks"]):
        destination = directory / "probabilities" / f"{index:05d}.parquet"
        receipt = destination.with_suffix(".json")
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if (
                record["sha256"] != sha256(destination)
                or record["input_sha256"] != source["sha256"]
                or record["run_config_sha256"] != sha256(directory / "run_config.json")
            ):
                raise ValueError(f"Resume chunk hash mismatch: {destination}")
        else:
            with threadpool_limits(limits=8):
                frame, features = cached_evaluation_features(output, source, processor, cache_key)
                probabilities = np.asarray(estimator.predict_proba(features), dtype=np.float64)
            validate_probabilities(probabilities, frame.height)
            exported = frame.select(METADATA_COLUMNS).with_columns(
                **{f"prob__{label}": probabilities[:, i] for i, label in enumerate(CLASSES)},
                predicted_label=pl.Series(np.asarray(CLASSES)[probabilities.argmax(axis=1)]),
                model_id=pl.lit(name),
                run_id=pl.lit(run_id),
                seed=pl.lit(SEED),
                feature_contract_sha256=pl.lit(prepared["feature_contract_sha256"]),
                sample_manifest_sha256=pl.lit(sample_hash),
                software_versions=pl.lit(json.dumps(versions(), sort_keys=True)),
            )
            write_parquet(destination, exported)
            record = {
                "path": str(destination.relative_to(output)),
                "rows": frame.height,
                "sha256": sha256(destination),
                "input_sha256": source["sha256"],
                "run_config_sha256": sha256(directory / "run_config.json"),
                "max_probability_sum_error": float(np.abs(probabilities.sum(axis=1) - 1).max()),
            }
            write_json(receipt, record)
        chunks.append(record)
        if index % 50 == 0:
            print(
                f"{run_id}: predicted {index + 1}/{len(prepared['evaluation_chunks'])}", flush=True
            )
    write_json(
        directory / "exports.json",
        {
            "status": "complete",
            "chunks": chunks,
            "rows": sum(c["rows"] for c in chunks),
            "prediction_seconds_this_invocation": time.monotonic() - started,
            "fit_status": json.loads(fit_path.read_text())["status"],
        },
    )
    summarize(directory)


def metric_rows(matrix: NDArray[np.int64]) -> dict[str, Any]:
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    diagonal = matrix.diagonal()
    denominator = support + predicted
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros(len(CLASSES)), where=denominator > 0)
    recall = np.divide(diagonal, support, out=np.zeros(len(CLASSES)), where=support > 0)
    present = support > 0
    return {
        "rows": int(matrix.sum()),
        "macro_f1": float(f1[present].mean()),
        "accuracy": float(diagonal.sum() / matrix.sum()),
        "per_class": {
            label: {
                "support": int(support[i]),
                "recall": float(recall[i]) if present[i] else None,
                "f1": float(f1[i]) if present[i] else None,
            }
            for i, label in enumerate(CLASSES)
        },
        "confusion_matrix": matrix.tolist(),
    }


def summarize(directory: Path) -> None:
    matrices: dict[tuple[str, str], NDArray[np.int64]] = {}
    labels = {label: i for i, label in enumerate(CLASSES)}
    manifest = json.loads((directory / "exports.json").read_text())
    for chunk in manifest["chunks"]:
        frame = pl.read_parquet(directory.parent / chunk["path"])
        grouped = frame.group_by(
            "_partition", "_canonical_device_id", "_label", "predicted_label"
        ).len()
        for partition, device, true, predicted, count in grouped.rows():
            matrix = matrices.setdefault((partition, device), np.zeros((8, 8), dtype=np.int64))
            matrix[labels[true], labels[predicted]] += count
    devices = [
        {"partition": key[0], "device": key[1], **metric_rows(matrix)}
        for key, matrix in sorted(matrices.items())
    ]
    aggregates = {}
    for partition in ("calibration", "test"):
        entries = [d for d in devices if d["partition"] == partition]
        scores = np.asarray([d["macro_f1"] for d in entries])
        total = sum(
            (m for k, m in matrices.items() if k[0] == partition),
            start=np.zeros((8, 8), dtype=np.int64),
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
                                d["per_class"][label]["recall"]
                                for d in entries
                                if d["per_class"][label]["support"] > 0
                            ]
                        )
                    ),
                    "devices": sum(d["per_class"][label]["support"] > 0 for d in entries),
                }
                for label in CLASSES
            },
            "leave_one_device_out": {
                d["device"]: float(np.delete(scores, i).mean()) if len(scores) > 1 else None
                for i, d in enumerate(entries)
            },
        }
    write_json(
        directory / "metrics.json",
        {
            "class_order": CLASSES,
            "devices": devices,
            "aggregates": aggregates,
            "metric_scope": (
                "descriptive point estimates only; no conformal calibration or inference"
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "sensitivity"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--model", choices=(*MODELS, "DUMMY"))
    args = parser.parse_args()
    output = args.root / "artifacts/experiments/baselines"
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".runner.lock"
    # flock is released by the OS even after a killed process; no stale-lock manual deletion.
    import fcntl

    with lock.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.write(str(os.getpid()))
        handle.flush()
        if args.stage == "prepare":
            prepare(args.root, output)
        elif args.stage == "run":
            for name in (args.model,) if args.model else (*MODELS, "DUMMY"):
                try:
                    run_model(args.root, output, name)
                except Exception as exc:
                    write_json(
                        output / name / "failure.json",
                        {
                            "exception": type(exc).__name__,
                            "message": str(exc),
                            "time": time.time(),
                            "configuration_changed": False,
                        },
                    )
                    raise
        else:
            ranking = []
            for name in MODELS:
                metrics = json.loads((output / name / "metrics.json").read_text())
                ranking.append((metrics["aggregates"]["test"]["mean_device_macro_f1"], name))
            ranking.sort(key=lambda item: (-item[0], item[1]))
            leader = ranking[0][1]
            write_json(
                output / "sensitivity_selection.json",
                {
                    "leader": leader,
                    "ranking": ranking,
                    "rule": (
                        "highest primary test point metric; model-ID tie break; descriptive only"
                    ),
                    "cap": 1000,
                    "changes_to_primary_model": False,
                },
            )
            run_model(args.root, output, leader, 1000)


if __name__ == "__main__":
    main()
