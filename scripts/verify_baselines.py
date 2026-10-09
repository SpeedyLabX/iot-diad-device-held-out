"""Reconcile BASE-002 manifests, every exported row, and independent point metrics."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, f1_score, recall_score

from iot_diad.modeling.baselines import (
    CAP,
    CLASSES,
    CONTRACT_PATHS,
    MODELS,
    SEED,
    make_transform_contract,
    object_hash,
    priorities,
    sha256,
    versions,
    write_json,
)
from iot_diad.modeling.data import METADATA_COLUMNS
from iot_diad.modeling.preprocessing import device_set_hash


def verify_sample(output: Path, prepared: dict[str, Any], config: dict[str, Any]) -> None:
    sample = pl.read_parquet(output / "training_sample.parquet")
    manifest = pl.read_csv(
        output / "training_sample_manifest.csv",
        schema_overrides={"_priority": pl.UInt64, "_row_index": pl.UInt32},
    )
    assert sample.select(manifest.columns).equals(manifest)
    assert sha256(output / "training_sample.parquet") == prepared["training_sample_sha256"]
    assert sha256(output / "training_sample_manifest.csv") == prepared["sample_manifest_sha256"]
    assert sample.height == prepared["training_rows"] <= 640_000
    assert sample.select("_capture_id", "_row_index").n_unique() == sample.height
    assert sample["_partition"].unique().to_list() == ["train"]
    cells = {
        (d, y): n for d, y, n in sample.group_by("_canonical_device_id", "_label").len().rows()
    }
    for cell in prepared["cells"]:
        if cell["partition"] == "train":
            assert cells[(cell["device"], cell["label"])] == min(CAP, cell["rows"])
    assert prepared["feature_contract_sha256"] == object_hash(config["features"])
    assert sample["_canonical_device_id"].n_unique() == 40
    # Recombine capture-local candidates without calling the production bottom_k helper.
    pool = None
    for directory in sorted((output / "cache").iterdir()):
        audit = json.loads((directory / "complete.json").read_text())
        path = directory / "candidates.parquet"
        assert sha256(path) == audit["candidates_sha256"]
        part = pl.read_parquet(path).select(manifest.columns)
        for capture in part.partition_by("_capture_id"):
            expected = priorities(
                capture["_capture_id"][0], capture["_row_index"].to_numpy().astype(np.uint64)
            )
            np.testing.assert_array_equal(capture["_priority"].to_numpy(), expected)
        pool = pl.concat([pool, part]) if pool is not None else part
        pool = pool.sort("_priority", "_capture_id", "_row_index").filter(
            pl.int_range(pl.len()).over("_canonical_device_id", "_label") < CAP
        )
    assert pool is not None
    assert pool.sort("_capture_id", "_row_index").equals(manifest)


def verify_transform_cache(
    root: Path, output: Path, prepared: dict[str, Any], run_id: str
) -> dict[str, Any]:
    directory = output / run_id
    run_config = json.loads((directory / "run_config.json").read_text())
    cache_record = json.loads((directory / "transform_cache.json").read_text())
    processor, _ = joblib.load(directory / "model.joblib")
    contract = make_transform_contract(
        root,
        processor,
        run_config["sample_manifest_sha256"],
        prepared["feature_contract_sha256"],
    )
    assert cache_record == {"cache_key": object_hash(contract), **contract}
    cache_key = cache_record["cache_key"]
    total_rows = 0
    output_features = processor.audit()["output_features"]
    chunk_count = len(prepared["evaluation_chunks"])
    probe_indices = {0, chunk_count // 2, chunk_count - 1}
    for index, source in enumerate(prepared["evaluation_chunks"]):
        path = output / "transformed" / cache_key / f"{source['sha256']}.npy"
        receipt = json.loads(path.with_suffix(".json").read_text())
        assert receipt == {
            "columns": output_features,
            "dtype": "float64",
            "input_sha256": source["sha256"],
            "processor_cache_key": cache_key,
            "rows": source["rows"],
            "sha256": sha256(path),
        }
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        assert values.shape == (source["rows"], output_features)
        assert values.dtype == np.float64 and np.isfinite(values).all()
        if index in probe_indices:
            raw = pl.read_parquet(output / source["path"])
            np.testing.assert_array_equal(values, processor.transform(raw))
        total_rows += source["rows"]
    assert total_rows == (
        prepared["rows_by_partition"]["calibration"] + prepared["rows_by_partition"]["test"]
    )
    return {
        "cache_key": cache_key,
        "chunks": chunk_count,
        "rows": total_rows,
        "direct_transform_probes": len(probe_indices),
    }


def independent_metrics(matrix: np.ndarray[Any, Any]) -> tuple[float, float, np.ndarray[Any, Any]]:
    true = np.repeat(np.arange(8), 8)
    predicted = np.tile(np.arange(8), 8)
    weights = matrix.ravel()
    present = np.flatnonzero(matrix.sum(axis=1))
    return (
        float(
            f1_score(
                true,
                predicted,
                labels=present,
                average="macro",
                sample_weight=weights,
                zero_division=0,
            )
        ),
        float(accuracy_score(true, predicted, sample_weight=weights)),
        recall_score(
            true,
            predicted,
            labels=np.arange(8),
            average=None,
            sample_weight=weights,
            zero_division=0,
        ),
    )


def verify_run(root: Path, output: Path, run_id: str, prepared: dict[str, Any]) -> dict[str, Any]:
    directory = output / run_id
    config = json.loads((directory / "run_config.json").read_text())
    exports = json.loads((directory / "exports.json").read_text())
    fit = json.loads((directory / "fit.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    assert config["class_order"] == metrics["class_order"] == list(CLASSES)
    assert sha256(directory / "model.joblib") == fit["model_sha256"]
    assert sha256(directory / "training_sample_manifest.csv") == config["sample_manifest_sha256"]
    assert config["feature_contract_sha256"] == prepared["feature_contract_sha256"]
    assert config["software_versions"] == versions()
    assert fit["execution_code_sha256"] == config["execution_code_sha256"]
    for source_path, digest in config["execution_code_sha256"].items():
        assert sha256(root / source_path) == digest
    processor, estimator = joblib.load(directory / "model.joblib")
    assert json.loads(json.dumps(estimator.get_params())) == config["estimator"]
    np.testing.assert_array_equal(estimator.classes_, np.arange(8))
    assert processor.audit() == fit["preprocessor"]
    assert config["seed"] == SEED
    assert fit["preprocessor"]["fit_partition"] == "train"
    assert fit["preprocessor"]["training_device_count"] == 40
    if config["cap"] == CAP:
        assert config["sample_manifest_sha256"] == prepared["sample_manifest_sha256"]
    else:
        sample = pl.read_csv(
            output / "training_sample_manifest.csv", schema_overrides={"_priority": pl.UInt64}
        )
        expected_sample = (
            sample.sort("_priority", "_capture_id", "_row_index")
            .filter(pl.int_range(pl.len()).over("_canonical_device_id", "_label") < 1000)
            .sort("_capture_id", "_row_index")
        )
        actual = pl.read_csv(
            directory / "training_sample_manifest.csv", schema_overrides={"_priority": pl.UInt64}
        )
        assert expected_sample.equals(actual)
    split = pl.read_csv(root / CONTRACT_PATHS[1])
    train_devices = set(split.filter(pl.col("partition") == "train")["canonical_device_id"])
    assert fit["preprocessor"]["training_device_set_sha256"] == device_set_hash(train_devices)
    assignments = {
        r["local_identifier"]: (r["canonical_device_id"], r["partition"])
        for r in split.iter_rows(named=True)
    }
    exclusions = set()
    with (root / CONTRACT_PATHS[2]).open() as handle:
        for row in csv.DictReader(handle):
            exclusions.add((row["capture_id"], int(row["row_index"])))
    labels = {label: i for i, label in enumerate(CLASSES)}
    matrices: dict[tuple[str, str], Any] = defaultdict(lambda: np.zeros((8, 8), dtype=np.int64))
    totals: Counter[str] = Counter()
    expected_chunks = prepared["evaluation_chunks"]
    assert len(exports["chunks"]) == len(expected_chunks)
    max_error = 0.0
    last_rows: dict[str, int] = {}
    for i, (chunk, source) in enumerate(zip(exports["chunks"], expected_chunks, strict=True)):
        path = output / chunk["path"]
        assert sha256(path) == chunk["sha256"]
        assert chunk["run_config_sha256"] == sha256(directory / "run_config.json")
        source_path = output / source["path"]
        assert sha256(source_path) == chunk["input_sha256"] == source["sha256"]
        frame = pl.read_parquet(path)
        raw = pl.read_parquet(source_path, columns=list(METADATA_COLUMNS))
        assert frame.select(METADATA_COLUMNS).equals(raw)
        assert frame.height == chunk["rows"] == source["rows"]
        assert [c for c in frame.columns if c.startswith("prob__")] == [
            f"prob__{c}" for c in CLASSES
        ]
        prob = frame.select(f"prob__{c}" for c in CLASSES).to_numpy()
        assert np.isfinite(prob).all() and (prob >= 0).all() and (prob <= 1).all()
        error = float(np.max(np.abs(prob.sum(axis=1) - 1)))
        max_error = max(max_error, error)
        assert error <= 1e-8
        np.testing.assert_array_equal(
            frame["predicted_label"].to_numpy(), np.asarray(CLASSES)[prob.argmax(axis=1)]
        )
        for column, expected in (
            ("model_id", config["model_id"]),
            ("run_id", run_id),
            ("seed", SEED),
            ("feature_contract_sha256", prepared["feature_contract_sha256"]),
            ("sample_manifest_sha256", config["sample_manifest_sha256"]),
            ("software_versions", json.dumps(versions(), sort_keys=True)),
        ):
            assert frame[column].unique().to_list() == [expected]
        capture = frame["_capture_id"][0]
        assert frame["_capture_id"].n_unique() == 1
        indices = frame["_row_index"].to_numpy()
        assert indices[0] > last_rows.get(capture, -1)
        assert np.all(np.diff(indices.astype(np.int64)) > 0)
        last_rows[capture] = int(indices[-1])
        assert not any((capture, int(index)) in exclusions for index in indices)
        for identifier, device, partition in (
            frame.select("_local_identifier", "_canonical_device_id", "_partition").unique().rows()
        ):
            assert assignments[identifier] == (device, partition)
            assert partition in ("calibration", "test")
        for partition, device, true, predicted, count in (
            frame.group_by("_partition", "_canonical_device_id", "_label", "predicted_label")
            .len()
            .rows()
        ):
            matrices[(partition, device)][labels[true], labels[predicted]] += count
            totals[partition] += count
        if i % 100 == 0:
            print(f"verify {run_id}: {i + 1}/{len(expected_chunks)}", flush=True)
    assert dict(totals) == {p: prepared["rows_by_partition"][p] for p in ("calibration", "test")}
    assert exports["rows"] == sum(totals.values())
    stored_devices = {(d["partition"], d["device"]): d for d in metrics["devices"]}
    assert set(stored_devices) == set(matrices)
    for key, matrix in matrices.items():
        stored = stored_devices[key]
        f1, accuracy, recall = independent_metrics(matrix)
        np.testing.assert_array_equal(stored["confusion_matrix"], matrix)
        assert np.isclose(stored["macro_f1"], f1, rtol=0, atol=1e-14)
        assert np.isclose(stored["accuracy"], accuracy, rtol=0, atol=1e-14)
        for j, label in enumerate(CLASSES):
            item = stored["per_class"][label]
            assert item["support"] == matrix[j].sum()
            if matrix[j].sum():
                assert np.isclose(item["recall"], recall[j], rtol=0, atol=1e-14)
            else:
                assert item["recall"] is None and item["f1"] is None
    expected_cells = {
        (c["partition"], c["device"], c["label"]): c["rows"]
        for c in prepared["cells"]
        if c["partition"] != "train"
    }
    actual_cells = {
        (p, d, CLASSES[j]): int(n)
        for (p, d), m in matrices.items()
        for j, n in enumerate(m.sum(axis=1))
        if n
    }
    assert actual_cells == expected_cells
    for partition in ("calibration", "test"):
        entries = sorted((d, m) for (p, d), m in matrices.items() if p == partition)
        assert len(entries) == 14
        scores = np.array([independent_metrics(m)[0] for _, m in entries])
        aggregate = metrics["aggregates"][partition]
        expected_values = {
            "mean_device_macro_f1": scores.mean(),
            "median_device_macro_f1": np.median(scores),
            "min_device_macro_f1": scores.min(),
            "max_device_macro_f1": scores.max(),
            "q25_device_macro_f1": np.quantile(scores, 0.25),
            "q75_device_macro_f1": np.quantile(scores, 0.75),
            "iqr_device_macro_f1": np.quantile(scores, 0.75) - np.quantile(scores, 0.25),
        }
        for metric_key, value in expected_values.items():
            assert np.isclose(aggregate[metric_key], value, rtol=0, atol=1e-14)
        for i, (device, _) in enumerate(entries):
            assert np.isclose(
                aggregate["leave_one_device_out"][device],
                np.delete(scores, i).mean(),
                rtol=0,
                atol=1e-14,
            )
        total = sum((m for _, m in entries), start=np.zeros((8, 8), dtype=np.int64))
        global_f1, global_accuracy, _ = independent_metrics(total)
        assert np.isclose(aggregate["global_row_weighted"]["macro_f1"], global_f1)
        assert np.isclose(aggregate["global_row_weighted"]["accuracy"], global_accuracy)
        for j, label in enumerate(CLASSES):
            recalls = [m[j, j] / m[j].sum() for _, m in entries if m[j].sum()]
            stored = aggregate["device_balanced_per_class_recall"][label]
            assert stored["devices"] == len(recalls)
            assert np.isclose(stored["recall"], np.mean(recalls), rtol=0, atol=1e-14)
    return {
        "run_id": run_id,
        "artifact_integrity": "passed",
        "fit_status": fit["status"],
        "rows_by_partition": dict(totals),
        "max_probability_sum_error": max_error,
        "test_mean_device_macro_f1": metrics["aggregates"]["test"]["mean_device_macro_f1"],
    }


def main() -> None:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--model")
    parser.add_argument("--sample-only", action="store_true")
    args = parser.parse_args()
    output = args.root / "artifacts/experiments/baselines"
    config = json.loads((output / "config.json").read_text())
    prepared = json.loads((output / "prepared.json").read_text())
    amendment = json.loads((args.root / "config/base002_kaggle_rerun.json").read_text())
    assert sha256(output / "config.json") == prepared["config_sha256"]
    for path, digest in config["contracts"].items():
        current_digest = sha256(args.root / path)
        if path == "docs/reports/gate3_protocol.md":
            assert digest == amendment["original_gate3_protocol_sha256"]
            assert current_digest == amendment["amended_gate3_protocol_sha256"]
        else:
            assert current_digest == digest
    gate2 = json.loads((args.root / CONTRACT_PATHS[3]).read_text())
    assert prepared["rows_by_partition"] == gate2["filtered_rows_by_partition"]
    verify_sample(output, prepared, config)
    if args.sample_only:
        print("Training sample verification passed.")
        return
    if not args.model:
        input_audit = json.loads((output / "input_verification.json").read_text())
        assert input_audit["status"] == "passed"
        assert input_audit["rows_by_partition"] == prepared["rows_by_partition"]
    selected = (
        json.loads((output / "sensitivity_selection.json").read_text()) if not args.model else {}
    )
    names = [args.model] if args.model else [*MODELS, "DUMMY", selected["leader"] + "-cap1000"]
    reports = [verify_run(args.root, output, name, prepared) for name in names]
    cache_report = verify_transform_cache(args.root, output, prepared, args.model or "LR")
    accepted = all(report["fit_status"] == "passed" for report in reports)
    report = {
        "artifact_integrity": "passed",
        "acceptance": "passed" if accepted else "failed",
        "runs": reports,
        "transform_cache": cache_report,
        "inference_performed": False,
        "verification_seconds": time.monotonic() - started,
    }
    write_json(
        output / (f"verification-{args.model}.json" if args.model else "verification.json"), report
    )
    print(json.dumps(report, indent=2))
    if not accepted:
        raise SystemExit("Frozen estimator convergence failure: human review required.")


if __name__ == "__main__":
    main()
