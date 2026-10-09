"""Independent small-sample oracles and full synthetic CONF-001 execution."""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from numpy.typing import NDArray

from iot_diad.modeling.baselines import CLASSES, MODELS, sha256, write_json, write_parquet
from iot_diad.modeling.conformal import (
    BASE,
    OUTPUT,
    PROBS,
    membership,
    quantile,
    read_json,
    run,
)


def oracle(arrays: list[NDArray[np.float64]], hierarchical: bool) -> float:
    nonempty = [a for a in arrays if len(a)]
    for score in sorted({float(s) for a in nonempty for s in a}):
        if hierarchical:
            mass = sum(
                (
                    Fraction(int(np.count_nonzero(a <= score)), (len(nonempty) + 1) * len(a))
                    for a in nonempty
                ),
                Fraction(),
            )
        else:
            mass = Fraction(
                sum(int(np.count_nonzero(a <= score)) for a in nonempty),
                sum(len(a) for a in nonempty) + 1,
            )
        if mass >= Fraction(9, 10):
            return score
    return math.inf


@pytest.mark.parametrize("hierarchical", [True, False])
def test_exact_quantile_ties_unequal_support(hierarchical: bool) -> None:
    rng = np.random.default_rng(27)
    for k in [0, 1, 8, 9, 10, 13, 14]:
        arrays = [np.sort(rng.integers(0, 11, size=3 + i * 7) / 10) for i in range(k)]
        assert quantile(arrays, hierarchical) == oracle(arrays, hierarchical)
    arrays = [np.array([0.0])] * 9
    assert quantile(arrays, hierarchical) == 0.0  # Exactly 9/10 finite mass.


def test_device_balance_differs_from_row_pooling() -> None:
    arrays = [np.zeros(10000)] + [np.array([0.8])] * 9
    assert quantile(arrays, True) == 0.8
    assert quantile(arrays, False) == 0.0
    assert math.isinf(quantile([np.zeros(10000)] * 8, True))


def test_membership_inclusive_threshold_and_infinity() -> None:
    p = np.array([[1.0, 0, 0, 0, 0, 0, 0, 0], [0.125] * 8])
    t = np.array([0, 1, math.inf, 0.875, 0, 0, 0, 0])
    np.testing.assert_array_equal(membership(p, t), [7, 14])


def synthetic_root(root: Path) -> None:
    base = root / BASE
    base.mkdir(parents=True)
    (base / "training_sample_manifest.csv").write_text("synthetic")
    for relative in [
        "docs/reports/gate3_protocol.md",
        "docs/reports/gate3_effective_protocol.md",
        "artifacts/splits/gate2_decision_summary.json",
    ]:
        p = root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("synthetic test fixture\n")
    split = [
        {
            "local_identifier": f"local{i}",
            "canonical_device_id": f"device{i}",
            "partition": "calibration" if i < 10 else "test" if i == 10 else "train",
        }
        for i in range(12)
    ]
    pl.DataFrame(split).write_csv(root / "artifacts/splits/device_split_manifest.csv")
    (root / "artifacts/splits/cross_partition_duplicate_exclusions.csv").write_text(
        "capture_id,row_index\n"
    )
    metadata = []
    cells = []
    for i in range(11):
        for y, label in enumerate(CLASSES):
            metadata.append(
                {
                    "_partition": split[i]["partition"],
                    "_canonical_device_id": f"device{i}",
                    "_local_identifier": f"local{i}",
                    "_capture_id": "capture",
                    "_row_index": i * 8 + y,
                    "_label": label,
                }
            )
            cells.append(
                {
                    "partition": split[i]["partition"],
                    "device": f"device{i}",
                    "label": label,
                    "rows": 1,
                }
            )
    frame = pl.DataFrame(metadata).with_columns(pl.col("_row_index").cast(pl.UInt32))
    source_path = base / "cache/000.parquet"
    write_parquet(source_path, frame)
    source = {"path": "cache/000.parquet", "sha256": sha256(source_path), "rows": frame.height}
    write_json(base / "prepared.json", {"evaluation_chunks": [source], "cells": cells})
    models = []
    for name in MODELS:
        directory = base / name
        directory.mkdir()
        (directory / "model.joblib").write_bytes(b"synthetic-no-fit")
        (directory / "training_sample_manifest.csv").write_text("synthetic")
        sample_hash = sha256(directory / "training_sample_manifest.csv")
        config = {"run_id": name, "model_id": name, "class_order": list(CLASSES)}
        write_json(directory / "run_config.json", config)
        write_json(directory / "fit.json", {"status": "passed"})
        rng = np.random.default_rng(41)
        probabilities = rng.uniform(size=(len(metadata), 8))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        exported = frame.with_columns(
            [pl.Series(c, probabilities[:, y]) for y, c in enumerate(PROBS)]
        ).with_columns(
            pl.lit(name).alias("model_id"),
            pl.lit(name).alias("run_id"),
            pl.lit(sample_hash).alias("sample_manifest_sha256"),
        )
        path = directory / "probabilities/00000.parquet"
        write_parquet(path, exported)
        receipt = {
            "path": f"{name}/probabilities/00000.parquet",
            "rows": frame.height,
            "sha256": sha256(path),
            "input_sha256": source["sha256"],
            "run_config_sha256": sha256(directory / "run_config.json"),
            "max_probability_sum_error": float(np.abs(probabilities.sum(axis=1) - 1).max()),
        }
        write_json(path.with_suffix(".json"), receipt)
        write_json(
            directory / "exports.json",
            {"chunks": [receipt], "fit_status": "passed", "status": "complete"},
        )
        model: dict[str, Any] = {
            "model_id": name,
            "run_id": name,
            "primary_learned_model": True,
            "class_order": list(CLASSES),
            "artifact_directory": str(directory.relative_to(root)),
        }
        for file, key in [
            ("model.joblib", "model_sha256"),
            ("fit.json", "fit_sha256"),
            ("run_config.json", "run_config_sha256"),
            ("exports.json", "exports_sha256"),
            ("training_sample_manifest.csv", "sample_manifest_sha256"),
        ]:
            model[key] = sha256(directory / file)
        models.append(model)
    write_json(
        base / "accepted_model_registry.json",
        {
            "status": "passed",
            "models": models,
            "class_order": list(CLASSES),
            "evaluation_rows": {"calibration": 80, "test": 8},
        },
    )


def test_complete_run_resume_and_corruption_rejection(tmp_path: Path) -> None:
    synthetic_root(tmp_path)
    report = run(tmp_path)
    assert report["status"] == "complete"
    original = {
        str(p): sha256(p)
        for p in (tmp_path / OUTPUT).rglob("*")
        if p.is_file() and p.name != "execution.json"
    }
    run(tmp_path)
    assert all(sha256(Path(p)) == digest for p, digest in original.items())
    metrics = read_json(tmp_path / OUTPUT / "LR/metrics.json")
    assert metrics["hierarchical"]["rows"] == 8
    thresholds = read_json(tmp_path / OUTPUT / "LR/thresholds.json")
    assert all(c["K_y"] == 10 and c["N_y"] == 10 for c in thresholds["classes"])
    path = tmp_path / OUTPUT / "LR/sets/00000.parquet"
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="corrupted"):
        run(tmp_path)


def test_source_corruption_rejected(tmp_path: Path) -> None:
    synthetic_root(tmp_path)
    (tmp_path / BASE / "LR/probabilities/00000.parquet").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Probability hash"):
        run(tmp_path)


def test_calibration_unchanged_when_test_labels_and_probabilities_change(tmp_path: Path) -> None:
    from iot_diad.modeling.conformal import calibrate

    synthetic_root(tmp_path)
    registry = read_json(tmp_path / BASE / "accepted_model_registry.json")
    model = registry["models"][0]
    prepared = read_json(tmp_path / BASE / "prepared.json")
    exports = read_json(tmp_path / BASE / "LR/exports.json")
    first, _ = calibrate(tmp_path, tmp_path / "first", model, prepared, exports)
    path = tmp_path / BASE / "LR/probabilities/00000.parquet"
    frame = pl.read_parquet(path)
    changed = frame.with_columns(
        [
            pl.when(pl.col("_partition") == "test")
            .then(pl.col(PROBS[7 - j]))
            .otherwise(pl.col(column))
            .alias(column)
            for j, column in enumerate(PROBS)
        ]
    ).with_columns(
        pl.when(pl.col("_partition") == "test")
        .then(pl.lit("Benign"))
        .otherwise(pl.col("_label"))
        .alias("_label")
    )
    write_parquet(path, changed)
    cache_path = tmp_path / BASE / prepared["evaluation_chunks"][0]["path"]
    write_parquet(cache_path, changed.select(pl.read_parquet(cache_path).columns))
    prepared["evaluation_chunks"][0]["sha256"] = sha256(cache_path)
    receipt = exports["chunks"][0]
    receipt["sha256"] = sha256(path)
    receipt["input_sha256"] = sha256(cache_path)
    receipt["max_probability_sum_error"] = float(
        np.abs(changed.select(PROBS).to_numpy().sum(axis=1) - 1).max()
    )
    write_json(path.with_suffix(".json"), receipt)
    second, _ = calibrate(tmp_path, tmp_path / "second", model, prepared, exports)
    assert first == second
