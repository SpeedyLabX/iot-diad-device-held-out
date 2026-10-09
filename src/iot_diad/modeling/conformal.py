"""Frozen CONF-001 calibration and streamed prediction sets; no estimator fitting."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray

from iot_diad.modeling.baselines import (
    CLASSES,
    MODELS,
    guard_manifest,
    sha256,
    validate_probabilities,
    versions,
    write_json,
    write_parquet,
)
from iot_diad.modeling.data import METADATA_COLUMNS, load_partition_contract

BASE = Path("artifacts/experiments/baselines")
OUTPUT = Path("artifacts/experiments/conformal")
METHODS = ("hierarchical", "equal_row_diagnostic")
PROBS = [f"prob__{label}" for label in CLASSES]
LIMITATION = (
    "Hierarchical exchangeability is an assumption, not an audited property. "
    "The deterministic, temporally structured, capture-shared benchmark supports empirical "
    "coverage reporting, not unconditional distribution-free coverage. The equal-row "
    "classwise comparator has no clustered-device validity claim."
)


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def quantile(arrays: list[NDArray[np.float64]], hierarchical: bool) -> float:
    """Invert the exact rational CDF on ordered nonnegative IEEE-754 float64 values.

    Inputs are sorted per-device disk maps. Binary search needs at most 62 iterations,
    each using O(K) memory and binary searches within the device maps. No weight is
    rounded and ties use the right-continuous empirical CDF.
    """
    arrays = [a for a in arrays if len(a)]
    count = sum(len(a) for a in arrays)
    groups = len(arrays)
    target = Fraction(9, 10)
    finite_mass = Fraction(groups, groups + 1) if hierarchical else Fraction(count, count + 1)
    if finite_mass < target:
        return math.inf
    lower, upper = 0, int(np.float64(1).view(np.uint64))
    while lower < upper:
        middle = (lower + upper) // 2
        value = np.uint64(middle).view(np.float64)
        counts = [int(np.searchsorted(a, value, side="right")) for a in arrays]
        cdf = (
            sum(
                (Fraction(n, (groups + 1) * len(a)) for n, a in zip(counts, arrays, strict=True)),
                Fraction(),
            )
            if hierarchical
            else Fraction(sum(counts), count + 1)
        )
        if cdf >= target:
            upper = middle
        else:
            lower = middle + 1
    return float(np.uint64(lower).view(np.float64))


def encode_threshold(value: float) -> float | str:
    return "infinity" if math.isinf(value) else value


def threshold_vector(thresholds: dict[str, Any], method: str) -> NDArray[np.float64]:
    return np.asarray(
        [math.inf if row[method] == "infinity" else row[method] for row in thresholds["classes"]],
        dtype=np.float64,
    )


def membership(
    probabilities: NDArray[np.float64], thresholds: NDArray[np.float64]
) -> NDArray[np.uint8]:
    selected = (1.0 - probabilities) <= thresholds
    return np.sum(selected.astype(np.uint16) * (1 << np.arange(8)), axis=1).astype(np.uint8)


def accepted_sources(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    registry = read_json(root / BASE / "accepted_model_registry.json")
    models = [m for m in registry["models"] if m["primary_learned_model"]]
    require(registry["status"] == "passed", "Registry is not accepted")
    require([m["model_id"] for m in models] == list(MODELS), "Wrong learned-model routing")
    require(registry["class_order"] == list(CLASSES), "Registry class order mismatch")
    for model in models:
        directory = root / model["artifact_directory"]
        for name, key in [
            ("exports.json", "exports_sha256"),
            ("fit.json", "fit_sha256"),
            ("run_config.json", "run_config_sha256"),
            ("model.joblib", "model_sha256"),
        ]:
            require(sha256(directory / name) == model[key], f"Source hash mismatch: {name}")
        config, fit = read_json(directory / "run_config.json"), read_json(directory / "fit.json")
        require(
            sha256(root / BASE / "training_sample_manifest.csv") == model["sample_manifest_sha256"],
            "Training sample hash mismatch",
        )
        exports = read_json(directory / "exports.json")
        require(model["class_order"] == config["class_order"] == list(CLASSES), "Class order")
        require(
            config["run_id"] == model["run_id"] and config["model_id"] == model["model_id"],
            "Run identity mismatch",
        )
        require(fit["status"] == exports["fit_status"] == "passed", "Unaccepted fit")
        require(exports["status"] == "complete", "Incomplete source export")
    return registry, models


def audited_frame(
    root: Path, model: dict[str, Any], record: dict[str, Any], source: dict[str, Any], index: int
) -> pl.DataFrame:
    directory = root / model["artifact_directory"]
    path = directory.parent / record["path"]
    require(path == directory / "probabilities" / f"{index:05d}.parquet", "Source routing")
    require(sha256(path) == record["sha256"], "Probability hash mismatch")
    require(read_json(path.with_suffix(".json")) == record, "Source receipt mismatch")
    require(record["run_config_sha256"] == model["run_config_sha256"], "Run hash mismatch")
    require(record["input_sha256"] == source["sha256"], "Input lineage mismatch")
    frame = pl.read_parquet(path)
    expected = pl.read_parquet(root / BASE / source["path"], columns=list(METADATA_COLUMNS))
    require(frame.height == record["rows"] == source["rows"], "Row count mismatch")
    require(frame.select(METADATA_COLUMNS).equals(expected), "Source row identity mismatch")
    require(frame["model_id"].unique().to_list() == [model["model_id"]], "Model column")
    require(frame["run_id"].unique().to_list() == [model["run_id"]], "Run column")
    require(
        frame["sample_manifest_sha256"].unique().to_list() == [model["sample_manifest_sha256"]],
        "Sample column",
    )
    require([c for c in frame.columns if c.startswith("prob__")] == PROBS, "Probability columns")
    probabilities = frame.select(PROBS).to_numpy()
    validate_probabilities(probabilities, frame.height)
    error = float(np.abs(probabilities.sum(axis=1) - 1).max())
    require(abs(error - record["max_probability_sum_error"]) <= 1e-15, "Probability audit")
    require(set(frame["_partition"].unique()) <= {"calibration", "test"}, "Unexpected partition")
    require(set(frame["_label"].unique()) <= set(CLASSES), "Unexpected class")
    return frame


def save_chunk(
    directory: Path, relative: str, frame: pl.DataFrame, index: int, source_hash: str
) -> dict[str, Any]:
    path = directory / relative
    receipt = path.with_suffix(".json")
    expected = {
        "path": relative,
        "rows": frame.height,
        "source_index": index,
        "source_sha256": source_hash,
    }
    if receipt.exists():
        previous = read_json(receipt)
        require(all(previous[k] == v for k, v in expected.items()), "Resume lineage mismatch")
        require(sha256(path) == previous["sha256"], "Resume chunk corrupted")
        require(pl.read_parquet(path).equals(frame), "Resume content differs")
        return previous
    write_parquet(path, frame)
    expected["sha256"] = sha256(path)
    write_json(receipt, expected)
    return expected


def check_population(root: Path, prepared: dict[str, Any]) -> None:
    """Validate canonical cache hashes and approved partition identities once per invocation."""
    contract = load_partition_contract(
        root / "artifacts/splits/device_split_manifest.csv",
        root / "artifacts/splits/cross_partition_duplicate_exclusions.csv",
    )
    counts: Counter[tuple[str, str, str]] = Counter()
    last_indices: dict[str, int] = {}
    for source in prepared["evaluation_chunks"]:
        path = root / BASE / source["path"]
        require(sha256(path) == source["sha256"], "Prepared input hash mismatch")
        frame = pl.read_parquet(path, columns=list(METADATA_COLUMNS))
        require(frame.height == source["rows"], "Prepared row count")
        for capture, part in frame.partition_by("_capture_id", as_dict=True).items():
            name = str(capture[0])
            indices = part["_row_index"].to_numpy()
            require(bool(np.all(np.diff(indices.astype(np.int64)) > 0)), "Duplicate/unordered keys")
            require(int(indices[0]) > last_indices.get(name, -1), "Repeated source row")
            last_indices[name] = int(indices[-1])
            require(
                not set(indices) & contract.excluded_rows_by_capture.get(name, frozenset()),
                "Excluded row retained",
            )
        for local, device, partition in (
            frame.select("_local_identifier", "_canonical_device_id", "_partition").unique().rows()
        ):
            require(contract.device_by_identifier.get(local) == device, "Physical identity")
            require(contract.partition_by_identifier.get(local) == partition, "Partition mismatch")
        for part, device, label, n in (
            frame.group_by("_partition", "_canonical_device_id", "_label").len().rows()
        ):
            counts[(part, device, label)] += n
    expected = {
        (c["partition"], c["device"], c["label"]): c["rows"]
        for c in prepared["cells"]
        if c["partition"] != "train"
    }
    require(dict(counts) == expected, "Prepared population cells differ")


def calibrate(
    root: Path,
    directory: Path,
    model: dict[str, Any],
    prepared: dict[str, Any],
    exports: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    chunks = []
    support: Counter[tuple[str, str]] = Counter()
    score_dir = directory / "sorted_scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    devices = sorted({c["device"] for c in prepared["cells"] if c["partition"] == "calibration"})
    # Rebuild derived sorted files on resume; committed calibration chunks are verified/reused.
    for y in range(8):
        for k in range(len(devices)):
            (score_dir / f"{y}-{k}.f64").write_bytes(b"")
    for index, (record, source) in enumerate(
        zip(exports["chunks"], prepared["evaluation_chunks"], strict=True)
    ):
        # Only calibration rows enter score/support/quantile calculations.
        frame = audited_frame(root, model, record, source, index).filter(
            pl.col("_partition") == "calibration"
        )
        if not frame.height:
            continue
        scores = np.zeros(frame.height, dtype=np.float64)
        labels = frame["_label"].to_numpy()
        for label in CLASSES:
            selected = labels == label
            scores[selected] = 1.0 - frame[f"prob__{label}"].to_numpy()[selected]
        scored = frame.select(METADATA_COLUMNS).with_columns(pl.Series("score", scores))
        chunks.append(
            save_chunk(
                directory, f"calibration/{index:05d}.parquet", scored, index, record["sha256"]
            )
        )
        for (device, label), group in scored.partition_by(
            "_canonical_device_id", "_label", as_dict=True
        ).items():
            support[(str(device), str(label))] += group.height
            path = score_dir / f"{CLASSES.index(str(label))}-{devices.index(str(device))}.f64"
            with path.open("ab") as handle:
                group["score"].to_numpy().astype("<f8").tofile(handle)
    expected = {
        (c["device"], c["label"]): c["rows"]
        for c in prepared["cells"]
        if c["partition"] == "calibration"
    }
    require(dict(support) == expected, "Calibration support mismatch")
    classes = []
    for y, label in enumerate(CLASSES):
        arrays: list[NDArray[np.float64]] = []
        device_records = []
        k_y = sum(support[(device, label)] > 0 for device in devices)
        for k, device in enumerate(devices):
            n = support[(device, label)]
            if not n:
                continue
            path = score_dir / f"{y}-{k}.f64"
            values = np.memmap(path, dtype="<f8", mode="r+", shape=(n,))
            values.sort(kind="quicksort")
            values.flush()
            arrays.append(values)
            device_records.append(
                {
                    "device": device,
                    "N_ky": n,
                    "device_weight": str(Fraction(1, k_y + 1)),
                    "score_weight": str(Fraction(1, (k_y + 1) * n)),
                    "sorted_scores_path": str(path.relative_to(directory)),
                    "sorted_scores_sha256": sha256(path),
                }
            )
        n_y = sum(len(a) for a in arrays)
        classes.append(
            {
                "label": label,
                "K_y": k_y,
                "N_y": n_y,
                "score_count": n_y,
                "devices": device_records,
                "absent_calibration_devices": [d for d in devices if not support[(d, label)]],
                "infinity_weight": str(Fraction(1, k_y + 1)),
                "equal_row_score_weight": str(Fraction(1, n_y + 1)),
                "equal_row_infinity_weight": str(Fraction(1, n_y + 1)),
                "hierarchical": encode_threshold(quantile(arrays, True)),
                "equal_row_diagnostic": encode_threshold(quantile(arrays, False)),
            }
        )
    thresholds = {
        "alpha": 0.10,
        "quantile_target": "9/10",
        "score": "1 - p_hat(y|x)",
        "fit_partition": "calibration",
        "class_order": list(CLASSES),
        "probability_columns": PROBS,
        "classes": classes,
        "limitation": LIMITATION,
    }
    guard_manifest(directory / "thresholds.json", thresholds)
    return thresholds, chunks


def aggregate(
    cells: Counter[tuple[str, str]], covered: Counter[tuple[str, str]], sizes: Counter[int]
) -> dict[str, Any]:
    records: list[dict[str, Any]] = [
        {
            "device": device,
            "label": label,
            "rows": n,
            "covered": covered[(device, label)],
            "coverage": covered[(device, label)] / n,
        }
        for (device, label), n in sorted(cells.items())
    ]
    classes = []
    for label in CLASSES:
        selected = [c for c in records if c["label"] == label]
        rows = sum(int(c["rows"]) for c in selected)
        hits = sum(int(c["covered"]) for c in selected)
        classes.append(
            {
                "label": label,
                "rows": rows,
                "covered": hits,
                "device_count": len(selected),
                "device_balanced_coverage": (
                    sum(float(c["coverage"]) for c in selected) / len(selected)
                    if selected
                    else None
                ),
                "row_weighted_coverage": hits / rows if rows else None,
            }
        )
    total = sum(sizes.values())
    total_size = sum(s * n for s, n in sizes.items())
    return {
        "cells": records,
        "classes": classes,
        "rows": total,
        "total_set_size": total_size,
        "singleton_count": sizes[1],
        "empty_count": sizes[0],
        "mean_set_size": total_size / total if total else None,
        "singleton_rate": sizes[1] / total if total else None,
        "empty_set_rate": sizes[0] / total if total else None,
        "worst_device_class_coverage": min((c["coverage"] for c in records), default=None),
    }


def evaluate(
    root: Path,
    directory: Path,
    model: dict[str, Any],
    prepared: dict[str, Any],
    exports: dict[str, Any],
    thresholds: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    chunks = []
    cells: Counter[tuple[str, str]] = Counter()
    covered: dict[str, Counter[tuple[str, str]]] = {m: Counter() for m in METHODS}
    sizes: dict[str, Counter[int]] = {m: Counter() for m in METHODS}
    vectors = {m: threshold_vector(thresholds, m) for m in METHODS}
    popcount = np.asarray([n.bit_count() for n in range(256)])
    for index, (record, source) in enumerate(
        zip(exports["chunks"], prepared["evaluation_chunks"], strict=True)
    ):
        frame = audited_frame(root, model, record, source, index).filter(
            pl.col("_partition") == "test"
        )
        if not frame.height:
            continue
        probabilities = frame.select(PROBS).to_numpy()
        result = frame.select(METADATA_COLUMNS)
        for method in METHODS:
            masks = membership(probabilities, vectors[method])
            result = result.with_columns(pl.Series(f"{method}_mask", masks))
            values, counts = np.unique(popcount[masks], return_counts=True)
            sizes[method].update({int(s): int(n) for s, n in zip(values, counts, strict=True)})
        chunks.append(
            save_chunk(directory, f"sets/{index:05d}.parquet", result, index, record["sha256"])
        )
        for (device, label), group in result.partition_by(
            "_canonical_device_id", "_label", as_dict=True
        ).items():
            key = (str(device), str(label))
            cells[key] += group.height
            bit = 1 << CLASSES.index(str(label))
            for method in METHODS:
                covered[method][key] += int(
                    np.count_nonzero(group[f"{method}_mask"].to_numpy() & bit)
                )
    expected = {
        (c["device"], c["label"]): c["rows"] for c in prepared["cells"] if c["partition"] == "test"
    }
    require(dict(cells) == expected, "Test population support mismatch")
    metrics: dict[str, Any] = {m: aggregate(cells, covered[m], sizes[m]) for m in METHODS}
    metrics["limitation"] = LIMITATION
    metrics["absent_test_device_class_cells"] = [
        {"device": device, "label": label, "rows": 0, "coverage": None}
        for device in sorted({d for d, _ in cells})
        for label in CLASSES
        if (device, label) not in cells
    ]
    guard_manifest(directory / "metrics.json", metrics)
    return metrics, chunks


def run(root: Path) -> dict[str, Any]:
    started = time.monotonic()
    registry, models = accepted_sources(root)
    prepared = read_json(root / BASE / "prepared.json")
    output = root / OUTPUT
    input_paths = [
        BASE / "accepted_model_registry.json",
        BASE / "prepared.json",
        Path("docs/reports/gate3_protocol.md"),
        Path("docs/reports/gate3_effective_protocol.md"),
        Path("artifacts/splits/device_split_manifest.csv"),
        Path("artifacts/splits/cross_partition_duplicate_exclusions.csv"),
        Path("artifacts/splits/gate2_decision_summary.json"),
    ]
    contract = {
        "task": "CONF-001",
        "alpha": 0.10,
        "models": models,
        "class_order": list(CLASSES),
        "probability_columns": PROBS,
        "source_hashes": {str(p): sha256(root / p) for p in input_paths},
        "code_sha256": sha256(Path(__file__)),
        "software": versions(),
        "membership_encoding": "uint8: bit j set iff 1-probability[j] <= threshold[j]",
        "score_dtype": "little-endian float64",
        "limitation": LIMITATION,
    }
    guard_manifest(output / "contract.json", contract)
    check_population(root, prepared)
    reports = []
    for model in models:
        name = model["model_id"]
        directory = output / name
        model_started = time.monotonic()
        print(f"{name}: calibration", flush=True)
        exports = read_json(root / model["artifact_directory"] / "exports.json")
        thresholds, calibration_chunks = calibrate(root, directory, model, prepared, exports)
        print(f"{name}: thresholds frozen; applying to test", flush=True)
        metrics, test_chunks = evaluate(root, directory, model, prepared, exports, thresholds)
        require(
            sum(c["rows"] for c in calibration_chunks)
            == registry["evaluation_rows"]["calibration"],
            "Calibration total mismatch",
        )
        require(
            sum(c["rows"] for c in test_chunks) == registry["evaluation_rows"]["test"],
            "Test total mismatch",
        )
        manifest = {
            "model_id": name,
            "run_id": model["run_id"],
            "status": "complete",
            "contract_sha256": sha256(output / "contract.json"),
            "thresholds_sha256": sha256(directory / "thresholds.json"),
            "metrics_sha256": sha256(directory / "metrics.json"),
            "calibration_chunks": calibration_chunks,
            "test_chunks": test_chunks,
        }
        guard_manifest(directory / "manifest.json", manifest)
        reports.append(
            {
                "model_id": name,
                "manifest_sha256": sha256(directory / "manifest.json"),
                "runtime_seconds": time.monotonic() - model_started,
            }
        )
        print(f"{name}: complete ({reports[-1]['runtime_seconds']:.1f}s)", flush=True)
    report = {
        "status": "complete",
        "models": reports,
        "runtime_seconds": time.monotonic() - started,
        "limitation": LIMITATION,
    }
    destination = output / (
        "resume_execution.json" if (output / "execution.json").exists() else "execution.json"
    )
    write_json(destination, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve()), indent=2))


if __name__ == "__main__":
    main()
