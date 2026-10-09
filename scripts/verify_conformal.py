"""Independently reconstruct CONF-001 evidence, without importing its runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import polars as pl

META = [
    "_partition",
    "_canonical_device_id",
    "_local_identifier",
    "_capture_id",
    "_row_index",
    "_label",
]
METHODS = ("hierarchical", "equal_row_diagnostic")
CLASSES = ["Benign", "Brute Force", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web-based"]
BASE = Path("artifacts/experiments/baselines")
GOVERNANCE = [
    "artifacts/device_mapping/approved_packet_device_mapping.csv",
    "artifacts/splits/device_split_manifest.csv",
    "artifacts/splits/cross_partition_duplicate_exclusions.csv",
    "docs/reports/gate3_protocol.md",
    "docs/reports/gate3_effective_protocol.md",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(path.read_text())
    return result


def threshold_value(value: Any) -> float:
    if value == "infinity":
        return math.inf
    require(isinstance(value, (int, float)) and math.isfinite(value), "invalid threshold")
    require(0 <= value <= 1, "threshold outside score range")
    return float(value)


def check_quantile(
    counts: dict[str, int],
    below: dict[str, int],
    through: dict[str, int],
    threshold: float,
    method: str,
) -> dict[str, str]:
    """Check exact CDF bracketing, including ties and the positive-infinity atom."""
    n = sum(counts.values())
    k = len(counts)
    if method == "hierarchical":
        left = sum((Fraction(below.get(d, 0), (k + 1) * v) for d, v in counts.items()), Fraction())
        right = sum(
            (Fraction(through.get(d, 0), (k + 1) * v) for d, v in counts.items()), Fraction()
        )
        finite_mass = Fraction(k, k + 1)
    else:
        left = Fraction(sum(below.values()), n + 1)
        right = Fraction(sum(through.values()), n + 1)
        finite_mass = Fraction(n, n + 1)
    if math.isinf(threshold):
        require(finite_mass < Fraction(9, 10), "unnecessary infinite threshold")
        left, right = finite_mass, Fraction(1)
    require(left < Fraction(9, 10) <= right, "threshold fails exact quantile CDF bracket")
    return {"cdf_strictly_below": str(left), "cdf_at_threshold": str(right)}


def metrics_from_counts(
    cells: dict[tuple[str, str], list[int]],
    totals: list[int],
) -> dict[str, Any]:
    rows, size, singleton, empty = totals
    records: list[dict[str, Any]] = [
        {"device": d, "label": y, "rows": n, "covered": c, "coverage": c / n if n else None}
        for (d, y), (n, c) in sorted(cells.items())
    ]
    classes = []
    for label in CLASSES:
        selected = [r for r in records if r["label"] == label and r["rows"]]
        count = sum(r["rows"] for r in selected)
        covered = sum(r["covered"] for r in selected)
        classes.append(
            {
                "label": label,
                "rows": count,
                "covered": covered,
                "device_count": len(selected),
                "device_balanced_coverage": sum(r["coverage"] for r in selected) / len(selected)
                if selected
                else None,
                "row_weighted_coverage": covered / count if count else None,
            }
        )
    return {
        "cells": records,
        "classes": classes,
        "rows": rows,
        "total_set_size": size,
        "singleton_count": singleton,
        "empty_count": empty,
        "mean_set_size": size / rows,
        "singleton_rate": singleton / rows,
        "empty_set_rate": empty / rows,
        "worst_device_class_coverage": min(r["coverage"] for r in records if r["rows"]),
    }


def equal(actual: Any, expected: Any, context: str) -> None:
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), context + " keys")
        for key in expected:
            equal(actual[key], expected[key], context + "." + str(key))
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), context + " length")
        for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
            equal(a, e, context + f"[{index}]")
    elif isinstance(expected, float):
        require(
            isinstance(actual, (int, float))
            and math.isclose(actual, expected, rel_tol=1e-14, abs_tol=1e-15),
            context + " numeric mismatch",
        )
    else:
        require(actual == expected, context + " mismatch")


def verify_model(
    root: Path,
    output: Path,
    model: dict[str, Any],
    registry: dict[str, Any],
    prepared: dict[str, Any],
) -> dict[str, Any]:
    temporary = TemporaryDirectory(prefix="conf001-independent-")
    scratch = Path(temporary.name)
    model_id = model["model_id"]
    source_dir = root / model["artifact_directory"]
    directory = output / model_id
    manifest = read(directory / "manifest.json")
    thresholds = read(directory / "thresholds.json")
    saved_metrics = read(directory / "metrics.json")
    require(
        digest(directory / "thresholds.json") == manifest["thresholds_sha256"], "threshold hash"
    )
    require(digest(directory / "metrics.json") == manifest["metrics_sha256"], "metrics hash")
    for name, filename in [
        ("exports", "exports.json"),
        ("fit", "fit.json"),
        ("metrics", "metrics.json"),
        ("model", "model.joblib"),
        ("run_config", "run_config.json"),
    ]:
        require(digest(source_dir / filename) == model[name + "_sha256"], name + " source hash")
    require(
        digest(root / BASE / "training_sample_manifest.csv") == model["sample_manifest_sha256"],
        "canonical training sample hash",
    )
    config = read(source_dir / "run_config.json")
    fit = read(source_dir / "fit.json")
    exports = read(source_dir / "exports.json")
    require(fit["status"] == exports["fit_status"] == "passed", "fit acceptance")
    require(exports["status"] == "complete", "incomplete source exports")
    require(exports["rows"] == sum(registry["evaluation_rows"].values()), "source export totals")
    require(config["class_order"] == model["class_order"] == CLASSES, "class order")
    require(config["run_id"] == model["run_id"] and config["model_id"] == model_id, "run routing")
    require(fit["model_sha256"] == model["model_sha256"], "fit model hash linkage")
    require(
        fit["training_rows"] == model["training_rows"] == prepared["training_rows"],
        "fit training row linkage",
    )
    require(config["sample_manifest_sha256"] == prepared["sample_manifest_sha256"], "sample")
    require(len(exports["chunks"]) == len(prepared["evaluation_chunks"]), "source chunks")
    require(
        manifest["model_id"] == model_id and manifest["run_id"] == model["run_id"],
        "conformal model routing",
    )
    require(manifest["status"] == "complete", "incomplete conformal outputs")
    require(manifest["contract_sha256"] == digest(output / "contract.json"), "contract linkage")
    require(thresholds["alpha"] == 0.1 and thresholds["quantile_target"] == "9/10", "alpha")
    require(thresholds["fit_partition"] == "calibration", "threshold partition")
    require(thresholds["score"] == "1 - p_hat(y|x)", "score definition")
    require(thresholds["class_order"] == CLASSES, "threshold class order")
    require(thresholds["probability_columns"] == ["prob__" + y for y in CLASSES], "score columns")
    records = thresholds["classes"]
    require([r["label"] for r in records] == CLASSES, "threshold class order")
    q = {method: np.array([threshold_value(r[method]) for r in records]) for method in METHODS}
    calibration: Counter[tuple[str, str]] = Counter()
    partition_cells: Counter[tuple[str, str, str]] = Counter()
    bracket: dict[tuple[str, str, str, str], int] = defaultdict(int)
    cells: dict[str, dict[tuple[str, str], list[int]]] = {
        method: defaultdict(lambda: [0, 0]) for method in METHODS
    }
    totals = {method: [0, 0, 0, 0] for method in METHODS}
    receipts = {}
    for partition, key in [("calibration", "calibration_chunks"), ("test", "test_chunks")]:
        receipts[partition] = {r["source_index"]: r for r in manifest[key]}
        require(len(receipts[partition]) == len(manifest[key]), "duplicate output source index")
    seen: dict[str, int] = {}
    processed: dict[str, set[int]] = {"calibration": set(), "test": set()}
    mapping = pl.read_csv(root / GOVERNANCE[0])
    approved = dict(
        mapping.filter(pl.col("canonical_device_id").is_not_null())
        .select("identifier", "canonical_device_id")
        .iter_rows()
    )
    split = dict(
        pl.read_csv(root / GOVERNANCE[1]).select("canonical_device_id", "partition").iter_rows()
    )
    excluded = pl.read_csv(root / GOVERNANCE[2]).select(
        pl.col("capture_id").alias("_capture_id"),
        pl.col("row_index").cast(pl.UInt32).alias("_row_index"),
    )
    max_probability_error = 0.0
    for index, (source, cache) in enumerate(
        zip(exports["chunks"], prepared["evaluation_chunks"], strict=True)
    ):
        path = source_dir.parent / source["path"]
        require(path.resolve().is_relative_to(source_dir.resolve()), "source escapes accepted run")
        require(digest(path) == source["sha256"], "source probability hash")
        require(read(path.with_suffix(".json")) == source, "source receipt")
        require(source["input_sha256"] == cache["sha256"], "cache linkage")
        require(source["run_config_sha256"] == model["run_config_sha256"], "run chunk linkage")
        cache_path = root / BASE / cache["path"]
        require(digest(cache_path) == cache["sha256"], "cache hash")
        frame = pl.read_parquet(path)
        cached = pl.read_parquet(cache_path, columns=META)
        require(frame.select(META).equals(cached), "prepared row identity")
        require(frame.height == source["rows"] == cache["rows"], "source row count")
        require(frame.select(META).null_count().sum_horizontal().item() == 0, "null metadata")
        require(
            frame.join(excluded, on=["_capture_id", "_row_index"], how="inner").is_empty(),
            "excluded duplicate retained",
        )
        for capture in frame.partition_by("_capture_id"):
            identity = capture["_capture_id"][0]
            indices = capture["_row_index"].to_numpy().astype(np.int64)
            require(bool(np.all(np.diff(indices) > 0)), "duplicate or unsorted source row")
            require(int(indices[0]) > seen.get(identity, -1), "source row repeated across shards")
            seen[identity] = int(indices[-1])
        for partition, device, identifier in (
            frame.select("_partition", "_canonical_device_id", "_local_identifier")
            .unique()
            .iter_rows()
        ):
            require(approved.get(identifier) == device, "physical-device identity")
            require(split.get(device) == partition and partition in processed, "split membership")
        require(frame["model_id"].unique().to_list() == [model_id], "probability model ID")
        require(frame["run_id"].unique().to_list() == [model["run_id"]], "probability run ID")
        for column, value in [
            ("seed", config["seed"]),
            ("feature_contract_sha256", config["feature_contract_sha256"]),
            ("sample_manifest_sha256", model["sample_manifest_sha256"]),
        ]:
            require(frame[column].unique().to_list() == [value], "source metadata: " + column)
        for software in frame["software_versions"].unique():
            require(json.loads(software) == config["software_versions"], "software metadata")
        probability_columns = ["prob__" + label for label in CLASSES]
        require(
            [c for c in frame.columns if c.startswith("prob__")] == probability_columns,
            "probability column mapping",
        )
        probability = frame.select(probability_columns).to_numpy()
        require(
            bool(
                np.isfinite(probability).all()
                and (probability >= 0).all()
                and (probability <= 1).all()
            ),
            "probability range",
        )
        error = float(np.abs(probability.sum(axis=1) - 1).max())
        require(error <= 1e-8, "probability sums")
        equal(source["max_probability_sum_error"], error, "source probability error")
        max_probability_error = max(max_probability_error, error)
        for partition in processed:
            selection = frame["_partition"].to_numpy() == partition
            part = frame.filter(pl.Series(selection))
            if not part.height:
                continue
            processed[partition].add(index)
            receipt = receipts[partition][index]
            output_path = directory / receipt["path"]
            require(output_path.resolve().is_relative_to(directory.resolve()), "output path escape")
            require(digest(output_path) == receipt["sha256"], "output chunk hash")
            require(read(output_path.with_suffix(".json")) == receipt, "output receipt")
            require(receipt["source_sha256"] == source["sha256"], "output source linkage")
            require(receipt["rows"] == part.height, "output receipt rows")
            exported = pl.read_parquet(output_path)
            require(exported.select(META).equals(part.select(META)), "output row identity")
            scores = 1.0 - probability[selection]
            label_indices = np.array([CLASSES.index(y) for y in part["_label"].to_list()])
            true_scores = scores[np.arange(part.height), label_indices]
            for device, label, count in (
                part.group_by("_canonical_device_id", "_label").len().rows()
            ):
                partition_cells[(partition, device, label)] += count
            if partition == "calibration":
                np.testing.assert_array_equal(exported["score"].to_numpy(), true_scores)
                scored = part.select("_canonical_device_id", "_label").with_columns(
                    pl.Series("score", true_scores)
                )
                for device_frame in scored.partition_by("_canonical_device_id", "_label"):
                    device = device_frame["_canonical_device_id"][0]
                    label = device_frame["_label"][0]
                    values = device_frame["score"].to_numpy()
                    calibration[(device, label)] += len(values)
                    with (scratch / f"{device}-{CLASSES.index(label)}.f64").open("ab") as handle:
                        values.astype("<f8").tofile(handle)
                    for method in METHODS:
                        threshold = q[method][CLASSES.index(label)]
                        bracket[(method, label, device, "below")] += int((values < threshold).sum())
                        bracket[(method, label, device, "through")] += int(
                            (values <= threshold).sum()
                        )
            else:
                for method in METHODS:
                    included = scores <= q[method]
                    expected_mask = np.zeros(part.height, dtype=np.uint8)
                    for j in range(8):
                        expected_mask |= included[:, j].astype(np.uint8) << j
                    require(exported.schema[method + "_mask"] == pl.UInt8, "mask dtype")
                    np.testing.assert_array_equal(
                        exported[method + "_mask"].to_numpy(), expected_mask
                    )
                    sizes = included.sum(axis=1)
                    totals[method][0] += part.height
                    totals[method][1] += int(sizes.sum())
                    totals[method][2] += int((sizes == 1).sum())
                    totals[method][3] += int((sizes == 0).sum())
                    covered = included[np.arange(part.height), label_indices]
                    grouped = (
                        part.select("_canonical_device_id", "_label")
                        .with_columns(pl.Series("covered", covered))
                        .group_by("_canonical_device_id", "_label")
                        .agg(pl.len().alias("rows"), pl.col("covered").sum())
                    )
                    for device, label, count, covered_count in grouped.iter_rows():
                        cells[method][(device, label)][0] += count
                        cells[method][(device, label)][1] += covered_count
        if (index + 1) % 200 == 0:
            print(f"{model_id}: independently verified {index + 1} source chunks", flush=True)
    for partition in processed:
        require(processed[partition] == set(receipts[partition]), "extra or missing output shard")
    expected_cells = {
        (c["partition"], c["device"], c["label"]): c["rows"]
        for c in prepared["cells"]
        if c["partition"] in processed
    }
    require(dict(partition_cells) == expected_cells, "complete population cell counts")
    for partition in processed:
        require(
            sum(n for (p, _, _), n in partition_cells.items() if p == partition)
            == registry["evaluation_rows"][partition],
            "full partition rows",
        )
    cdf_evidence = []
    for record in records:
        label = record["label"]
        counts = {d: n for (d, y), n in calibration.items() if y == label}
        require(record["K_y"] == len(counts) and record["N_y"] == sum(counts.values()), "support")
        require({d["device"]: d["N_ky"] for d in record["devices"]} == counts, "device support")
        require(record["score_count"] == sum(counts.values()), "score count")
        require(
            Fraction(record["infinity_weight"]) == Fraction(1, len(counts) + 1), "infinity atom"
        )
        require(
            Fraction(record["equal_row_score_weight"])
            == Fraction(record["equal_row_infinity_weight"])
            == Fraction(1, sum(counts.values()) + 1),
            "equal row atom weights",
        )
        calibration_devices = {d for d, p in split.items() if p == "calibration"}
        require(
            record["absent_calibration_devices"] == sorted(calibration_devices - set(counts)),
            "absent calibration devices",
        )
        require(len(record["devices"]) == len(counts), "repeated support device")
        for device in record["devices"]:
            score_path = directory / device["sorted_scores_path"]
            require(score_path.resolve().is_relative_to(directory.resolve()), "score path escape")
            require(digest(score_path) == device["sorted_scores_sha256"], "sorted score hash")
            require(score_path.stat().st_size == 8 * device["N_ky"], "sorted score byte count")
            expected_path = scratch / f"{device['device']}-{CLASSES.index(label)}.f64"
            expected_scores = np.memmap(expected_path, dtype="<f8", mode="r+")
            expected_scores.sort(kind="quicksort")
            expected_scores.flush()
            require(
                digest(expected_path) == digest(score_path), "independent sorted score equality"
            )
            del expected_scores
            require(
                Fraction(device["device_weight"]) == Fraction(1, len(counts) + 1), "device mass"
            )
            require(
                Fraction(device["score_weight"]) == Fraction(1, (len(counts) + 1) * device["N_ky"]),
                "score mass",
            )
        for method in METHODS:
            below = {d: bracket[(method, label, d, "below")] for d in counts}
            through = {d: bracket[(method, label, d, "through")] for d in counts}
            evidence = check_quantile(
                counts, below, through, q[method][CLASSES.index(label)], method
            )
            cdf_evidence.append({"label": label, "method": method, **evidence})
    for method in METHODS:
        reconstructed = metrics_from_counts(cells[method], totals[method])
        actual = saved_metrics[method]
        actual = {
            **actual,
            "cells": sorted(actual["cells"], key=lambda r: (r["device"], r["label"])),
        }
        equal(actual, reconstructed, model_id + "." + method)
    absent = [
        {"device": d, "label": y, "rows": 0, "coverage": None}
        for d in sorted(d for d, p in split.items() if p == "test")
        for y in CLASSES
        if (d, y) not in cells["hierarchical"]
    ]
    equal(saved_metrics["absent_test_device_class_cells"], absent, "absent test cells")
    temporary.cleanup()
    return {
        "model_id": model_id,
        "status": "passed",
        "source_chunks": len(exports["chunks"]),
        "calibration_rows": sum(calibration.values()),
        "test_rows": totals["hierarchical"][0],
        "max_probability_sum_error": max_probability_error,
        "exact_quantile_cdf": cdf_evidence,
        "manifest_sha256": digest(directory / "manifest.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments/conformal"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / args.output
    start = time.perf_counter()
    registry_path = root / BASE / "accepted_model_registry.json"
    registry = read(registry_path)
    require(registry["status"] == "passed" and registry["class_order"] == CLASSES, "registry")
    models = [m for m in registry["models"] if m["primary_learned_model"]]
    require([m["model_id"] for m in models] == ["LR", "ET", "HGB", "RF", "MLP"], "model routing")
    contract = read(output / "contract.json")
    require(contract["task"] == "CONF-001" and contract["alpha"] == 0.1, "task contract")
    require(contract["models"] == models, "contract accepted model entries")
    require(contract["class_order"] == CLASSES, "contract class order")
    require(contract["probability_columns"] == ["prob__" + y for y in CLASSES], "contract columns")
    require(
        contract["code_sha256"] == digest(root / "src/iot_diad/modeling/conformal.py"),
        "runner source hash",
    )
    required_sources = {
        str(BASE / "accepted_model_registry.json"),
        str(BASE / "prepared.json"),
        "docs/reports/gate3_protocol.md",
        "docs/reports/gate3_effective_protocol.md",
        "artifacts/splits/device_split_manifest.csv",
        "artifacts/splits/cross_partition_duplicate_exclusions.csv",
        "artifacts/splits/gate2_decision_summary.json",
    }
    require(required_sources <= set(contract["source_hashes"]), "missing contract input hashes")
    for path, expected_digest in contract["source_hashes"].items():
        require(digest(root / path) == expected_digest, "contract input hash: " + path)
    prepared = read(root / BASE / "prepared.json")
    gate2 = read(root / "artifacts/splits/gate2_decision_summary.json")
    require(prepared["status"] == "passed", "prepared input status")
    require(
        prepared["rows_by_partition"] == gate2["filtered_rows_by_partition"],
        "Gate 2 prepared population",
    )
    require(
        registry["evaluation_rows"]
        == {p: gate2["filtered_rows_by_partition"][p] for p in ("calibration", "test")},
        "Gate 2 registry population",
    )
    results = [verify_model(root, output, model, registry, prepared) for model in models]
    result = {
        "status": "passed",
        "registry_sha256": digest(registry_path),
        "verifier_sha256": digest(Path(__file__)),
        "models": results,
        "seconds": time.perf_counter() - start,
        "checks": "Every source/output row; exact rational CDF; independent membership/metrics",
        "governance_sha256": {p: digest(root / p) for p in GOVERNANCE},
    }
    (output / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "passed", "models": len(results), "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
