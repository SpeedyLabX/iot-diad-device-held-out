"""Frozen STAT-001 paired-device inference; packet rows are never replicates."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import time
from pathlib import Path
from statistics import NormalDist
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from iot_diad.modeling.baselines import CLASSES, MODELS, sha256, versions

BASE = Path("artifacts/experiments/baselines")
OUTPUT = Path("artifacts/experiments/statistics")
CONFIRMATORY = ("LR", "ET", "HGB")
CONFIRMATORY_PAIRS = (("LR", "ET"), ("LR", "HGB"), ("ET", "HGB"))
SEED = 2027
BOOTSTRAP_REPLICATES = 50_000
CONFIDENCE_LEVEL = 0.95
ALPHA = 0.05
LIMITATION = (
    "Bootstrap intervals are approximate with only 14 outer physical devices. The benchmark "
    "is deterministic, temporally structured, and capture-shared; packet rows are not "
    "independent inferential replicates."
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == content, f"Resume artifact differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def guarded_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def guarded_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, stream.getvalue().encode())


def guarded_npy(path: Path, value: NDArray[Any]) -> None:
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    atomic_bytes(path, stream.getvalue())


def close(actual: float, expected: float, context: str) -> None:
    require(math.isclose(actual, expected, rel_tol=1e-14, abs_tol=1e-15), context)


def average_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return ranks


def exact_signed_rank(differences: NDArray[np.float64]) -> dict[str, Any]:
    """Two-sided exhaustive sign enumeration with zero removal and exact-tie midranks."""
    require(differences.ndim == 1 and len(differences) > 0, "Invalid paired differences")
    nonzero = differences[differences != 0.0]
    zeros = int(len(differences) - len(nonzero))
    if not len(nonzero):
        return {
            "statistic": 0.0,
            "w_plus": 0.0,
            "w_minus": 0.0,
            "raw_p_value": 1.0,
            "devices": int(len(differences)),
            "nonzero_differences": 0,
            "zero_differences": zeros,
            "absolute_rank_tie_groups": 0,
            "enumerations": 1,
            "status": "all_zero_differences",
        }
    absolute = np.abs(nonzero)
    ranks = average_ranks(absolute)
    doubled = np.rint(2 * ranks).astype(np.int64)
    total = int(doubled.sum())
    observed_plus = int(doubled[nonzero > 0].sum())
    observed_min = min(observed_plus, total - observed_plus)
    extreme = 0
    for signs in range(1 << len(nonzero)):
        plus = sum(int(rank) for i, rank in enumerate(doubled) if signs & (1 << i))
        extreme += min(plus, total - plus) <= observed_min
    counts = np.unique(absolute, return_counts=True)[1]
    return {
        "statistic": observed_min / 2,
        "w_plus": observed_plus / 2,
        "w_minus": (total - observed_plus) / 2,
        "raw_p_value": extreme / (1 << len(nonzero)),
        "devices": int(len(differences)),
        "nonzero_differences": int(len(nonzero)),
        "zero_differences": zeros,
        "absolute_rank_tie_groups": int(np.sum(counts > 1)),
        "enumerations": 1 << len(nonzero),
        "status": "exhaustive_exact_sign_enumeration",
    }


def holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 1.0
    m = len(p_values)
    for rank in range(m, 0, -1):
        index = order[rank - 1]
        running = min(running, m * p_values[index] / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def friedman(values: NDArray[np.float64]) -> dict[str, Any]:
    require(values.shape[1] == 3, "Frozen Friedman family must contain three models")
    n, k = values.shape
    ranked = np.vstack([average_ranks(row) for row in values])
    rank_sums = ranked.sum(axis=0)
    raw = 12 * float(np.square(rank_sums).sum()) / (n * k * (k + 1)) - 3 * n * (k + 1)
    ties = 0
    for row in values:
        counts = np.unique(row, return_counts=True)[1]
        ties += int(sum(int(c) ** 3 - int(c) for c in counts if c > 1))
    correction = 1 - ties / (n * (k**3 - k))
    statistic = raw / correction if correction > 0 else 0.0
    # The frozen family has k=3, hence chi-square df=2 and SF(x)=exp(-x/2).
    p_value = math.exp(-statistic / 2)
    return {
        "family": "confirmatory_model_comparison",
        "models": list(CONFIRMATORY),
        "devices": n,
        "statistic": statistic,
        "degrees_of_freedom": 2,
        "raw_p_value": p_value,
        "rank_sums": {model: float(rank_sums[i]) for i, model in enumerate(CONFIRMATORY)},
        "tie_correction": correction,
        "method": "device-blocked Friedman; chi-square approximation",
    }


def metric_from_matrix(
    matrix: NDArray[np.int64],
) -> tuple[float, NDArray[np.float64], NDArray[np.int64]]:
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    diagonal = matrix.diagonal()
    denominator = support + predicted
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros(8), where=denominator > 0)
    recall = np.divide(diagonal, support, out=np.full(8, np.nan), where=support > 0)
    return float(f1[support > 0].mean()), recall, support


def test_devices(root: Path) -> list[str]:
    with (root / "artifacts/splits/device_split_manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    devices = sorted(row["canonical_device_id"] for row in rows if row["partition"] == "test")
    require(len(devices) == 14 and len(set(devices)) == 14, "Expected 14 test physical devices")
    return devices


def load_outcomes(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    registry_path = root / BASE / "accepted_model_registry.json"
    registry = read_json(registry_path)
    routed = [m for m in registry["models"] if m["primary_learned_model"]]
    require(registry["status"] == "passed", "Registry is not accepted")
    require(
        [m["model_id"] for m in routed] == list(MODELS), "Accepted routing is not LR/ET/HGB/RF/MLP"
    )
    require(
        registry["confirmatory_model_family"] == list(CONFIRMATORY), "Confirmatory family changed"
    )
    require(registry["class_order"] == list(CLASSES), "Class order changed")
    devices = test_devices(root)
    outcomes: dict[str, dict[str, dict[str, Any]]] = {}
    for model in routed:
        directory = root / model["artifact_directory"]
        metrics_path = directory / "metrics.json"
        require(
            sha256(metrics_path) == model["metrics_sha256"],
            f"Metrics hash mismatch: {model['model_id']}",
        )
        metrics = read_json(metrics_path)
        require(metrics["class_order"] == list(CLASSES), "Metrics class order mismatch")
        selected = [d for d in metrics["devices"] if d["partition"] == "test"]
        require(sorted(d["device"] for d in selected) == devices, "Test device identities changed")
        by_device: dict[str, dict[str, Any]] = {}
        for record in selected:
            matrix = np.asarray(record["confusion_matrix"], dtype=np.int64)
            require(
                matrix.shape == (8, 8) and bool(np.all(matrix >= 0)), "Invalid confusion matrix"
            )
            macro_f1, recall, support = metric_from_matrix(matrix)
            require(int(matrix.sum()) == record["rows"], "Device row count mismatch")
            close(macro_f1, record["macro_f1"], "Stored macro-F1 mismatch")
            for index, label in enumerate(CLASSES):
                stored = record["per_class"][label]
                require(int(support[index]) == stored["support"], "Stored support mismatch")
                if support[index]:
                    close(float(recall[index]), stored["recall"], "Stored recall mismatch")
                else:
                    require(stored["recall"] is None, "Absent class recall must be null")
            by_device[record["device"]] = {
                "rows": int(matrix.sum()),
                "macro_f1": macro_f1,
                "recall": recall,
                "support": support,
            }
        mean = float(np.mean([by_device[d]["macro_f1"] for d in devices]))
        close(mean, metrics["aggregates"]["test"]["mean_device_macro_f1"], "Aggregate mismatch")
        outcomes[model["model_id"]] = by_device
    reference = outcomes[MODELS[0]]
    for model in MODELS[1:]:
        for device in devices:
            require(
                outcomes[model][device]["rows"] == reference[device]["rows"],
                "Population rows differ",
            )
            require(
                np.array_equal(outcomes[model][device]["support"], reference[device]["support"]),
                "Ground-truth class support differs",
            )
    return registry, routed, outcomes


def effect_summary(differences: NDArray[np.float64]) -> dict[str, Any]:
    return {
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "q25_difference": float(np.quantile(differences, 0.25)),
        "q75_difference": float(np.quantile(differences, 0.75)),
        "min_difference": float(np.min(differences)),
        "max_difference": float(np.max(differences)),
        "positive_devices": int(np.sum(differences > 0)),
        "negative_devices": int(np.sum(differences < 0)),
        "zero_devices": int(np.sum(differences == 0)),
    }


def bca_interval(
    observed: float, replicates: NDArray[np.float64], jackknife: NDArray[np.float64]
) -> tuple[float | None, float | None, float | None, float | None, str]:
    normal = NormalDist()
    ties = int(np.sum(replicates == observed))
    proportion = (int(np.sum(replicates < observed)) + ties / 2) / len(replicates)
    proportion = min(max(proportion, 0.5 / len(replicates)), 1 - 0.5 / len(replicates))
    z0 = normal.inv_cdf(proportion)
    centered = float(np.mean(jackknife)) - jackknife
    denominator = 6 * float(np.square(centered).sum()) ** 1.5
    if denominator == 0 or not math.isfinite(denominator):
        return None, None, z0, None, "unavailable_nonfinite_acceleration"
    acceleration = float(np.power(centered, 3).sum()) / denominator
    tail = (1 - CONFIDENCE_LEVEL) / 2
    probabilities = []
    for probability in (tail, 1 - tail):
        z = normal.inv_cdf(probability)
        divisor = 1 - acceleration * (z0 + z)
        if divisor == 0:
            return None, None, z0, acceleration, "unavailable_zero_bca_denominator"
        probabilities.append(normal.cdf(z0 + (z0 + z) / divisor))
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities):
        return None, None, z0, acceleration, "unavailable_invalid_bca_probability"
    interval = np.quantile(replicates, probabilities, method="linear")
    return float(interval[0]), float(interval[1]), z0, acceleration, "available"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    started = time.monotonic()
    output_existed = output.exists()
    registry, routed, outcomes = load_outcomes(root)
    devices = test_devices(root)
    registry_path = root / BASE / "accepted_model_registry.json"
    source_paths = [
        root / "docs/reports/gate3_protocol.md",
        root / "docs/reports/gate3_effective_protocol.md",
        registry_path,
        root / "artifacts/splits/device_split_manifest.csv",
        root / "artifacts/splits/cross_partition_duplicate_exclusions.csv",
    ]
    contract = {
        "task_id": "STAT-001",
        "status": "frozen",
        "inference_unit": "physical test device",
        "device_count": 14,
        "class_order": list(CLASSES),
        "accepted_models": list(MODELS),
        "confirmatory_models": list(CONFIRMATORY),
        "confirmatory_pairs": [list(pair) for pair in CONFIRMATORY_PAIRS],
        "confirmatory_method": (
            "two-sided paired Wilcoxon signed-rank with exhaustive sign enumeration"
        ),
        "confirmatory_correction": "Holm within exactly three hypotheses",
        "exploratory_family": "three confirmatory-model pairs across eight per-class recalls",
        "exploratory_correction": "Benjamini-Hochberg within exactly 24 hypotheses",
        "zero_difference_handling": (
            "discard exact zero differences before ranking; all-zero vector has p=1"
        ),
        "tie_handling": (
            "average ranks for exact absolute ties; enumerate all signs of nonzero "
            "paired differences"
        ),
        "bootstrap": {
            "seed": SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "sample_size": 14,
            "unit": "complete physical device record",
            "confidence_level": CONFIDENCE_LEVEL,
            "percentile_quantile_method": "linear",
            "bca_bias_ties": "half weight for bootstrap estimates equal to the observed estimate",
            "estimands": "five model means and three frozen confirmatory paired mean differences",
        },
        "limitation": LIMITATION,
        "source_sha256": {str(path.relative_to(root)): sha256(path) for path in source_paths},
        "model_routing": [
            {
                key: model[key]
                for key in (
                    "model_id",
                    "run_id",
                    "artifact_directory",
                    "metrics_sha256",
                    "exports_sha256",
                    "fit_sha256",
                    "run_config_sha256",
                    "model_sha256",
                    "sample_manifest_sha256",
                )
            }
            for model in routed
        ],
        "runner_sha256": sha256(Path(__file__)),
        "software": versions(),
        "registry_primary_metric": registry["primary_metric"],
    }
    guarded_json(output / "contract.json", contract)

    outcome_rows: list[dict[str, Any]] = []
    recall_rows: list[dict[str, Any]] = []
    for model in MODELS:
        route = next(item for item in routed if item["model_id"] == model)
        for device in devices:
            record = outcomes[model][device]
            outcome_rows.append(
                {
                    "model_id": model,
                    "run_id": route["run_id"],
                    "device": device,
                    "rows": record["rows"],
                    "macro_f1": record["macro_f1"],
                    "metrics_sha256": route["metrics_sha256"],
                }
            )
            for index, label in enumerate(CLASSES):
                recall_rows.append(
                    {
                        "model_id": model,
                        "run_id": route["run_id"],
                        "device": device,
                        "class": label,
                        "support": int(record["support"][index]),
                        "recall": float(record["recall"][index])
                        if record["support"][index]
                        else "",
                    }
                )
    guarded_csv(output / "device_macro_f1.csv", list(outcome_rows[0]), outcome_rows)
    guarded_csv(output / "device_class_recall.csv", list(recall_rows[0]), recall_rows)

    macro = {
        model: np.asarray([outcomes[model][device]["macro_f1"] for device in devices])
        for model in MODELS
    }
    guarded_json(
        output / "friedman.json",
        friedman(np.column_stack([macro[model] for model in CONFIRMATORY])),
    )
    confirmatory_rows: list[dict[str, Any]] = []
    confirmatory_difference_rows: list[dict[str, Any]] = []
    for model_a, model_b in CONFIRMATORY_PAIRS:
        differences = macro[model_a] - macro[model_b]
        result = {
            "family": "confirmatory_model_comparison",
            "hypothesis_id": f"macro_f1__{model_a}__vs__{model_b}",
            "model_a": model_a,
            "model_b": model_b,
            "estimand": "equal-device mean paired macro-F1 difference (model_a - model_b)",
            **effect_summary(differences),
            **exact_signed_rank(differences),
            "correction_method": "Holm",
        }
        confirmatory_rows.append(result)
        confirmatory_difference_rows.extend(
            {
                "hypothesis_id": result["hypothesis_id"],
                "model_a": model_a,
                "model_b": model_b,
                "device": device,
                "model_a_macro_f1": macro[model_a][index],
                "model_b_macro_f1": macro[model_b][index],
                "paired_difference": differences[index],
            }
            for index, device in enumerate(devices)
        )
    adjusted = holm([float(row["raw_p_value"]) for row in confirmatory_rows])
    for row, value in zip(confirmatory_rows, adjusted, strict=True):
        row["adjusted_p_value"] = value
        row["reject_at_0_05"] = value <= ALPHA
    guarded_csv(
        output / "confirmatory_comparisons.csv", list(confirmatory_rows[0]), confirmatory_rows
    )
    guarded_csv(
        output / "confirmatory_device_differences.csv",
        list(confirmatory_difference_rows[0]),
        confirmatory_difference_rows,
    )

    exploratory_rows: list[dict[str, Any]] = []
    exploratory_difference_rows: list[dict[str, Any]] = []
    for class_index, label in enumerate(CLASSES):
        for model_a, model_b in CONFIRMATORY_PAIRS:
            contributing = [
                device
                for device in devices
                if outcomes[model_a][device]["support"][class_index] > 0
                and outcomes[model_b][device]["support"][class_index] > 0
            ]
            a = np.asarray([outcomes[model_a][d]["recall"][class_index] for d in contributing])
            b = np.asarray([outcomes[model_b][d]["recall"][class_index] for d in contributing])
            differences = a - b
            result = {
                "family": "exploratory_per_class_recall",
                "hypothesis_id": f"recall__{label}__{model_a}__vs__{model_b}",
                "class": label,
                "model_a": model_a,
                "model_b": model_b,
                "estimand": "equal-device mean paired class-recall difference (model_a - model_b)",
                **effect_summary(differences),
                **exact_signed_rank(differences),
                "correction_method": "Benjamini-Hochberg",
            }
            exploratory_rows.append(result)
            exploratory_difference_rows.extend(
                {
                    "hypothesis_id": result["hypothesis_id"],
                    "class": label,
                    "model_a": model_a,
                    "model_b": model_b,
                    "device": device,
                    "support": int(outcomes[model_a][device]["support"][class_index]),
                    "model_a_recall": a[index],
                    "model_b_recall": b[index],
                    "paired_difference": differences[index],
                }
                for index, device in enumerate(contributing)
            )
    require(len(exploratory_rows) == 24, "Exploratory family must have exactly 24 hypotheses")
    adjusted = benjamini_hochberg([float(row["raw_p_value"]) for row in exploratory_rows])
    for row, value in zip(exploratory_rows, adjusted, strict=True):
        row["adjusted_p_value"] = value
        row["reject_at_0_05"] = value <= ALPHA
    guarded_csv(
        output / "exploratory_recall_comparisons.csv", list(exploratory_rows[0]), exploratory_rows
    )
    guarded_csv(
        output / "exploratory_recall_device_differences.csv",
        list(exploratory_difference_rows[0]),
        exploratory_difference_rows,
    )

    rng = np.random.default_rng(SEED)
    samples = rng.integers(
        0, len(devices), size=(BOOTSTRAP_REPLICATES, len(devices)), dtype=np.int16
    )
    estimand_names = list(MODELS) + [f"{a}-{b}" for a, b in CONFIRMATORY_PAIRS]
    observed = [float(np.mean(macro[model])) for model in MODELS]
    values = [macro[model] for model in MODELS]
    for model_a, model_b in CONFIRMATORY_PAIRS:
        values.append(macro[model_a] - macro[model_b])
        observed.append(float(np.mean(values[-1])))
    replicate_estimates = np.column_stack([value[samples].mean(axis=1) for value in values])
    guarded_npy(output / "bootstrap_device_indices.npy", samples)
    guarded_npy(output / "bootstrap_estimates.npy", replicate_estimates)
    estimand_rows = [
        {
            "column_index": index,
            "estimand_id": name,
            "type": "model_mean" if index < len(MODELS) else "confirmatory_paired_mean_difference",
        }
        for index, name in enumerate(estimand_names)
    ]
    guarded_csv(output / "bootstrap_estimands.csv", list(estimand_rows[0]), estimand_rows)
    bootstrap_rows: list[dict[str, Any]] = []
    loo_rows: list[dict[str, Any]] = []
    for column, (name, vector, estimate) in enumerate(
        zip(estimand_names, values, observed, strict=True)
    ):
        jackknife = np.asarray([np.delete(vector, i).mean() for i in range(len(devices))])
        percentile = np.quantile(
            replicate_estimates[:, column],
            [(1 - CONFIDENCE_LEVEL) / 2, (1 + CONFIDENCE_LEVEL) / 2],
            method="linear",
        )
        bca_low, bca_high, bias, acceleration, status = bca_interval(
            estimate, replicate_estimates[:, column], jackknife
        )
        bootstrap_rows.append(
            {
                "estimand_id": name,
                "type": estimand_rows[column]["type"],
                "observed_estimate": estimate,
                "devices": len(devices),
                "seed": SEED,
                "replicates": BOOTSTRAP_REPLICATES,
                "confidence_level": CONFIDENCE_LEVEL,
                "percentile_lower": float(percentile[0]),
                "percentile_upper": float(percentile[1]),
                "bca_lower": "" if bca_low is None else bca_low,
                "bca_upper": "" if bca_high is None else bca_high,
                "bca_bias_correction": "" if bias is None else bias,
                "bca_acceleration": "" if acceleration is None else acceleration,
                "bca_status": status,
                "resampling_unit": "complete physical device",
                "limitation": LIMITATION,
            }
        )
        loo_rows.extend(
            {
                "estimand_id": name,
                "omitted_device": device,
                "leave_one_device_out_estimate": jackknife[index],
            }
            for index, device in enumerate(devices)
        )
    guarded_csv(output / "bootstrap_summary.csv", list(bootstrap_rows[0]), bootstrap_rows)
    guarded_csv(output / "leave_one_device_out.csv", list(loo_rows[0]), loo_rows)

    core_names = [
        "contract.json",
        "device_macro_f1.csv",
        "device_class_recall.csv",
        "friedman.json",
        "confirmatory_comparisons.csv",
        "confirmatory_device_differences.csv",
        "exploratory_recall_comparisons.csv",
        "exploratory_recall_device_differences.csv",
        "bootstrap_device_indices.npy",
        "bootstrap_estimates.npy",
        "bootstrap_estimands.csv",
        "bootstrap_summary.csv",
        "leave_one_device_out.csv",
    ]
    guarded_json(
        output / "manifest.json",
        {
            "status": "complete",
            "task_id": "STAT-001",
            "files": [
                {
                    "path": name,
                    "size_bytes": (output / name).stat().st_size,
                    "sha256": sha256(output / name),
                }
                for name in core_names
            ],
        },
    )
    runtime = {
        "status": "passed",
        "seconds_this_invocation": time.monotonic() - started,
        "resume_mode": output_existed,
    }
    runtime_name = "resume_execution.json" if output_existed else "execution.json"
    temporary = output / f"{runtime_name}.tmp"
    temporary.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output / runtime_name)
    print(json.dumps(runtime, sort_keys=True))


if __name__ == "__main__":
    main()
