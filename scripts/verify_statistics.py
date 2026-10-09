"""Independent STAT-001 artifact verifier; imports no runner calculations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
from scipy.stats import friedmanchisquare, rankdata  # type: ignore[import-untyped]

CLASSES = ("Benign", "Brute Force", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web-based")
MODELS = ("LR", "ET", "HGB", "RF", "MLP")
PRIMARY = ("LR", "ET", "HGB")
PAIRS = (("LR", "ET"), ("LR", "HGB"), ("ET", "HGB"))
SEED = 2027
REPLICATES = 50_000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(path.read_text())
    return result


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual: float, expected: float, context: str) -> None:
    require(math.isclose(actual, expected, rel_tol=1e-13, abs_tol=1e-14), context)


def reconstructed(
    root: Path,
) -> tuple[dict[str, Any], list[str], dict[str, dict[str, dict[str, Any]]]]:
    registry = read_json(root / "artifacts/experiments/baselines/accepted_model_registry.json")
    routed = [entry for entry in registry["models"] if entry["primary_learned_model"]]
    require([entry["model_id"] for entry in routed] == list(MODELS), "model routing")
    require(registry["confirmatory_model_family"] == list(PRIMARY), "confirmatory routing")
    split = rows(root / "artifacts/splits/device_split_manifest.csv")
    devices = sorted(row["canonical_device_id"] for row in split if row["partition"] == "test")
    require(len(devices) == 14, "test device count")
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for route in routed:
        metric_path = root / route["artifact_directory"] / "metrics.json"
        require(digest(metric_path) == route["metrics_sha256"], "metrics hash")
        source = read_json(metric_path)
        records = [record for record in source["devices"] if record["partition"] == "test"]
        require(sorted(record["device"] for record in records) == devices, "device membership")
        result[route["model_id"]] = {}
        for record in records:
            matrix = np.asarray(record["confusion_matrix"], dtype=np.int64)
            support = matrix.sum(axis=1)
            predicted = matrix.sum(axis=0)
            diagonal = matrix.diagonal()
            f1 = np.divide(
                2 * diagonal, support + predicted, out=np.zeros(8), where=support + predicted > 0
            )
            recall = np.divide(diagonal, support, out=np.full(8, np.nan), where=support > 0)
            macro = float(np.mean(f1[support > 0]))
            close(macro, float(record["macro_f1"]), "source macro-F1")
            result[route["model_id"]][record["device"]] = {
                "rows": int(matrix.sum()),
                "macro_f1": macro,
                "support": support,
                "recall": recall,
            }
    for model in MODELS[1:]:
        for device in devices:
            require(
                np.array_equal(
                    result[model][device]["support"], result[MODELS[0]][device]["support"]
                ),
                "support pairing",
            )
    return registry, devices, result


def signed_rank(
    differences: np.ndarray[Any, np.dtype[np.float64]],
) -> tuple[float, float, int, int]:
    nonzero = differences[differences != 0]
    if not len(nonzero):
        return 0.0, 1.0, 0, len(differences)
    doubled = np.rint(2 * rankdata(np.abs(nonzero), method="average")).astype(np.int64)
    total = int(doubled.sum())
    observed = int(doubled[nonzero > 0].sum())
    statistic = min(observed, total - observed)
    enumerated = []
    for signs in itertools.product((0, 1), repeat=len(nonzero)):
        plus = sum(int(rank) for sign, rank in zip(signs, doubled, strict=True) if sign)
        enumerated.append(min(plus, total - plus))
    p_value = sum(value <= statistic for value in enumerated) / len(enumerated)
    return statistic / 2, p_value, len(nonzero), len(differences) - len(nonzero)


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values, kind="stable")
    adjusted = [0.0] * len(values)
    previous = 0.0
    for position, index_value in enumerate(order):
        index = int(index_value)
        previous = max(previous, (len(values) - position) * values[index])
        adjusted[index] = min(previous, 1.0)
    return adjusted


def bh(values: list[float]) -> list[float]:
    order = np.argsort(values, kind="stable")
    adjusted = [0.0] * len(values)
    previous = 1.0
    for rank in range(len(values), 0, -1):
        index = int(order[rank - 1])
        previous = min(previous, len(values) * values[index] / rank)
        adjusted[index] = previous
    return adjusted


def bca(
    observed: float,
    bootstrap: np.ndarray[Any, np.dtype[np.float64]],
    jackknife: np.ndarray[Any, np.dtype[np.float64]],
) -> tuple[float | None, float | None, float | None, float | None, str]:
    normal = NormalDist()
    proportion = (np.sum(bootstrap < observed) + 0.5 * np.sum(bootstrap == observed)) / len(
        bootstrap
    )
    proportion = min(max(float(proportion), 0.5 / len(bootstrap)), 1 - 0.5 / len(bootstrap))
    bias = normal.inv_cdf(proportion)
    centered = jackknife.mean() - jackknife
    denominator = 6 * float(np.square(centered).sum()) ** 1.5
    if denominator == 0 or not math.isfinite(denominator):
        return None, None, bias, None, "unavailable_nonfinite_acceleration"
    acceleration = float(np.power(centered, 3).sum()) / denominator
    probabilities = []
    for probability in (0.025, 0.975):
        z = normal.inv_cdf(probability)
        divisor = 1 - acceleration * (bias + z)
        if divisor == 0:
            return None, None, bias, acceleration, "unavailable_zero_bca_denominator"
        probabilities.append(normal.cdf(bias + (bias + z) / divisor))
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities):
        return None, None, bias, acceleration, "unavailable_invalid_bca_probability"
    interval = np.quantile(bootstrap, probabilities, method="linear")
    return float(interval[0]), float(interval[1]), bias, acceleration, "available"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments/statistics"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    registry, devices, outcomes = reconstructed(root)
    contract = read_json(output / "contract.json")
    require(contract["accepted_models"] == list(MODELS), "contract models")
    require(contract["confirmatory_models"] == list(PRIMARY), "contract family")
    require(
        contract["device_count"] == 14 and contract["inference_unit"] == "physical test device",
        "unit",
    )
    require(contract["bootstrap"]["seed"] == SEED, "bootstrap seed")
    require(contract["bootstrap"]["replicates"] == REPLICATES, "bootstrap replicates")
    require(
        contract["runner_sha256"] == digest(root / "src/iot_diad/modeling/statistics.py"),
        "runner hash",
    )
    for relative, expected in contract["source_sha256"].items():
        require(digest(root / relative) == expected, f"contract source {relative}")
    require(
        contract["source_sha256"]["artifacts/experiments/baselines/accepted_model_registry.json"]
        == digest(root / "artifacts/experiments/baselines/accepted_model_registry.json"),
        "registry hash",
    )
    routed = [entry for entry in registry["models"] if entry["primary_learned_model"]]
    require(
        contract["model_routing"]
        == [{key: entry[key] for key in contract["model_routing"][0]} for entry in routed],
        "contract route records",
    )

    macro_rows = rows(output / "device_macro_f1.csv")
    recall_rows = rows(output / "device_class_recall.csv")
    require(len(macro_rows) == 70 and len(recall_rows) == 560, "outcome row counts")
    for row in macro_rows:
        record = outcomes[row["model_id"]][row["device"]]
        require(int(row["rows"]) == record["rows"], "exported device rows")
        close(float(row["macro_f1"]), record["macro_f1"], "exported macro-F1")
    for row in recall_rows:
        index = CLASSES.index(row["class"])
        record = outcomes[row["model_id"]][row["device"]]
        require(int(row["support"]) == int(record["support"][index]), "exported support")
        if record["support"][index]:
            close(float(row["recall"]), float(record["recall"][index]), "exported recall")
        else:
            require(row["recall"] == "", "unsupported recall")

    macro = {
        model: np.asarray([outcomes[model][d]["macro_f1"] for d in devices]) for model in MODELS
    }
    friedman = read_json(output / "friedman.json")
    scipy_result = friedmanchisquare(*(macro[model] for model in PRIMARY))
    close(float(friedman["statistic"]), float(scipy_result.statistic), "Friedman statistic")
    close(float(friedman["raw_p_value"]), float(scipy_result.pvalue), "Friedman p-value")

    confirmatory = rows(output / "confirmatory_comparisons.csv")
    differences = rows(output / "confirmatory_device_differences.csv")
    require(len(confirmatory) == 3 and len(differences) == 42, "confirmatory family size")
    raw = []
    for row, (model_a, model_b) in zip(confirmatory, PAIRS, strict=True):
        require(row["family"] == "confirmatory_model_comparison", "confirmatory family label")
        diff = macro[model_a] - macro[model_b]
        statistic, p_value, nonzero, zero = signed_rank(diff)
        close(float(row["statistic"]), statistic, "signed-rank statistic")
        close(float(row["raw_p_value"]), p_value, "signed-rank p-value")
        close(float(row["mean_difference"]), float(diff.mean()), "mean paired effect")
        require(
            int(row["nonzero_differences"]) == nonzero and int(row["zero_differences"]) == zero,
            "zero handling",
        )
        raw.append(p_value)
        selected = [item for item in differences if item["hypothesis_id"] == row["hypothesis_id"]]
        require([item["device"] for item in selected] == devices, "paired device order")
        for index, item in enumerate(selected):
            close(float(item["paired_difference"]), float(diff[index]), "paired difference")
    for row, adjusted in zip(confirmatory, holm(raw), strict=True):
        close(float(row["adjusted_p_value"]), adjusted, "Holm adjustment")

    exploratory = rows(output / "exploratory_recall_comparisons.csv")
    exploratory_differences = rows(output / "exploratory_recall_device_differences.csv")
    require(len(exploratory) == 24, "exploratory family size")
    raw = []
    for row in exploratory:
        class_index = CLASSES.index(row["class"])
        contributing = [
            d for d in devices if outcomes[row["model_a"]][d]["support"][class_index] > 0
        ]
        diff = np.asarray(
            [
                outcomes[row["model_a"]][d]["recall"][class_index]
                - outcomes[row["model_b"]][d]["recall"][class_index]
                for d in contributing
            ]
        )
        statistic, p_value, _, _ = signed_rank(diff)
        require(int(row["devices"]) == len(contributing), "exploratory device count")
        close(float(row["statistic"]), statistic, "exploratory statistic")
        close(float(row["raw_p_value"]), p_value, "exploratory p-value")
        selected = [
            item
            for item in exploratory_differences
            if item["hypothesis_id"] == row["hypothesis_id"]
        ]
        require([item["device"] for item in selected] == contributing, "exploratory pairing")
        raw.append(p_value)
    for row, adjusted in zip(exploratory, bh(raw), strict=True):
        close(float(row["adjusted_p_value"]), adjusted, "BH adjustment")

    samples = np.load(output / "bootstrap_device_indices.npy", allow_pickle=False)
    estimates = np.load(output / "bootstrap_estimates.npy", allow_pickle=False)
    expected_samples = np.random.default_rng(SEED).integers(
        0, 14, size=(REPLICATES, 14), dtype=np.int16
    )
    require(np.array_equal(samples, expected_samples), "bootstrap sample draw")
    vectors = [macro[model] for model in MODELS] + [macro[a] - macro[b] for a, b in PAIRS]
    expected_estimates = np.column_stack([vector[samples].mean(axis=1) for vector in vectors])
    require(np.array_equal(estimates, expected_estimates), "bootstrap estimates")
    summaries = rows(output / "bootstrap_summary.csv")
    loo = rows(output / "leave_one_device_out.csv")
    require(len(summaries) == 8 and len(loo) == 112, "bootstrap summary sizes")
    for column, (summary, vector) in enumerate(zip(summaries, vectors, strict=True)):
        observed = float(vector.mean())
        jackknife = np.asarray([np.delete(vector, index).mean() for index in range(14)])
        percentile = np.quantile(estimates[:, column], [0.025, 0.975], method="linear")
        expected_bca = bca(observed, estimates[:, column], jackknife)
        close(float(summary["observed_estimate"]), observed, "bootstrap observed")
        close(float(summary["percentile_lower"]), float(percentile[0]), "percentile lower")
        close(float(summary["percentile_upper"]), float(percentile[1]), "percentile upper")
        require(summary["bca_status"] == expected_bca[4], "BCa status")
        if expected_bca[0] is not None:
            upper = expected_bca[1]
            assert upper is not None
            close(float(summary["bca_lower"]), expected_bca[0], "BCa lower")
            close(float(summary["bca_upper"]), upper, "BCa upper")
        selected = [item for item in loo if item["estimand_id"] == summary["estimand_id"]]
        require([item["omitted_device"] for item in selected] == devices, "LOO devices")
        for index, item in enumerate(selected):
            close(
                float(item["leave_one_device_out_estimate"]),
                float(jackknife[index]),
                "LOO estimate",
            )

    manifest = read_json(output / "manifest.json")
    require(manifest["status"] == "complete", "manifest status")
    for record in manifest["files"]:
        path = output / record["path"]
        require(path.stat().st_size == record["size_bytes"], "manifest size")
        require(digest(path) == record["sha256"], "manifest hash")
    verification = {
        "status": "passed",
        "task_id": "STAT-001",
        "accepted_models": list(MODELS),
        "confirmatory_hypotheses": len(confirmatory),
        "exploratory_hypotheses": len(exploratory),
        "test_devices": len(devices),
        "bootstrap_replicates": REPLICATES,
        "contract_sha256": digest(output / "contract.json"),
        "manifest_sha256": digest(output / "manifest.json"),
        "verifier_sha256": digest(Path(__file__)),
        "checks": [
            "accepted registry routing and source hashes",
            "test physical-device identity and partition membership",
            "confusion-matrix metric reconstruction",
            "paired Friedman and exhaustive signed-rank calculations",
            "Holm and separate Benjamini-Hochberg families",
            (
                "complete-device sampling, percentile and BCa intervals, "
                "and leave-one-device-out estimates"
            ),
            "artifact hashes and manifest",
        ],
    }
    temporary = output / "verification.json.tmp"
    temporary.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output / "verification.json")
    print(json.dumps(verification, sort_keys=True))


if __name__ == "__main__":
    main()
