"""Deterministic physical-device split construction for the approved packet population."""

from __future__ import annotations

import csv
import hashlib
import json
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

PARTITIONS = ("train", "calibration", "test")


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    random_candidates: int
    partition_sizes: tuple[int, int, int]
    minimum_support: tuple[int, int, int]
    total_rows_weight: float
    class_rows_weight: float
    class_support_weight: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> SplitConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))["split"]
    sizes = raw["partition_sizes"]
    support = raw["minimum_class_device_support"]
    weights = raw["objective_weights"]
    return SplitConfig(
        seed=int(raw["seed"]),
        random_candidates=int(raw["random_candidates"]),
        partition_sizes=tuple(int(sizes[name]) for name in PARTITIONS),  # type: ignore[arg-type]
        minimum_support=tuple(int(support[name]) for name in PARTITIONS),  # type: ignore[arg-type]
        total_rows_weight=float(weights["total_rows"]),
        class_rows_weight=float(weights["class_rows"]),
        class_support_weight=float(weights["class_device_support"]),
    )


def split_score(
    matrix: NDArray[np.float64],
    assignment: NDArray[np.int64],
    config: SplitConfig,
) -> float:
    support_matrix = (matrix > 0).astype(np.float64)
    class_totals = matrix.sum(axis=0)
    support_totals = support_matrix.sum(axis=0)
    all_rows = float(class_totals.sum())
    fractions = np.asarray(config.partition_sizes, dtype=np.float64) / matrix.shape[0]
    score = 0.0
    for index, target_fraction in enumerate(fractions):
        selected = assignment == index
        class_rows = matrix[selected].sum(axis=0)
        class_support = support_matrix[selected].sum(axis=0)
        row_fraction = float(class_rows.sum()) / all_rows
        score += config.total_rows_weight * (row_fraction - target_fraction) ** 2
        score += config.class_rows_weight * float(
            np.square(class_rows / class_totals - target_fraction).sum()
        )
        score += config.class_support_weight * float(
            np.square(class_support / support_totals - target_fraction).sum()
        )
        missing_support = int((class_support < config.minimum_support[index]).sum())
        score += 10.0 * missing_support
    return score


def optimize_assignments(
    matrix: NDArray[np.float64], config: SplitConfig
) -> tuple[NDArray[np.int64], float]:
    if sum(config.partition_sizes) != matrix.shape[0]:
        raise ValueError("Partition sizes must equal the number of approved devices")
    if np.any(matrix.sum(axis=0) == 0):
        raise ValueError("Every modeled class must have at least one row")

    labels = np.repeat(np.arange(len(PARTITIONS), dtype=np.int64), config.partition_sizes)
    rng = np.random.default_rng(config.seed)
    best_score = float("inf")
    best_assignment: NDArray[np.int64] | None = None
    for _ in range(config.random_candidates):
        candidate = rng.permutation(labels)
        candidate_score = split_score(matrix, candidate, config)
        if candidate_score < best_score:
            best_score = candidate_score
            best_assignment = candidate.copy()
    if best_assignment is None:
        raise ValueError("random_candidates must be positive")

    improved = True
    while improved:
        improved = False
        best_swap: tuple[int, int] | None = None
        swap_score = best_score
        for left in range(matrix.shape[0]):
            for right in range(left + 1, matrix.shape[0]):
                if best_assignment[left] == best_assignment[right]:
                    continue
                best_assignment[left], best_assignment[right] = (
                    best_assignment[right],
                    best_assignment[left],
                )
                candidate_score = split_score(matrix, best_assignment, config)
                best_assignment[left], best_assignment[right] = (
                    best_assignment[right],
                    best_assignment[left],
                )
                if candidate_score < swap_score - 1e-15:
                    swap_score = candidate_score
                    best_swap = (left, right)
        if best_swap is not None:
            left, right = best_swap
            best_assignment[left], best_assignment[right] = (
                best_assignment[right],
                best_assignment[left],
            )
            best_score = swap_score
            improved = True
    return best_assignment, best_score


def load_device_class_matrix(
    mapping_path: Path, coverage_path: Path, inventory_path: Path
) -> tuple[list[dict[str, str]], list[str], NDArray[np.float64], list[dict[str, str]]]:
    mapping = read_csv(mapping_path)
    approved = {
        row["identifier"]: row
        for row in mapping
        if row["disposition"] == "approved_primary_packet_mapping"
    }
    excluded = [row for row in mapping if row["identifier"] not in approved]
    capture_labels = {
        row["capture_id"]: row["coarse_label"]
        for row in read_csv(inventory_path)
        if row["feature_view"] == "packet"
    }
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in read_csv(coverage_path):
        if row["identifier"] not in approved:
            continue
        capture_id = row["capture_id"]
        if capture_id not in capture_labels:
            raise ValueError(f"Missing packet capture metadata: {capture_id}")
        canonical_id = approved[row["identifier"]]["canonical_device_id"]
        counts[canonical_id][capture_labels[capture_id]] += int(row["row_count"])

    devices = sorted(approved.values(), key=lambda row: row["canonical_device_id"])
    classes = sorted({label for values in counts.values() for label in values})
    matrix = np.asarray(
        [[counts[row["canonical_device_id"]][label] for label in classes] for row in devices],
        dtype=np.float64,
    )
    expected_rows = sum(int(row["row_count"]) for row in devices)
    if int(matrix.sum()) != expected_rows:
        raise ValueError("Approved mapping and packet coverage row totals disagree")
    return devices, classes, matrix, excluded


def write_outputs(
    root: Path,
    config_path: Path,
    config: SplitConfig,
    devices: list[dict[str, str]],
    classes: list[str],
    matrix: NDArray[np.float64],
    excluded: list[dict[str, str]],
    assignment: NDArray[np.int64],
    objective: float,
) -> None:
    output = root / "artifacts/splits"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "device_split_manifest.csv"
    manifest_rows: list[dict[str, str]] = []
    for index, device in enumerate(devices):
        manifest_rows.append(
            {
                "canonical_device_id": device["canonical_device_id"],
                "local_identifier": device["identifier"],
                "published_name": device["published_name"],
                "published_mac": device["published_mac"],
                "partition": PARTITIONS[int(assignment[index])],
                "row_count": str(int(matrix[index].sum())),
                "capture_count": device["capture_count"],
                "supported_coarse_classes": str(int((matrix[index] > 0).sum())),
            }
        )
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    class_rows: list[dict[str, str]] = []
    class_totals = matrix.sum(axis=0)
    for partition_index, partition in enumerate(PARTITIONS):
        selected = assignment == partition_index
        for class_index, label in enumerate(classes):
            row_count = int(matrix[selected, class_index].sum())
            class_rows.append(
                {
                    "partition": partition,
                    "coarse_label": label,
                    "row_count": str(row_count),
                    "source_device_count": str(int((matrix[selected, class_index] > 0).sum())),
                    "total_class_rows": str(int(class_totals[class_index])),
                    "row_fraction": f"{row_count / class_totals[class_index]:.12f}",
                }
            )
    class_path = output / "class_partition_counts.csv"
    with class_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)

    excluded_path = output / "excluded_packet_identifiers.csv"
    excluded_fields = [
        "identifier",
        "identifier_kind",
        "row_count",
        "disposition",
        "rationale",
    ]
    with excluded_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=excluded_fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in excluded_fields} for row in excluded)

    partition_summary: dict[str, dict[str, Any]] = {}
    for index, partition in enumerate(PARTITIONS):
        selected = assignment == index
        partition_summary[partition] = {
            "device_count": int(selected.sum()),
            "row_count": int(matrix[selected].sum()),
            "row_fraction": float(matrix[selected].sum() / matrix.sum()),
            "minimum_class_device_support": int(
                min(
                    (matrix[selected, class_index] > 0).sum() for class_index in range(len(classes))
                )
            ),
        }
    largest_share = max(
        float(matrix[:, class_index].max() / class_totals[class_index])
        for class_index in range(len(classes))
    )
    summary = {
        "generated_at": "2026-10-05",
        "seed": config.seed,
        "random_candidates": config.random_candidates,
        "objective_score": objective,
        "objective": (
            "fixed device counts; minimize total-row, per-class-row, and per-class device-support "
            "fraction deviations; enforce minimum class device support"
        ),
        "approved_device_count": len(devices),
        "eligible_row_count": int(matrix.sum()),
        "excluded_identifier_count": len(excluded),
        "excluded_row_count": sum(int(row["row_count"]) for row in excluded),
        "partitions": partition_summary,
        "coarse_labels": classes,
        "largest_single_device_class_row_fraction": largest_share,
        "config_sha256": sha256(config_path),
        "approved_mapping_sha256": sha256(
            root / "artifacts/device_mapping/approved_packet_device_mapping.csv"
        ),
        "coverage_sha256": sha256(root / "artifacts/device_mapping/packet_identifier_coverage.csv"),
        "manifest_sha256": sha256(manifest_path),
        "class_counts_sha256": sha256(class_path),
        "excluded_sha256": sha256(excluded_path),
    }
    (output / "split_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    config_path = root / "config/split.toml"
    config = load_config(config_path)
    devices, classes, matrix, excluded = load_device_class_matrix(
        root / "artifacts/device_mapping/approved_packet_device_mapping.csv",
        root / "artifacts/device_mapping/packet_identifier_coverage.csv",
        root / "artifacts/data_audit/file_inventory.csv",
    )
    assignment, objective = optimize_assignments(matrix, config)
    write_outputs(
        root, config_path, config, devices, classes, matrix, excluded, assignment, objective
    )
    print((root / "artifacts/splits/split_summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
