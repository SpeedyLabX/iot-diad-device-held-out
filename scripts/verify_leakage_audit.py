"""Independently reconcile split-isolation and leakage artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest_path = ROOT / "artifacts/splits/device_split_manifest.csv"
    coverage_path = ROOT / "artifacts/device_mapping/packet_identifier_coverage.csv"
    leakage_dir = ROOT / "artifacts/leakage"
    overlap_path = leakage_dir / "capture_partition_overlap.csv"
    duplicate_path = leakage_dir / "verified_cross_partition_duplicates.csv"
    manifest = read_csv(manifest_path)
    coverage = read_csv(coverage_path)
    overlap = read_csv(overlap_path)
    duplicates = read_csv(duplicate_path)
    summary = json.loads((leakage_dir / "leakage_summary.json").read_text(encoding="utf-8"))

    identifiers = {row["local_identifier"]: row["partition"] for row in manifest}
    assert len(manifest) == len(identifiers) == 68
    assert len({row["canonical_device_id"] for row in manifest}) == 68

    capture_partitions: dict[str, set[str]] = defaultdict(set)
    graph: dict[str, set[str]] = defaultdict(set)
    for row in coverage:
        identifier = row["identifier"]
        if identifier not in identifiers:
            continue
        capture = row["capture_id"]
        capture_partitions[capture].add(identifiers[identifier])
        left = f"device:{identifier}"
        right = f"capture:{capture}"
        graph[left].add(right)
        graph[right].add(left)
    assert len(capture_partitions) == 180
    assert all(values == {"train", "calibration", "test"} for values in capture_partitions.values())

    seen: set[str] = set()
    pending = [next(iter(graph))]
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        pending.extend(graph[node] - seen)
    assert seen == set(graph)
    assert sum(node.startswith("device:") for node in seen) == 68
    assert sum(node.startswith("capture:") for node in seen) == 180

    assert len(overlap) == 180
    assert all(int(row["partition_count"]) == 3 for row in overlap)
    assert summary["manifest_sha256"] == sha256(manifest_path)
    assert summary["coverage_sha256"] == sha256(coverage_path)
    assert summary["capture_overlap_sha256"] == sha256(overlap_path)
    assert summary["identifier_overlap_count"] == 0
    assert summary["physical_device_overlap_count"] == 0
    assert summary["nontrivial_device_and_capture_disjoint_split_possible"] is False

    cross = pl.read_parquet(leakage_dir / "cross_partition_feature_hashes.parquet")
    assert cross.height == 8
    assert len(duplicates) == 58
    duplicate_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in duplicates:
        duplicate_groups[(row["hash_a"], row["hash_b"])].append(row)
    assert len(duplicate_groups) == 8
    assert all(
        len({row["feature_vector_sha256"] for row in rows}) == 1
        for rows in duplicate_groups.values()
    )
    assert all(len({row["partition"] for row in rows}) > 1 for rows in duplicate_groups.values())
    assert all(
        len({row["coarse_label"] for row in rows}) == 1 for rows in duplicate_groups.values()
    )
    raw_check = summary["raw_duplicate_verification"]
    assert raw_check["status"] == "passed"
    assert raw_check["cross_partition_feature_vector_count"] == 8
    assert raw_check["affected_rows"] == 58
    assert raw_check["verified_rows_sha256"] == sha256(duplicate_path)
    print(
        json.dumps(
            {
                "status": "passed",
                "approved_devices": 68,
                "captures_shared_by_all_partitions": 180,
                "device_capture_components": 1,
                "cross_partition_feature_vectors": 8,
                "affected_rows": 58,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
