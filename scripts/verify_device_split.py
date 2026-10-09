"""Verify the candidate device-held-out split independently from the optimizer."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "artifacts/splits"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    mapping = read_csv(ROOT / "artifacts/device_mapping/approved_packet_device_mapping.csv")
    approved = {
        row["canonical_device_id"]: row
        for row in mapping
        if row["disposition"] == "approved_primary_packet_mapping"
    }
    excluded = [row for row in mapping if row["canonical_device_id"] == ""]
    manifest_path = SPLITS / "device_split_manifest.csv"
    class_path = SPLITS / "class_partition_counts.csv"
    excluded_path = SPLITS / "excluded_packet_identifiers.csv"
    manifest = read_csv(manifest_path)
    class_counts = read_csv(class_path)
    exclusions = read_csv(excluded_path)
    summary = json.loads((SPLITS / "split_summary.json").read_text(encoding="utf-8"))

    manifest_ids = [row["canonical_device_id"] for row in manifest]
    assert len(manifest) == len(manifest_ids) == len(set(manifest_ids)) == 68
    assert set(manifest_ids) == set(approved)
    assert sum(int(row["row_count"]) for row in manifest) == 34_627_307
    assert {row["partition"] for row in manifest} == {"train", "calibration", "test"}
    expected_sizes = {"train": 40, "calibration": 14, "test": 14}
    actual_sizes: dict[str, int] = defaultdict(int)
    for row in manifest:
        actual_sizes[row["partition"]] += 1
        source = approved[row["canonical_device_id"]]
        assert row["local_identifier"] == source["identifier"]
        assert int(row["row_count"]) == int(source["row_count"])
    assert dict(actual_sizes) == expected_sizes

    assert len(exclusions) == len(excluded) == 47
    assert sum(int(row["row_count"]) for row in exclusions) == 1_045_532
    assert {row["identifier"] for row in exclusions} == {row["identifier"] for row in excluded}

    partition_rows: dict[str, int] = defaultdict(int)
    for row in manifest:
        partition_rows[row["partition"]] += int(row["row_count"])
    class_partition_rows: dict[str, int] = defaultdict(int)
    for row in class_counts:
        class_partition_rows[row["partition"]] += int(row["row_count"])
        minimum = 10 if row["partition"] == "train" else 5
        assert int(row["source_device_count"]) >= minimum
    assert dict(class_partition_rows) == dict(partition_rows)
    assert sum(partition_rows.values()) == summary["eligible_row_count"] == 34_627_307
    assert summary["excluded_identifier_count"] == 47
    assert summary["excluded_row_count"] == 1_045_532
    assert summary["manifest_sha256"] == sha256(manifest_path)
    assert summary["class_counts_sha256"] == sha256(class_path)
    assert summary["excluded_sha256"] == sha256(excluded_path)
    print(
        json.dumps(
            {
                "status": "passed",
                "partition_devices": actual_sizes,
                "partition_rows": partition_rows,
                "eligible_rows": sum(partition_rows.values()),
                "excluded_rows": summary["excluded_row_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
