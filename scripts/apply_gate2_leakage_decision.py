"""Materialize the approved symmetric cross-partition duplicate exclusion."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEAKAGE = ROOT / "artifacts/leakage"
SPLITS = ROOT / "artifacts/splits"
SOURCE = LEAKAGE / "verified_cross_partition_duplicates.csv"
OUTPUT = SPLITS / "cross_partition_duplicate_exclusions.csv"
SUMMARY = SPLITS / "gate2_decision_summary.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows = read_csv(SOURCE)
    exclusions = [
        {
            "capture_id": row["capture_id"],
            "row_index": row["row_index"],
            "partition": row["partition"],
            "local_identifier": row["device_mac"],
            "coarse_label": row["coarse_label"],
            "feature_vector_sha256": row["feature_vector_sha256"],
            "reason": "exact_candidate_feature_vector_observed_in_multiple_partitions",
        }
        for row in rows
    ]
    exclusions.sort(key=lambda row: (row["capture_id"], int(row["row_index"])))
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exclusions[0]))
        writer.writeheader()
        writer.writerows(exclusions)

    removed = Counter(row["partition"] for row in exclusions)
    original = Counter()
    for row in read_csv(SPLITS / "device_split_manifest.csv"):
        original[row["partition"]] += int(row["row_count"])
    filtered = {partition: original[partition] - removed[partition] for partition in original}
    summary = {
        "decision_date": "2026-10-05",
        "decision_authority": "project lead",
        "split_protocol": "physical-device-disjoint and capture-shared",
        "claim_boundary": (
            "unseen physical devices within shared captures and attack scenarios; no unseen-"
            "capture, unseen-environment, or temporal-generalization claim"
        ),
        "duplicate_rule": (
            "exclude every row in every partition whose complete candidate-feature vector occurs "
            "in more than one partition"
        ),
        "excluded_duplicate_rows": len(exclusions),
        "excluded_duplicate_rows_by_partition": dict(sorted(removed.items())),
        "unfiltered_rows_by_partition": dict(sorted(original.items())),
        "filtered_rows_by_partition": dict(sorted(filtered.items())),
        "filtered_eligible_rows": sum(filtered.values()),
        "source_sha256": sha256(SOURCE),
        "exclusion_manifest_sha256": sha256(OUTPUT),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
