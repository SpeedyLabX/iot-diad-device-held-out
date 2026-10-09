"""Verify the approved Gate 2 protocol and symmetric duplicate exclusion."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "artifacts/splits"
LEAKAGE = ROOT / "artifacts/leakage"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source_path = LEAKAGE / "verified_cross_partition_duplicates.csv"
    exclusion_path = SPLITS / "cross_partition_duplicate_exclusions.csv"
    source = read_csv(source_path)
    exclusions = read_csv(exclusion_path)
    summary = json.loads((SPLITS / "gate2_decision_summary.json").read_text(encoding="utf-8"))

    source_keys = {(row["capture_id"], row["row_index"]) for row in source}
    exclusion_keys = {(row["capture_id"], row["row_index"]) for row in exclusions}
    assert len(source) == len(source_keys) == len(exclusions) == len(exclusion_keys) == 58
    assert source_keys == exclusion_keys
    assert all(
        row["reason"] == "exact_candidate_feature_vector_observed_in_multiple_partitions"
        for row in exclusions
    )

    vector_partitions: dict[str, set[str]] = defaultdict(set)
    for row in exclusions:
        vector_partitions[row["feature_vector_sha256"]].add(row["partition"])
    assert len(vector_partitions) == 8
    assert all(len(partitions) > 1 for partitions in vector_partitions.values())

    removed = Counter(row["partition"] for row in exclusions)
    assert dict(sorted(removed.items())) == {"calibration": 13, "test": 21, "train": 24}
    assert summary["excluded_duplicate_rows"] == 58
    assert summary["excluded_duplicate_rows_by_partition"] == dict(sorted(removed.items()))
    assert summary["filtered_rows_by_partition"] == {
        "calibration": 6_620_817,
        "test": 7_912_338,
        "train": 20_094_094,
    }
    assert summary["filtered_eligible_rows"] == 34_627_249
    assert summary["source_sha256"] == sha256(source_path)
    assert summary["exclusion_manifest_sha256"] == sha256(exclusion_path)
    assert summary["split_protocol"] == "physical-device-disjoint and capture-shared"
    print(
        json.dumps(
            {
                "status": "passed",
                "excluded_rows": 58,
                "excluded_vectors": 8,
                "filtered_eligible_rows": summary["filtered_eligible_rows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
