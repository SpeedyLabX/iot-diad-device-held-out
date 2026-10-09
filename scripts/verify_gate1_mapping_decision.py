"""Independently verify the approved Gate 1 mapping artifact and its conservation totals."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT / "artifacts/device_mapping"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = read_csv(MAPPING / "approved_packet_device_mapping.csv")
    summary = json.loads((MAPPING / "gate1_decision_summary.json").read_text(encoding="utf-8"))
    by_disposition: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_disposition.setdefault(row["disposition"], []).append(row)

    approved = by_disposition["approved_primary_packet_mapping"]
    ambiguous = by_disposition["excluded_primary_ambiguous_identity"]
    unresolved = by_disposition["excluded_primary_unresolved_identity"]

    assert len(rows) == summary["total_identifiers"] == 115
    assert sum(int(row["row_count"]) for row in rows) == summary["total_rows"] == 35_672_839
    assert len(approved) == summary["approved_identifiers"] == 68
    assert sum(int(row["row_count"]) for row in approved) == summary["approved_rows"] == 34_627_307
    assert len(ambiguous) == summary["ambiguous_excluded_identifiers"] == 1
    assert sum(int(row["row_count"]) for row in ambiguous) == 35_755
    assert ambiguous[0]["identifier"] == "Raspberry Pi 4-2 GB"
    assert len(unresolved) == summary["unresolved_excluded_identifiers"] == 46
    assert sum(int(row["row_count"]) for row in unresolved) == 1_009_777
    assert {row["identifier"] for row in unresolved if row["identifier_kind"] == "text_label"} == {
        "not sure"
    }
    assert len({row["canonical_device_id"] for row in approved}) == 68
    assert all(row["canonical_device_id"] for row in approved)
    assert all(not row["canonical_device_id"] for row in ambiguous + unresolved)
    assert all(row["rationale"] for row in rows)
    assert summary["primary_evaluation"] == "packet physical-device-held-out"
    print(
        json.dumps(
            {
                "status": "passed",
                "approved_identifiers": len(approved),
                "approved_rows": sum(int(row["row_count"]) for row in approved),
                "excluded_identifiers": len(ambiguous) + len(unresolved),
                "excluded_rows": sum(int(row["row_count"]) for row in ambiguous + unresolved),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
