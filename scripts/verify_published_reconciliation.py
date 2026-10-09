"""Independently verify published-inventory reconciliation artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/device_mapping/literature"
LOCAL = ROOT / "artifacts/device_mapping/packet_identifier_summary.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def main() -> None:
    inventory = read_csv(OUT / "table1_inventory.csv")
    candidates = read_csv(OUT / "table1_identifier_candidates.csv")
    local = read_csv(LOCAL)
    summary = json.loads((OUT / "reconciliation_summary.json").read_text(encoding="utf-8"))

    assert len(inventory) == 110
    assert len(candidates) == len(local) == 115
    assert len({row["entry_id"] for row in inventory}) == 110
    assert sum(bool(row["published_mac_normalized"]) for row in inventory) == 70
    assert Counter(row["publisher_pdf_page"] for row in inventory) == {"6": 62, "7": 48}

    local_by_id = {row["identifier"]: row for row in local}
    assert set(local_by_id) == {row["identifier"] for row in candidates}
    assert sum(int(row["row_count"]) for row in local) == 35_672_839
    assert sum(int(row["local_row_count"]) for row in candidates) == 35_672_839

    match_counts: Counter[str] = Counter()
    match_rows: Counter[str] = Counter()
    for row in candidates:
        original = local_by_id[row["identifier"]]
        assert row["local_row_count"] == original["row_count"]
        assert row["local_capture_count"] == original["capture_count"]
        ids = json.loads(row["candidate_entry_ids_json"])
        names = json.loads(row["candidate_names_json"])
        macs = json.loads(row["candidate_macs_json"])
        pages = json.loads(row["candidate_pdf_pages_json"])
        matched = [entry for entry in inventory if entry["entry_id"] in ids]
        assert len(ids) == len(matched) == int(row["candidate_entry_count"])
        assert names == [entry["published_name"] for entry in matched]
        assert macs == [entry["published_mac_normalized"] for entry in matched]
        assert pages == sorted({entry["publisher_pdf_page"] for entry in matched})
        assert row["canonical_device_id"] == ""
        if matched:
            assert row["status"] == "cross_release_candidate_not_approved"
        else:
            assert row["status"] == "unresolved_unmatched"

        if row["match_type"] == "exact_name":
            assert all(entry["published_name"] == row["identifier"] for entry in matched)
        elif row["match_type"] == "mac":
            assert all(
                entry["published_mac_normalized"] == row["identifier"].lower() for entry in matched
            )
        elif row["match_type"] == "normalized_name":
            assert all(
                normalize_name(entry["published_name"]) == normalize_name(row["identifier"])
                for entry in matched
            )
        else:
            assert row["match_type"] == "unmatched" and not matched
        match_counts[row["match_type"]] += 1
        match_rows[row["match_type"]] += int(row["local_row_count"])

    assert dict(sorted(match_counts.items())) == summary["match_type_identifier_counts"]
    assert dict(sorted(match_rows.items())) == summary["match_type_local_row_counts"]
    assert summary["approved_identities"] == 0
    assert summary["ambiguous_matched_identifiers"] == ["Raspberry Pi 4-2 GB"]
    assert sum(match_rows[k] for k in ("exact_name", "mac", "normalized_name")) == 33_493_631

    inventory_by_name = {row["published_name"]: row for row in inventory}
    assert inventory_by_name["Arlo Q Indoor Camera"]["published_mac_normalized"] == (
        "40:5d:82:35:14:c8"
    )
    assert inventory_by_name["Wyze Camera"]["published_mac_normalized"] == "7c:78:b2:86:0d:81"
    assert sum(normalize_name(row["published_name"]) == "raspberrypi42gb" for row in inventory) == 8

    for relative, expected in summary["hashes_sha256"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected, relative

    result = {
        "status": "passed",
        "inventory_entries": len(inventory),
        "local_identifiers": len(candidates),
        "local_packet_rows": sum(match_rows.values()),
        "candidate_matched_identifiers": sum(
            match_counts[k] for k in ("exact_name", "mac", "normalized_name")
        ),
        "candidate_matched_rows": sum(
            match_rows[k] for k in ("exact_name", "mac", "normalized_name")
        ),
        "ambiguous_identifiers": summary["ambiguous_matched_identifiers"],
        "approved_identities": 0,
        "checks": [
            "inventory and local counts conserve",
            "candidate arrays and pages resolve to inventory entries",
            "match predicates hold for every identifier",
            "all canonical identities remain blank",
            "source and script hashes match summary",
            "visually inspected Table 1 anchor values are present",
        ],
    }
    (OUT / "coordinator_verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
