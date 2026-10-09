"""Verify the full-paper evidence and residual alias artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT / "artifacts/device_mapping/literature"
SOURCES = ROOT / "artifacts/research/sources"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


def main() -> None:
    pdf = SOURCES / "Device_Identification_and_Anomaly_Detection_in_IoT_Environments.pdf"
    raw_path = SOURCES / "Device_Identification_and_Anomaly_Detection_in_IoT_Environments.raw.txt"
    raw_pages = raw_path.read_text(encoding="utf-8").split("\f")
    summary_path = MAPPING / "full_text_resolution_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    aliases = read_csv(MAPPING / "residual_alias_candidates.csv")
    inventory = {row["entry_id"]: row for row in read_csv(MAPPING / "table1_inventory.csv")}

    assert pdf.read_bytes().startswith(b"%PDF-1.4")
    assert len([page for page in raw_pages if page.strip()]) == 19
    assert sha256(pdf) == summary["paper_sha256"]
    assert sha256(raw_path) == summary["raw_text_sha256"]

    article_13627 = raw_pages[2]
    article_13636 = raw_pages[11]
    article_13639 = raw_pages[14]
    article_13640 = raw_pages[15]
    assert "PCAP" in article_13627 and "CICIoT2023 attack" in article_13627
    assert "70% train-test split" in article_13636
    assert "40 different\ndevice classes" in article_13636
    assert "device_mac" in article_13639 and "45 different devices" in article_13639
    assert "70:30 train-test split" in article_13640
    all_text = " ".join(raw_pages)
    assert "device-disjoint" not in all_text.casefold()
    assert "held-out physical" not in all_text.casefold()

    assert len(aliases) == 4
    assert sum(int(row["local_row_count"]) for row in aliases) == 1_169_431
    expected = {
        "Amazon Echo Dot 1": "Amazon Alexa Echo Dot 1",
        "Amazon Echo Dot 2": "Amazon Alexa Echo Dot 2",
        "Amazon Echo Spot": "Amazon Alexa Echo Spot",
        "Amazon Echo Studio": "Amazon Alexa Echo Studio",
    }
    assert {row["identifier"] for row in aliases} == set(expected)
    for row in aliases:
        ids = json.loads(row["candidate_entry_ids_json"])
        assert len(ids) == int(row["candidate_entry_count"]) == 1
        entry = inventory[ids[0]]
        assert entry["published_name"] == expected[row["identifier"]]
        assert [token for token in words(entry["published_name"]) if token != "alexa"] == words(
            row["identifier"]
        )
        assert row["canonical_device_id"] == ""
        assert row["status"] == "deterministic_name_alias_candidate_not_approved"

    assert summary["combined_candidate_identifiers"] == 69
    assert summary["combined_candidate_rows"] == 34_663_062
    assert summary["remaining_unmatched_identifiers"] == 46
    assert summary["remaining_unmatched_rows"] == 1_009_777
    assert summary["remaining_unmatched_mac_identifiers"] == 45
    assert summary["remaining_unmatched_mac_rows"] == 788_935
    assert summary["remaining_locally_administered_mac_identifiers"] == 14
    assert summary["remaining_locally_administered_mac_rows"] == 292_966
    assert summary["remaining_text_identifiers"] == ["not sure"]
    assert summary["approved_identities"] == 0

    result = {
        "status": "passed",
        "paper_sha256": summary["paper_sha256"],
        "paper_pages": 19,
        "page_evidence_checked": [13627, 13636, 13639, 13640],
        "alias_candidates": 4,
        "alias_rows": 1_169_431,
        "combined_candidate_identifiers": 69,
        "combined_candidate_rows": 34_663_062,
        "approved_identities": 0,
    }
    (MAPPING / "full_text_coordinator_verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
