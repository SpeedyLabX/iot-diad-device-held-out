"""Generate conservative residual alias candidates from the published inventory."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT / "artifacts/device_mapping/literature"
SOURCES = ROOT / "artifacts/research/sources"
INVENTORY = MAPPING / "table1_inventory.csv"
BASE_CANDIDATES = MAPPING / "table1_identifier_candidates.csv"
PDF = SOURCES / "Device_Identification_and_Anomaly_Detection_in_IoT_Environments.pdf"
RAW_TEXT = SOURCES / "Device_Identification_and_Anomaly_Detection_in_IoT_Environments.raw.txt"
LAYOUT_TEXT = SOURCES / "Device_Identification_and_Anomaly_Detection_in_IoT_Environments.txt"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    inventory = read_csv(INVENTORY)
    candidates = read_csv(BASE_CANDIDATES)
    raw_text = RAW_TEXT.read_text(encoding="utf-8")
    normalized_paper = " ".join(raw_text.replace("-\n", "").split())

    required_evidence = (
        "network traffic data (PCAP files) from the state-of-the-art CICIoT2023 dataset",
        "All experiments were conducted using a 70% train-test split",
        "Both sets of experiments were performed using a 70:30 train-test split",
        "we identified 40 different device classes",
        "we identified 45 different devices",
    )
    missing = [phrase for phrase in required_evidence if phrase not in normalized_paper]
    if missing:
        raise ValueError(f"Expected full-text evidence is missing: {missing}")

    alias_rows: list[dict[str, str]] = []
    for local in candidates:
        if local["match_type"] != "unmatched" or local["identifier_kind"] != "text_label":
            continue
        local_tokens = tokens(local["identifier"])
        matches = [
            entry
            for entry in inventory
            if "alexa" in tokens(entry["published_name"])
            and [token for token in tokens(entry["published_name"]) if token != "alexa"]
            == local_tokens
        ]
        if not matches:
            continue
        alias_rows.append(
            {
                "identifier": local["identifier"],
                "local_row_count": local["local_row_count"],
                "local_capture_count": local["local_capture_count"],
                "alias_rule": "published_name_remove_token_alexa",
                "candidate_entry_count": str(len(matches)),
                "candidate_entry_ids_json": json.dumps([row["entry_id"] for row in matches]),
                "candidate_names_json": json.dumps([row["published_name"] for row in matches]),
                "candidate_macs_json": json.dumps(
                    [row["published_mac_normalized"] for row in matches]
                ),
                "source_doi": "10.3390/s23135941",
                "continuity_source_doi": "10.1109/JIOT.2024.3522863",
                "status": "deterministic_name_alias_candidate_not_approved",
                "canonical_device_id": "",
            }
        )

    if not alias_rows:
        raise ValueError("Expected at least one deterministic residual alias")
    output = MAPPING / "residual_alias_candidates.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(alias_rows[0]))
        writer.writeheader()
        writer.writerows(alias_rows)

    base_matched = [row for row in candidates if row["match_type"] != "unmatched"]
    base_matched_rows = sum(int(row["local_row_count"]) for row in base_matched)
    alias_identifiers = {row["identifier"] for row in alias_rows}
    alias_matched_rows = sum(int(row["local_row_count"]) for row in alias_rows)
    total_rows = sum(int(row["local_row_count"]) for row in candidates)
    remaining = [
        row
        for row in candidates
        if row["match_type"] == "unmatched" and row["identifier"] not in alias_identifiers
    ]
    remaining_macs = [row for row in remaining if row["identifier_kind"] == "mac_form"]
    local_admin_macs = [
        row for row in remaining_macs if int(row["identifier"].split(":", 1)[0], 16) & 0x02
    ]
    summary = {
        "retrieval_date": "2026-10-05",
        "paper_doi": "10.1109/JIOT.2024.3522863",
        "paper_pages": 19,
        "paper_sha256": sha256(PDF),
        "layout_text_sha256": sha256(LAYOUT_TEXT),
        "raw_text_sha256": sha256(RAW_TEXT),
        "provenance": (
            "PDF supplied by project lead; IEEE licensed copy downloaded via "
            "RMIT University Library"
        ),
        "paper_evidence": {
            "dataset_continuity": (
                "CICIoT2023 PCAP files are the stated source; pp. 13627 and 13636"
            ),
            "device_target": "device_mac is the multiclass target; p. 13639",
            "device_split": "70:30 train-test split over samples; pp. 13639-13640",
            "held_out_physical_devices": "not reported",
            "preprocessing_fit_scope": "not reported",
            "device_count_discrepancy": (
                "40 device classes on p. 13636 versus 45 devices/classes on pp. 13639-13640"
            ),
        },
        "base_candidate_identifiers": len(base_matched),
        "base_candidate_rows": base_matched_rows,
        "new_alias_candidate_identifiers": len(alias_rows),
        "new_alias_candidate_rows": alias_matched_rows,
        "combined_candidate_identifiers": len(base_matched) + len(alias_rows),
        "combined_candidate_rows": base_matched_rows + alias_matched_rows,
        "combined_candidate_row_fraction": (base_matched_rows + alias_matched_rows) / total_rows,
        "remaining_unmatched_identifiers": len(remaining),
        "remaining_unmatched_rows": sum(int(row["local_row_count"]) for row in remaining),
        "remaining_unmatched_mac_identifiers": len(remaining_macs),
        "remaining_unmatched_mac_rows": sum(int(row["local_row_count"]) for row in remaining_macs),
        "remaining_locally_administered_mac_identifiers": len(local_admin_macs),
        "remaining_locally_administered_mac_rows": sum(
            int(row["local_row_count"]) for row in local_admin_macs
        ),
        "remaining_text_identifiers": [
            row["identifier"] for row in remaining if row["identifier_kind"] == "text_label"
        ],
        "approved_identities": 0,
    }
    (MAPPING / "full_text_resolution_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
