"""Reconcile local identifiers with archived CICIoT2023 Table 1; approve no identities."""

import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET  # nosec B405: pinned archive, DTD rejected below
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/device_mapping/literature"
LOCAL = ROOT / "artifacts/device_mapping/packet_identifier_summary.csv"
DOI = "10.3390/s23135941"
XML_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10346235/fullTextXML"


def normalize_name(value: str) -> str:
    """Ignore case and non-alphanumeric separators; never remove lexical tokens."""
    return "".join(c for c in value.casefold() if c.isalnum())


def normalize_mac(value: str) -> str:
    return "".join(value.split()).lower()


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_inventory() -> list[dict[str, str]]:
    data = (OUT / "ciciot2023.xml").read_bytes()
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ValueError("DTD/entity declarations are not allowed in this archive")
    root = ET.fromstring(data)  # nosec B314: DTD and entity declarations rejected
    table = root.find('.//table-wrap[@id="sensors-23-05941-t001"]')
    if table is None:
        raise ValueError("Expected Table 1 is missing")
    inventory = []
    for number, row in enumerate(table.findall(".//tr")[1:], start=1):
        cells = ["".join(cell.itertext()).strip() for cell in row]
        if len(cells) != 7:
            raise ValueError(f"Unexpected Table 1 structure at row {number}")
        for side, start in (("left", 1), ("right", 4)):
            name, category, mac = cells[start : start + 3]
            normalized_mac = "" if mac == "N/A" else normalize_mac(mac)
            if normalized_mac and not re.fullmatch(
                r"(?:[0-9a-f]{2}:){5}[0-9a-f]{2}", normalized_mac
            ):
                raise ValueError(f"Invalid MAC at row {number}: {mac}")
            inventory.append(
                {
                    "entry_id": f"T1-{number:02d}-{side}",
                    "published_name": name,
                    "published_category": category,
                    "published_mac_verbatim": mac,
                    "published_mac_normalized": normalized_mac,
                    "table_body_row": str(number),
                    "table_side": side,
                    "source_doi": DOI,
                    "source_table": "1",
                    "publisher_pdf_page": "6" if number <= 31 else "7",
                    "mirror_pdf_page": "6" if number <= 31 else "7",
                }
            )
    return inventory


def match_identifiers(inventory: list[dict[str, str]]) -> list[dict[str, str]]:
    with LOCAL.open(newline="", encoding="utf-8") as handle:
        local = list(csv.DictReader(handle))
    candidates = []
    for row in local:
        identifier = row["identifier"]
        matches = [p for p in inventory if p["published_name"] == identifier]
        match_type = "exact_name"
        if not matches and row["identifier_kind"] == "mac_form":
            matches = [
                p for p in inventory if p["published_mac_normalized"] == normalize_mac(identifier)
            ]
            match_type = "mac"
        if not matches and row["identifier_kind"] == "text_label":
            matches = [
                p
                for p in inventory
                if normalize_name(p["published_name"]) == normalize_name(identifier)
            ]
            match_type = "normalized_name"
        candidates.append(
            {
                "identifier": identifier,
                "identifier_kind": row["identifier_kind"],
                "local_row_count": row["row_count"],
                "local_capture_count": row["capture_count"],
                "match_type": match_type if matches else "unmatched",
                "candidate_entry_count": str(len(matches)),
                "candidate_entry_ids_json": json.dumps([p["entry_id"] for p in matches]),
                "candidate_names_json": json.dumps([p["published_name"] for p in matches]),
                "candidate_macs_json": json.dumps([p["published_mac_normalized"] for p in matches]),
                "candidate_pdf_pages_json": json.dumps(
                    sorted({p["publisher_pdf_page"] for p in matches})
                ),
                "source_doi": DOI,
                "source_table": "1",
                "status": "cross_release_candidate_not_approved"
                if matches
                else "unresolved_unmatched",
                "canonical_device_id": "",
            }
        )
    return candidates


def main() -> None:
    inventory = read_inventory()
    candidates = match_identifiers(inventory)
    write_csv(OUT / "table1_inventory.csv", inventory)
    write_csv(OUT / "table1_identifier_candidates.csv", candidates)
    counts = Counter(row["match_type"] for row in candidates)
    row_counts: Counter[str] = Counter()
    for row in candidates:
        row_counts[row["match_type"]] += int(row["local_row_count"])
    name_counts = Counter(normalize_name(row["published_name"]) for row in inventory)
    mac_counts = Counter(
        row["published_mac_normalized"] for row in inventory if row["published_mac_normalized"]
    )
    summary = {
        "source_doi": DOI,
        "source_xml_url": XML_URL,
        "retrieval_date": "2026-10-03",
        "inventory_entries": len(inventory),
        "inventory_mac_entries": sum(bool(p["published_mac_normalized"]) for p in inventory),
        "inventory_na_mac_entries": sum(not p["published_mac_normalized"] for p in inventory),
        "paper_narrative_device_count": 105,
        "inventory_count_discrepancy": "110 table entries versus narrative 105; unresolved",
        "local_identifiers": len(candidates),
        "local_packet_rows": sum(int(row["local_row_count"]) for row in candidates),
        "match_type_identifier_counts": dict(sorted(counts.items())),
        "match_type_local_row_counts": dict(sorted(row_counts.items())),
        "ambiguous_matched_identifiers": [
            r["identifier"] for r in candidates if int(r["candidate_entry_count"]) > 1
        ],
        "normalized_name_collisions": {k: v for k, v in name_counts.items() if v > 1},
        "mac_collisions": {k: v for k, v in mac_counts.items() if v > 1},
        "approved_identities": 0,
        "hashes_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                OUT / "ciciot2023.xml",
                OUT / "ciciot2023.pdf",
                LOCAL,
                Path(__file__).resolve(),
            )
        },
    }
    (OUT / "reconciliation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
