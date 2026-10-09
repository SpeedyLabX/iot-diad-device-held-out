"""Materialize the project lead's approved Gate 1 packet mapping disposition."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT / "artifacts/device_mapping"
LITERATURE = MAPPING / "literature"
BASE = LITERATURE / "table1_identifier_candidates.csv"
ALIASES = LITERATURE / "residual_alias_candidates.csv"
OUTPUT = MAPPING / "approved_packet_device_mapping.csv"
SUMMARY = MAPPING / "gate1_decision_summary.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def single_json_value(row: dict[str, str], field: str) -> str:
    values = json.loads(row[field])
    if len(values) != 1:
        raise ValueError(f"Expected one {field} value for {row['identifier']}: {values}")
    return str(values[0])


def main() -> None:
    base_rows = read_csv(BASE)
    aliases = {row["identifier"]: row for row in read_csv(ALIASES)}
    decisions: list[dict[str, str]] = []

    for base in base_rows:
        identifier = base["identifier"]
        candidate = aliases.get(identifier, base)
        candidate_count = int(candidate["candidate_entry_count"])
        if candidate_count == 1:
            disposition = "approved_primary_packet_mapping"
            canonical_device_id = single_json_value(candidate, "candidate_entry_ids_json")
            published_name = single_json_value(candidate, "candidate_names_json")
            published_mac = single_json_value(candidate, "candidate_macs_json")
            evidence_rule = candidate.get("alias_rule") or base["match_type"]
            rationale = (
                "Project lead approved the unique CICIoT2023 Table 1-backed candidate "
                "for the packet-primary physical-device evaluation."
            )
        elif candidate_count > 1:
            disposition = "excluded_primary_ambiguous_identity"
            canonical_device_id = ""
            published_name = ""
            published_mac = ""
            evidence_rule = base["match_type"]
            rationale = (
                f"Excluded from the primary evaluation because {candidate_count} published "
                "physical entries satisfy the evidence rule."
            )
        else:
            disposition = "excluded_primary_unresolved_identity"
            canonical_device_id = ""
            published_name = ""
            published_mac = ""
            evidence_rule = base["match_type"]
            rationale = (
                "Excluded from the primary evaluation because no unique published physical "
                "identity is supported."
            )

        decisions.append(
            {
                "identifier": identifier,
                "identifier_kind": base["identifier_kind"],
                "row_count": base["local_row_count"],
                "capture_count": base["local_capture_count"],
                "disposition": disposition,
                "canonical_device_id": canonical_device_id,
                "published_name": published_name,
                "published_mac": published_mac,
                "evidence_rule": evidence_rule,
                "source_doi": candidate["source_doi"],
                "rationale": rationale,
            }
        )

    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decisions[0]))
        writer.writeheader()
        writer.writerows(decisions)

    approved = [row for row in decisions if row["disposition"] == "approved_primary_packet_mapping"]
    ambiguous = [
        row for row in decisions if row["disposition"] == "excluded_primary_ambiguous_identity"
    ]
    unresolved = [
        row for row in decisions if row["disposition"] == "excluded_primary_unresolved_identity"
    ]
    summary = {
        "decision_date": "2026-10-05",
        "decision_authority": "project lead",
        "primary_evaluation": "packet physical-device-held-out",
        "flow_evaluation": (
            "no unseen-physical-device claim; separately framed analysis only until an "
            "authoritative Flow-IP-to-device correspondence is verified"
        ),
        "source_candidate_sha256": sha256(BASE),
        "source_alias_sha256": sha256(ALIASES),
        "total_identifiers": len(decisions),
        "total_rows": sum(int(row["row_count"]) for row in decisions),
        "approved_identifiers": len(approved),
        "approved_rows": sum(int(row["row_count"]) for row in approved),
        "ambiguous_excluded_identifiers": len(ambiguous),
        "ambiguous_excluded_rows": sum(int(row["row_count"]) for row in ambiguous),
        "unresolved_excluded_identifiers": len(unresolved),
        "unresolved_excluded_rows": sum(int(row["row_count"]) for row in unresolved),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
