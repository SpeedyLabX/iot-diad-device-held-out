"""Independently reconcile MAP-001 artifacts and selected raw CSV records."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/device_mapping"
RAW = ROOT / "data/raw/CIC-IoT-IDAD-Dataset-2024"
MAC = re.compile(r"[0-9a-fA-F]{2}([:-])(?:[0-9a-fA-F]{2}\1){4}[0-9a-fA-F]{2}")
PRIVATE = re.compile(r"^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.)")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(value: str) -> str:
    value = value.strip()
    return value.lower().replace("-", ":") if MAC.fullmatch(value) else value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    inventory_path = ROOT / "artifacts/data_audit/file_inventory.csv"
    identifiers_path = ROOT / "artifacts/data_audit/identifier_counts.csv"
    inventory = read(inventory_path)
    by_capture = {r["capture_id"]: r for r in inventory}
    manifest = read(OUTPUT / "source_manifest.csv")
    summary = json.loads((OUTPUT / "run_summary.json").read_text())
    require(
        summary["extractor_sha256"] == digest(ROOT / "src/iot_diad/device_mapping.py"),
        "Extractor code differs from recorded run",
    )
    require(summary["inventory_sha256"] == digest(inventory_path), "Inventory hash differs")
    require(summary["audit_identifiers_sha256"] == digest(identifiers_path), "Audit hash differs")
    for filename, expected_hash in summary["artifact_sha256"].items():
        require(digest(OUTPUT / filename) == expected_hash, f"Artifact hash differs: {filename}")
    require(len(manifest) == len(inventory), "Manifest file count differs")
    require({r["capture_id"] for r in manifest} == set(by_capture), "Manifest captures differ")
    for item in manifest:
        expected = by_capture[item["capture_id"]]
        for field in ("relative_path", "feature_view", "size_bytes", "row_count", "schema_hash"):
            require(item[field] == expected[field], f"Manifest mismatch: {field}")
        stat = (RAW / item["relative_path"]).stat()
        require(stat.st_size == int(item["size_bytes"]), "Raw file size changed")
        require(stat.st_mtime_ns == int(item["mtime_ns"]), "Raw file modification time changed")

    evidence = {v: read(OUTPUT / f"{v}_endpoint_evidence.csv") for v in ("packet", "flow")}
    capture_totals: Counter[str] = Counter()
    audit_totals: Counter[tuple[str, str, str, str]] = Counter()
    audit_captures: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for view, rows in evidence.items():
        fields: tuple[str, ...] = ("device_mac",) if view == "packet" else ("Src IP", "Dst IP")
        for row in rows:
            capture = row["capture_id"]
            count = int(row["row_count"])
            require(count > 0, "Nonpositive evidence count")
            capture_totals[capture] += count
            for field in fields:
                value = row[field].strip()
                if value and (view == "packet" or PRIVATE.match(value)):
                    key = (view, by_capture[capture]["coarse_label"], field, value)
                    audit_totals[key] += count
                    audit_captures[key].add(capture)
    require(
        capture_totals == {k: int(v["row_count"]) for k, v in by_capture.items()},
        "Evidence per-capture row totals differ",
    )
    expected_audit = {
        (r["feature_view"], r["coarse_label"], r["field"], r["value"]): (
            int(r["row_count"]),
            int(r["capture_count"]),
        )
        for r in read(identifiers_path)
    }
    require(
        {k: (v, len(audit_captures[k])) for k, v in audit_totals.items()} == expected_audit,
        "Identifier row/capture counts differ from prior audit",
    )

    selected = sorted(
        (r for r in inventory if r["feature_view"] == "packet"), key=lambda r: int(r["size_bytes"])
    )[:2]
    selected += [r for r in inventory if r["has_header"] == "False"]
    header_source = next(
        r for r in inventory if r["feature_view"] == "flow" and r["has_header"] == "True"
    )
    with (RAW / header_source["relative_path"]).open(newline="") as handle:
        flow_header = next(csv.reader(handle))
    raw_checks: list[dict[str, Any]] = []
    for item in selected:
        view = item["feature_view"]
        fields = (
            ("device_mac", "src_mac", "dst_mac", "src_ip", "dst_ip")
            if view == "packet"
            else ("Src IP", "Dst IP")
        )
        path = RAW / item["relative_path"]
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(
                handle, fieldnames=flow_header if item["has_header"] == "False" else None
            )
            actual = Counter(tuple(r[f] for f in fields) for r in reader)
        exported: Counter[tuple[str, ...]] = Counter()
        for row in evidence[view]:
            if row["capture_id"] == item["capture_id"]:
                exported[tuple(row[f] for f in fields)] += int(row["row_count"])
        require(actual == exported, f"Raw tuple mismatch: {path}")
        raw_checks.append(
            {
                "relative_path": item["relative_path"],
                "rows": sum(actual.values()),
                "tuples": len(actual),
                "raw_sha256": digest(path),
            }
        )

    packet_links: set[tuple[str, str, str]] = set()
    for row in evidence["packet"]:
        device = normalized(row["device_mac"])
        for side in ("src", "dst"):
            if device and device == normalized(row[f"{side}_mac"]):
                packet_links.add((row["capture_id"], device, row[f"{side}_ip"].strip()))
    candidates = read(OUTPUT / "candidate_mapping.csv")
    for row in candidates:
        require(
            row["status"] == "unresolved" and not row["canonical_device_id"],
            "Candidate incorrectly accepts an identity",
        )
        if row["feature_view"] != "flow" or not row["packet_candidate_identifier"]:
            continue
        link = (row["packet_capture_id"], row["packet_candidate_identifier"], row["ip"])
        require(link in packet_links, "Flow link lacks packet endpoint equality")
        flow, packet = by_capture[row["capture_id"]], by_capture[row["packet_capture_id"]]
        for field in ("coarse_label", "fine_label"):
            require(flow[field] == packet[field], "Flow link crosses label groups")
        require(
            Path(flow["relative_path"]).stem.removesuffix(".pcap_Flow")
            == Path(packet["relative_path"]).stem,
            "Flow link lacks filename candidate",
        )
        require(row["confidence"] == "tentative", "Flow link overstates confidence")
    result = {
        "status": "passed",
        "reviewer": "coordinator (independent of map-worker)",
        "verification_script_sha256": digest(Path(__file__)),
        "manifest_files": len(manifest),
        "exact_evidence_rows": sum(capture_totals.values()),
        "audit_identifier_groups_reconciled": len(expected_audit),
        "candidate_rows_checked": len(candidates),
        "raw_capture_checks": raw_checks,
        "limits": "Full artifact reconciliation; independent raw parsing covers seven captures "
        "only. "
        "No physical identities, capture alignment, or exclusions approved.",
    }
    (OUTPUT / "coordinator_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
