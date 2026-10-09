"""Extract exact identifier evidence; never approve physical-device identities."""

from __future__ import annotations

import argparse
import csv
import hashlib
import ipaddress
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import polars as pl

from iot_diad.metadata import SourceFile, discover_sources, strip_capture_suffix
from iot_diad.profiling import scan_strings
from iot_diad.schema import SchemaInfo, find_canonical_schemas, inspect_schema

PACKET_FIELDS = ("device_mac", "src_mac", "dst_mac", "src_ip", "dst_ip")
FLOW_FIELDS = ("Src IP", "Dst IP")
MAC = re.compile(r"^[0-9a-fA-F]{2}(?P<sep>[:-])(?:[0-9a-fA-F]{2}(?P=sep)){4}[0-9a-fA-F]{2}$")
PRIVATE = tuple(ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))
Row = dict[str, Any]


def normalize_identifier(value: str | None) -> str:
    value = (value or "").strip()
    return value.lower().replace("-", ":") if MAC.fullmatch(value) else value


def identifier_kind(value: str) -> str:
    if not value:
        return "missing"
    return "mac_form" if MAC.fullmatch(value) else "text_label"


def ip_scope(value: str) -> str:
    if not value:
        return "missing"
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return "invalid"
    if address.version == 4 and any(address in network for network in PRIVATE):
        return "rfc1918"
    return "other_ip"


def capture_key(source: SourceFile) -> tuple[str, str, str]:
    return source.coarse_label, source.fine_label, strip_capture_suffix(source.path.name)


def count_tuples(source: SourceFile, schema: SchemaInfo) -> list[Row]:
    fields = PACKET_FIELDS if source.feature_view == "packet" else FLOW_FIELDS
    return (
        scan_strings(source, schema)
        .select(fields)
        .group_by(fields)
        .agg(pl.len().alias("row_count"))
        .sort(fields, nulls_last=False)
        .collect(engine="streaming")
        .to_dicts()
    )


def packet_candidates(source: SourceFile, tuples: list[Row]) -> tuple[list[Row], list[Row]]:
    """Endpoint equality is observational evidence, never an alias/physical-identity join."""
    counts: Counter[tuple[str, str, str, str]] = Counter()
    identifiers: Counter[str] = Counter()
    orientations: dict[str, Counter[str]] = defaultdict(Counter)
    for row in tuples:
        device = normalize_identifier(row["device_mac"])
        identifiers[device] += row["row_count"]
        matches = [
            side
            for side in ("src", "dst")
            if device and device == normalize_identifier(row[f"{side}_mac"])
        ]
        orientation = "both" if len(matches) == 2 else matches[0] if matches else "neither"
        orientations[device][orientation] += row["row_count"]
        if not matches:
            counts[(device, "", "none", orientation)] += row["row_count"]
        for side in matches:
            counts[(device, (row[f"{side}_ip"] or "").strip(), side, orientation)] += row[
                "row_count"
            ]
    candidates = []
    for (device, ip, side, orientation), count in sorted(counts.items()):
        candidates.append(
            {
                "feature_view": "packet",
                "capture_id": source.capture_id,
                "relative_path": source.relative_path,
                "coarse_label": source.coarse_label,
                "fine_label": source.fine_label,
                "identifier": device,
                "identifier_kind": identifier_kind(device),
                "field": "device_mac",
                "endpoint_side": side,
                "ip": ip,
                "ip_scope": ip_scope(ip),
                "row_count": count,
                "packet_capture_id": source.capture_id,
                "packet_candidate_identifier": device,
                "alignment": "same_packet_row",
                "evidence": "identifier_endpoint_equality"
                if side != "none"
                else "no_endpoint_match",
                "orientation": orientation,
                "confidence": "direct_endpoint_observation" if side != "none" else "none",
                "status": "unresolved",
                "canonical_device_id": "",
            }
        )
    coverage = [
        {
            "capture_id": source.capture_id,
            "identifier": device,
            "identifier_kind": identifier_kind(device),
            "row_count": count,
            **{
                f"{side}_rows": orientations[device][side]
                for side in ("src", "dst", "both", "neither")
            },
        }
        for device, count in sorted(identifiers.items())
    ]
    return candidates, coverage


def flow_candidates(
    source: SourceFile,
    tuples: list[Row],
    packet_sources: list[SourceFile],
    packet_rows: list[Row],
) -> list[Row]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in tuples:
        for field in FLOW_FIELDS:
            counts[(field, (row[field] or "").strip())] += row["row_count"]
    aligned = [s for s in packet_sources if capture_key(s) == capture_key(source)]
    aligned_ids = {s.capture_id for s in aligned}
    by_ip: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in packet_rows:
        if (
            row["capture_id"] in aligned_ids
            and row["endpoint_side"] != "none"
            and row["ip_scope"] in ("rfc1918", "other_ip")
        ):
            by_ip[row["ip"]].add((row["capture_id"], row["identifier"]))
    result = []
    for (field, ip), count in sorted(counts.items()):
        links = sorted(by_ip.get(ip, set()))
        alignment = (
            "unique_filename_candidate"
            if len(aligned) == 1
            else ("ambiguous_filename_candidates" if aligned else "no_filename_candidate")
        )
        for packet_id, device in links or [("", "")]:
            result.append(
                {
                    "feature_view": "flow",
                    "capture_id": source.capture_id,
                    "relative_path": source.relative_path,
                    "coarse_label": source.coarse_label,
                    "fine_label": source.fine_label,
                    "identifier": ip,
                    "identifier_kind": "ip_endpoint",
                    "field": field,
                    "endpoint_side": "src" if field == "Src IP" else "dst",
                    "ip": ip,
                    "ip_scope": ip_scope(ip),
                    "row_count": count,
                    "packet_capture_id": packet_id,
                    "packet_candidate_identifier": device,
                    "alignment": alignment,
                    "evidence": "filename_and_endpoint_ip" if device else "flow_endpoint_only",
                    "orientation": "not_applicable",
                    "confidence": "tentative" if device else "none",
                    "status": "unresolved",
                    "canonical_device_id": "",
                }
            )
    return result


def write_csv(path: Path, rows: list[Row]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(raw_dir: Path, inventory_path: Path, audit_identifiers: Path, output: Path) -> Row:
    sources = discover_sources(raw_dir)
    canonical = find_canonical_schemas(sources)
    with inventory_path.open(newline="") as handle:
        inventory = {row["relative_path"]: row for row in csv.DictReader(handle)}
    if set(inventory) != {s.relative_path for s in sources}:
        raise ValueError("Raw paths differ from audit inventory")
    output.mkdir(parents=True, exist_ok=True)
    packet_rows: list[Row] = []
    packet_coverage: list[Row] = []
    flow_data: list[tuple[SourceFile, list[Row]]] = []
    manifest: list[Row] = []
    audit_counts: Counter[tuple[str, str, str, str]] = Counter()
    audit_captures: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    all_evidence: dict[str, list[Row]] = {"packet": [], "flow": []}
    for index, source in enumerate(sources, 1):
        before = source.path.stat()
        schema = inspect_schema(source, canonical[source.feature_view])
        if schema.status not in ("canonical", "headerless"):
            raise ValueError(f"Unexpected schema: {source.relative_path}")
        rows = count_tuples(source, schema)
        after = source.path.stat()
        expected = inventory[source.relative_path]
        total = sum(row["row_count"] for row in rows)
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or total != int(expected["row_count"])
            or after.st_size != int(expected["size_bytes"])
            or schema.schema_hash != expected["schema_hash"]
        ):
            raise ValueError(f"Raw data mismatch/change: {source.relative_path}")
        digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode())
        manifest.append(
            {
                "relative_path": source.relative_path,
                "capture_id": source.capture_id,
                "feature_view": source.feature_view,
                "size_bytes": after.st_size,
                "mtime_ns": after.st_mtime_ns,
                "schema_status": schema.status,
                "schema_hash": schema.schema_hash,
                "row_count": total,
                "expected_row_count": int(expected["row_count"]),
                "tuple_count": len(rows),
                "projected_tuple_counts_sha256": digest.hexdigest(),
            }
        )
        fields = ("device_mac",) if source.feature_view == "packet" else FLOW_FIELDS
        for row in rows:
            all_evidence[source.feature_view].append({"capture_id": source.capture_id, **row})
            for field in fields:
                value = (row[field] or "").strip()
                if value and (source.feature_view == "packet" or ip_scope(value) == "rfc1918"):
                    key = (source.feature_view, source.coarse_label, field, value)
                    audit_counts[key] += row["row_count"]
                    audit_captures[key].add(source.capture_id)
        if source.feature_view == "packet":
            candidates, coverage = packet_candidates(source, rows)
            packet_rows.extend(candidates)
            packet_coverage.extend(coverage)
        else:
            flow_data.append((source, rows))
        print(
            f"{index}/{len(sources)} {source.feature_view} {total:,} rows {source.path.name}",
            flush=True,
        )
    with audit_identifiers.open(newline="") as handle:
        expected_identifiers = list(csv.DictReader(handle))
    expected_counts = {
        (r["feature_view"], r["coarse_label"], r["field"], r["value"]): (
            int(r["row_count"]),
            int(r["capture_count"]),
        )
        for r in expected_identifiers
    }
    actual_counts = {k: (v, len(audit_captures[k])) for k, v in audit_counts.items()}
    if expected_counts != actual_counts:
        raise ValueError("Extracted class identifier counts differ from exact audit")
    packet_sources = [s for s in sources if s.feature_view == "packet"]
    flow_rows = [
        row
        for source, tuples in flow_data
        for row in flow_candidates(source, tuples, packet_sources, packet_rows)
    ]
    write_csv(output / "candidate_mapping.csv", packet_rows + flow_rows)
    write_csv(output / "packet_identifier_coverage.csv", packet_coverage)
    write_csv(output / "source_manifest.csv", manifest)
    write_csv(output / "packet_endpoint_evidence.csv", all_evidence["packet"])
    write_csv(output / "flow_endpoint_evidence.csv", all_evidence["flow"])
    identifier_summary: list[Row] = []
    for device in sorted({r["identifier"] for r in packet_coverage}):
        coverage = [r for r in packet_coverage if r["identifier"] == device]
        links = [r for r in packet_rows if r["identifier"] == device]
        ips = sorted({r["ip"] for r in links if r["ip_scope"] == "rfc1918"})
        identifier_summary.append(
            {
                "identifier": device,
                "identifier_kind": identifier_kind(device),
                "row_count": sum(r["row_count"] for r in coverage),
                "capture_count": len(coverage),
                **{
                    f"{side}_rows": sum(r[f"{side}_rows"] for r in coverage)
                    for side in ("src", "dst", "both", "neither")
                },
                "rfc1918_ip_count": len(ips),
                "rfc1918_ips_json": json.dumps(ips),
                "status": "unresolved",
                "canonical_device_id": "",
            }
        )
    write_csv(output / "packet_identifier_summary.csv", identifier_summary)
    ip_identifiers: dict[str, set[str]] = defaultdict(set)
    for row in packet_rows:
        if row["ip_scope"] == "rfc1918":
            ip_identifiers[row["ip"]].add(row["identifier"])
    summary = {
        "evidence_scope": "exact_full_data_projected_identifiers",
        "physical_identities_approved": 0,
        "raw_directory": str(raw_dir),
        "file_count": len(sources),
        "row_count": sum(r["row_count"] for r in manifest),
        "rows_by_view": {
            v: sum(r["row_count"] for r in manifest if r["feature_view"] == v)
            for v in ("packet", "flow")
        },
        "headerless_files": sum(r["schema_status"] == "headerless" for r in manifest),
        "audit_identifier_reconciliation": "exact_match_rows_and_capture_counts",
        "packet_identifier_count": len(identifier_summary),
        "packet_orientation_rows": {
            side: sum(r[f"{side}_rows"] for r in packet_coverage)
            for side in ("src", "dst", "both", "neither")
        },
        "packet_rfc1918_ips": len(ip_identifiers),
        "packet_rfc1918_ips_with_multiple_identifiers": sum(
            len(v) > 1 for v in ip_identifiers.values()
        ),
        "flow_rfc1918_ips": len({r["ip"] for r in flow_rows if r["ip_scope"] == "rfc1918"}),
        "candidate_row_count": len(packet_rows) + len(flow_rows),
        "inventory_sha256": file_hash(inventory_path),
        "audit_identifiers_sha256": file_hash(audit_identifiers),
        "extractor_sha256": file_hash(Path(__file__)),
        "polars_version": pl.__version__,
        "python_version": sys.version,
        "polars_threads": pl.thread_pool_size(),
        "input_fingerprint_scope": (
            "path/size/mtime/schema and ordered grouped projected values; not full raw SHA256"
        ),
        "normalization": (
            "strip whitespace; case/separator normalization only for valid MAC-form values; "
            "preserve label case"
        ),
        "flow_link_rule": (
            "same coarse/fine label and exact capture basename after removing .pcap_Flow; "
            "same endpoint IP; tentative only"
        ),
        "row_count_semantics": (
            "Flow candidate counts repeat for multiple candidate identities; "
            "packet both orientation counts repeat per side; "
            "use evidence and coverage tables for totals"
        ),
        "artifact_sha256": {p.name: file_hash(p) for p in sorted(output.glob("*.csv"))},
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/CIC-IoT-IDAD-Dataset-2024"))
    parser.add_argument(
        "--inventory", type=Path, default=Path("artifacts/data_audit/file_inventory.csv")
    )
    parser.add_argument(
        "--audit-identifiers", type=Path, default=Path("artifacts/data_audit/identifier_counts.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/device_mapping"))
    args = parser.parse_args()
    summary = run(args.raw_dir, args.inventory, args.audit_identifiers, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
