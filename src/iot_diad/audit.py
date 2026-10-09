from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import tomllib
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from iot_diad.metadata import FeatureView, SourceFile, discover_sources
from iot_diad.schema import SchemaInfo, find_canonical_schemas, inspect_schema


@dataclass(frozen=True, slots=True)
class AuditConfig:
    raw_dir: Path
    output_dir: Path
    report_path: Path
    workers: int
    private_ipv4_pattern: str


@dataclass(frozen=True, slots=True)
class FileAudit:
    relative_path: str
    feature_view: FeatureView
    coarse_label: str
    fine_label: str
    capture_id: str
    size_bytes: int
    row_count: int
    column_count: int
    has_header: bool
    schema_status: str
    schema_hash: str


def load_config(path: Path) -> AuditConfig:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    workers = int(raw["audit"].get("workers", min(4, os.cpu_count() or 1)))
    if workers < 1:
        raise ValueError("audit.workers must be positive")
    return AuditConfig(
        raw_dir=Path(raw["paths"]["raw_dir"]),
        output_dir=Path(raw["paths"]["output_dir"]),
        report_path=Path(raw["paths"]["report_path"]),
        workers=workers,
        private_ipv4_pattern=str(raw["audit"]["private_ipv4_pattern"]),
    )


def count_physical_lines(path: Path, chunk_size: int = 16 * 1024 * 1024) -> int:
    line_count = 0
    last_byte = b""
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            line_count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if last_byte and last_byte != b"\n":
        line_count += 1
    return line_count


def audit_file(source: SourceFile, schema: SchemaInfo) -> FileAudit:
    physical_lines = count_physical_lines(source.path)
    row_count = physical_lines - int(schema.has_header)
    if row_count < 0:
        raise ValueError(f"Invalid row count for {source.relative_path}")
    return FileAudit(
        relative_path=source.relative_path,
        feature_view=source.feature_view,
        coarse_label=source.coarse_label,
        fine_label=source.fine_label,
        capture_id=source.capture_id,
        size_bytes=source.size_bytes,
        row_count=row_count,
        column_count=schema.column_count,
        has_header=schema.has_header,
        schema_status=schema.status,
        schema_hash=schema.schema_hash,
    )


def run_inventory(
    sources: list[SourceFile], schemas: dict[str, SchemaInfo], workers: int
) -> list[FileAudit]:
    results: list[FileAudit] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(audit_file, source, schemas[source.relative_path]): source
            for source in sources
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            elapsed = max(time.monotonic() - started, 1e-9)
            rate = completed / elapsed
            remaining = (len(sources) - completed) / rate if rate else 0.0
            message = (
                f"Inventory {completed:>3}/{len(sources)} | "
                f"{result.row_count:,} rows | ETA {format_duration(remaining)}"
            )
            sys.stdout.write("\r\033[K" + message)
            sys.stdout.flush()
    sys.stdout.write("\n")
    return sorted(results, key=lambda item: item.relative_path)


def profile_identifiers(
    sources: list[SourceFile],
    schemas: dict[str, SchemaInfo],
    private_ipv4_pattern: str,
) -> list[dict[str, Any]]:
    import polars as pl

    totals: dict[tuple[str, str, str, str], list[Any]] = {}
    targets: dict[FeatureView, tuple[str, ...]] = {
        "packet": ("device_mac",),
        "flow": ("Src IP", "Dst IP"),
    }

    for index, source in enumerate(sources, start=1):
        schema = schemas[source.relative_path]
        fields = targets[source.feature_view]
        lazy = scan_identifier_fields(source, schema, fields, pl)
        if source.feature_view == "flow":
            lazy = lazy.filter(pl.col("value").str.contains(private_ipv4_pattern))
        counts = (
            lazy.filter(pl.col("value").is_not_null() & (pl.col("value") != ""))
            .group_by(["field", "value"])
            .agg(pl.len().alias("rows"))
            .collect(engine="streaming")
        )
        for field, value, rows in counts.iter_rows():
            key = (source.feature_view, source.coarse_label, field, str(value))
            entry = totals.setdefault(key, [0, set()])
            entry[0] += int(rows)
            entry[1].add(source.capture_id)
        sys.stdout.write(f"\r\033[KIdentifiers {index:>3}/{len(sources)} | {source.relative_path}")
        sys.stdout.flush()
    sys.stdout.write("\n")

    rows = []
    for (view, label, field, value), (row_count, captures) in sorted(totals.items()):
        rows.append(
            {
                "feature_view": view,
                "coarse_label": label,
                "field": field,
                "value": value,
                "row_count": row_count,
                "capture_count": len(captures),
            }
        )
    return rows


def scan_identifier_fields(
    source: SourceFile,
    schema: SchemaInfo,
    fields: tuple[str, ...],
    pl: Any,
) -> Any:
    options: dict[str, Any] = {
        "has_header": schema.has_header,
        "schema_overrides": {field: pl.String for field in fields},
        "infer_schema_length": 100,
        "low_memory": True,
    }
    if not schema.has_header:
        options["new_columns"] = list(schema.columns)
    return (
        pl.scan_csv(source.path, **options)
        .select([pl.col(field).cast(pl.String) for field in fields])
        .unpivot(variable_name="field", value_name="value")
        .with_columns(pl.col("value").str.strip_chars())
    )


def aggregate_classes(inventory: Sequence[FileAudit]) -> list[dict[str, Any]]:
    aggregates: dict[tuple[str, str], dict[str, Any]] = {}
    for item in inventory:
        key = (item.feature_view, item.coarse_label)
        row = aggregates.setdefault(
            key,
            {
                "feature_view": item.feature_view,
                "coarse_label": item.coarse_label,
                "file_count": 0,
                "row_count": 0,
                "size_bytes": 0,
            },
        )
        row["file_count"] += 1
        row["row_count"] += item.row_count
        row["size_bytes"] += item.size_bytes
    return [aggregates[key] for key in sorted(aggregates)]


def aggregate_schemas(inventory: Sequence[FileAudit]) -> list[dict[str, Any]]:
    aggregates: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for item in inventory:
        key = (item.feature_view, item.schema_status, item.schema_hash, item.column_count)
        row = aggregates.setdefault(
            key,
            {
                "feature_view": item.feature_view,
                "schema_status": item.schema_status,
                "schema_hash": item.schema_hash,
                "column_count": item.column_count,
                "file_count": 0,
                "row_count": 0,
                "example_path": item.relative_path,
            },
        )
        row["file_count"] += 1
        row["row_count"] += item.row_count
    return [aggregates[key] for key in sorted(aggregates)]


def write_outputs(
    config: AuditConfig,
    inventory: list[FileAudit],
    identifiers: list[dict[str, Any]] | None,
    discovered_file_count: int,
) -> None:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_rows = [asdict(item) for item in inventory]
    class_rows = aggregate_classes(inventory)
    schema_rows = aggregate_schemas(inventory)

    write_csv(output_dir / "file_inventory.csv", inventory_rows)
    write_csv(output_dir / "class_summary.csv", class_rows)
    write_csv(output_dir / "schema_summary.csv", schema_rows)
    if identifiers is not None:
        write_csv(output_dir / "identifier_counts.csv", identifiers)
    else:
        (output_dir / "identifier_counts.csv").unlink(missing_ok=True)

    summary = {
        "raw_directory": config.raw_dir.as_posix(),
        "file_count": len(inventory),
        "discovered_file_count": discovered_file_count,
        "complete_inventory": len(inventory) == discovered_file_count,
        "row_count": sum(item.row_count for item in inventory),
        "size_bytes": sum(item.size_bytes for item in inventory),
        "headerless_file_count": sum(item.schema_status == "headerless" for item in inventory),
        "class_count": len({item.coarse_label for item in inventory}),
        "feature_views": sorted({item.feature_view for item in inventory}),
        "identifier_profiled": identifiers is not None,
    }
    write_json(output_dir / "audit_summary.json", summary)
    report = render_report(summary, class_rows, schema_rows, identifiers)
    write_text(output_dir / "data_summary.md", report)
    write_text(config.report_path, report)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def render_report(
    summary: dict[str, Any],
    classes: Sequence[dict[str, Any]],
    schemas: Sequence[dict[str, Any]],
    identifiers: Sequence[dict[str, Any]] | None,
) -> str:
    lines = [
        "# CIC IoT-DIAD 2024 Data Summary",
        "",
        "## Inventory",
        "",
        f"- CSV files audited: {summary['file_count']:,} of {summary['discovered_file_count']:,}",
        f"- Complete inventory: {'yes' if summary['complete_inventory'] else 'no'}",
        f"- Rows: {summary['row_count']:,}",
        f"- Size: {format_bytes(summary['size_bytes'])}",
        f"- Headerless files: {summary['headerless_file_count']:,}",
        "",
        "## Class Distribution",
        "",
        "| Feature view | Class | Files | Rows | Size |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in classes:
        lines.append(
            f"| {row['feature_view']} | {row['coarse_label']} | "
            f"{row['file_count']:,} | {row['row_count']:,} | "
            f"{format_bytes(row['size_bytes'])} |"
        )

    lines.extend(
        [
            "",
            "## Schema Audit",
            "",
            "| Feature view | Status | Columns | Files | Rows |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in schemas:
        lines.append(
            f"| {row['feature_view']} | {row['schema_status']} | "
            f"{row['column_count']} | {row['file_count']:,} | {row['row_count']:,} |"
        )

    lines.extend(["", "## Device Identity", ""])
    if identifiers is None:
        lines.append("Identifier profiling was not run.")
    else:
        packet_devices = {
            str(row["value"])
            for row in identifiers
            if row["feature_view"] == "packet" and row["field"] == "device_mac"
        }
        mac_address = re.compile(r"^(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")
        mac_values = {value for value in packet_devices if mac_address.fullmatch(value)}
        named_values = packet_devices - mac_values
        flow_ips = {
            str(row["value"])
            for row in identifiers
            if row["feature_view"] == "flow" and row["field"] in {"Src IP", "Dst IP"}
        }
        lines.extend(
            [
                f"- Distinct packet `device_mac` values: {len(packet_devices):,}",
                f"  - MAC-form values: {len(mac_values):,}",
                f"  - Device-name or other textual values: {len(named_values):,}",
                f"- Distinct private IP values in flow data: {len(flow_ips):,}",
                "- The mixed representation in `device_mac` does not support treating every value "
                "as a physical-device identifier without additional validation.",
                "- A physical-device split also requires mapping flow IPs to the same validated "
                "device units.",
            ]
        )

    lines.extend(
        [
            "",
            "## Label Contract",
            "",
            "Source rows are not reliably labeled. Coarse and fine labels are derived from the "
            "directory hierarchy and capture filename. Source identifier columns remain available "
            "for grouping and leakage audits, but must not be model features.",
            "",
            "## Counting Method",
            "",
            "Row totals use bounded-memory physical line counts after detecting whether each file "
            "has a header. Identifier counts are parsed from projected columns with Polars.",
            "",
        ]
    )
    return "\n".join(lines)


def format_bytes(value: int | float) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def select_sources(sources: list[SourceFile], limit: int | None) -> list[SourceFile]:
    if limit is None:
        return sources
    if limit < 1:
        raise ValueError("--limit-files must be positive")
    by_view = {
        view: sorted(
            (source for source in sources if source.feature_view == view),
            key=lambda source: (source.size_bytes, source.relative_path),
        )
        for view in ("flow", "packet")
    }
    selected: list[SourceFile] = []
    index = 0
    while len(selected) < limit:
        added = False
        for view in ("flow", "packet"):
            if index < len(by_view[view]) and len(selected) < limit:
                selected.append(by_view[view][index])
                added = True
        if not added:
            break
        index += 1
    return sorted(selected, key=lambda source: source.relative_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit CIC IoT-DIAD 2024 feature CSV files.")
    parser.add_argument("--config", type=Path, default=Path("config/audit.toml"))
    parser.add_argument(
        "--stage",
        choices=("inventory", "all"),
        default="all",
        help="inventory counts rows and schemas; all also profiles device identifiers",
    )
    parser.add_argument("--limit-files", type=int, help="process a deterministic subset")
    parser.add_argument(
        "--output-dir", type=Path, help="override the configured artifact directory"
    )
    parser.add_argument("--report-path", type=Path, help="override the configured report path")
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    if args.report_path is not None:
        config = replace(config, report_path=args.report_path)
    all_sources = discover_sources(config.raw_dir)
    sources = select_sources(all_sources, args.limit_files)
    canonical = find_canonical_schemas(all_sources)
    schemas = {
        source.relative_path: inspect_schema(source, canonical[source.feature_view])
        for source in sources
    }
    inventory = run_inventory(sources, schemas, config.workers)
    identifiers = None
    if args.stage == "all":
        identifiers = profile_identifiers(sources, schemas, config.private_ipv4_pattern)
    write_outputs(config, inventory, identifiers, len(all_sources))
    print(f"Audit artifacts written to {config.output_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        run(parser.parse_args(argv))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
