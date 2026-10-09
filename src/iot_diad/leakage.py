"""Split-aware leakage diagnostics for the packet-primary evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
import tomllib
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl

from iot_diad.metadata import SourceFile, discover_sources
from iot_diad.roles import candidate_feature_columns
from iot_diad.schema import find_canonical_schemas, inspect_schema

PARTITION_IDS = {"train": 0, "calibration": 1, "test": 2}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw_dir(root: Path) -> Path:
    raw = tomllib.loads((root / "config/audit.toml").read_text(encoding="utf-8"))
    return root / str(raw["paths"]["raw_dir"])


def capture_overlap(
    coverage: list[dict[str, str]], identifier_partitions: dict[str, str]
) -> tuple[list[dict[str, str]], dict[str, set[str]]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    graph: dict[str, set[str]] = defaultdict(set)
    for row in coverage:
        identifier = row["identifier"]
        if identifier not in identifier_partitions:
            continue
        capture = row["capture_id"]
        partition = identifier_partitions[identifier]
        counts[capture][partition] += int(row["row_count"])
        device_node = f"device:{identifier}"
        capture_node = f"capture:{capture}"
        graph[device_node].add(capture_node)
        graph[capture_node].add(device_node)
    rows = []
    for capture in sorted(counts):
        values = counts[capture]
        partitions = sorted(values)
        rows.append(
            {
                "capture_id": capture,
                "partition_count": str(len(partitions)),
                "partitions": ";".join(partitions),
                "train_rows": str(values.get("train", 0)),
                "calibration_rows": str(values.get("calibration", 0)),
                "test_rows": str(values.get("test", 0)),
                "eligible_rows": str(sum(values.values())),
            }
        )
    return rows, graph


def connected_components(graph: dict[str, set[str]]) -> list[dict[str, int]]:
    seen: set[str] = set()
    components: list[dict[str, int]] = []
    for start in sorted(graph):
        if start in seen:
            continue
        pending = [start]
        seen.add(start)
        devices = 0
        captures = 0
        while pending:
            node = pending.pop()
            devices += node.startswith("device:")
            captures += node.startswith("capture:")
            for neighbor in graph[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
        components.append({"device_count": devices, "capture_count": captures})
    return components


def scan_packet_hashes(
    source: SourceFile,
    columns: tuple[str, ...],
    features: list[str],
    identifier_partition_ids: dict[str, int],
    output_path: Path,
) -> None:
    schema = {column: pl.String for column in columns}
    partition = pl.col("device_mac").replace_strict(identifier_partition_ids).cast(pl.UInt8)
    frame = (
        pl.scan_csv(source.path, has_header=True, schema=schema, low_memory=True)
        .filter(pl.col("device_mac").is_in(identifier_partition_ids))
        .select(
            partition.alias("partition_id"),
            pl.lit(source.coarse_label).alias("coarse_label"),
            pl.struct(features)
            .hash(seed=2027, seed_1=2029, seed_2=2039, seed_3=2053)
            .alias("hash_a"),
            pl.struct(features)
            .hash(seed=2081, seed_1=2083, seed_2=2087, seed_3=2089)
            .alias("hash_b"),
        )
    )
    frame.sink_parquet(output_path, compression="zstd", mkdir=True)


def build_feature_hash_audit(
    root: Path,
    packet_sources: list[SourceFile],
    columns: tuple[str, ...],
    identifier_partition_ids: dict[str, int],
    output: Path,
    keep_shards: bool,
) -> dict[str, Any]:
    features = candidate_feature_columns("packet", columns)
    shard_dir = output / "hash_shards"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True)
    started = time.monotonic()
    for index, source in enumerate(packet_sources, start=1):
        digest = hashlib.sha256(source.relative_path.encode()).hexdigest()[:16]
        scan_packet_hashes(
            source,
            columns,
            features,
            identifier_partition_ids,
            shard_dir / f"{digest}.parquet",
        )
        elapsed = max(time.monotonic() - started, 1e-9)
        eta = (len(packet_sources) - index) / (index / elapsed)
        print(
            f"Hashed {index}/{len(packet_sources)} packet captures; ETA {eta / 60:.1f} min",
            flush=True,
        )

    cross_path = output / "cross_partition_feature_hashes.parquet"
    hashes = pl.scan_parquet(shard_dir / "*.parquet")
    grouped = hashes.group_by("hash_a", "hash_b").agg(
        (pl.col("partition_id") == 0).sum().alias("train_rows"),
        (pl.col("partition_id") == 1).sum().alias("calibration_rows"),
        (pl.col("partition_id") == 2).sum().alias("test_rows"),
        pl.col("coarse_label").n_unique().alias("coarse_label_count"),
    )
    cross = grouped.filter(
        (pl.col("train_rows") > 0).cast(pl.UInt8)
        + (pl.col("calibration_rows") > 0).cast(pl.UInt8)
        + (pl.col("test_rows") > 0).cast(pl.UInt8)
        > 1
    ).with_columns(
        (
            (pl.col("train_rows") > 0).cast(pl.UInt8)
            + 2 * (pl.col("calibration_rows") > 0).cast(pl.UInt8)
            + 4 * (pl.col("test_rows") > 0).cast(pl.UInt8)
        ).alias("partition_mask")
    )
    cross.sink_parquet(cross_path, compression="zstd", mkdir=True)
    summary_frame = (
        pl.scan_parquet(cross_path)
        .group_by("partition_mask")
        .agg(
            pl.len().alias("hash_group_count"),
            pl.col("train_rows").sum(),
            pl.col("calibration_rows").sum(),
            pl.col("test_rows").sum(),
            (pl.col("coarse_label_count") > 1).sum().alias("cross_label_hash_group_count"),
        )
        .sort("partition_mask")
        .collect(engine="streaming")
    )
    eligible_rows = int(
        pl.scan_parquet(shard_dir / "*.parquet").select(pl.len()).collect(engine="streaming").item()
    )
    result = {
        "hash_method": "two independent Polars UInt64 hashes over raw candidate-feature strings",
        "candidate_feature_count": len(features),
        "eligible_rows_hashed": eligible_rows,
        "cross_partition_hash_group_count": int(summary_frame["hash_group_count"].sum()),
        "cross_partition_rows_by_mask": summary_frame.to_dicts(),
        "cross_partition_hashes_sha256": sha256(cross_path),
    }
    if not keep_shards:
        shutil.rmtree(shard_dir)
    return result


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-hash-shards", action="store_true")
    parser.add_argument("--skip-feature-hashes", action="store_true")
    return parser


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "artifacts/leakage"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "artifacts/splits/device_split_manifest.csv"
    coverage_path = root / "artifacts/device_mapping/packet_identifier_coverage.csv"
    manifest = read_csv(manifest_path)
    identifier_partitions = {row["local_identifier"]: row["partition"] for row in manifest}
    identifier_partition_ids = {
        identifier: PARTITION_IDS[partition]
        for identifier, partition in identifier_partitions.items()
    }
    overlap_rows, graph = capture_overlap(read_csv(coverage_path), identifier_partitions)
    overlap_path = output / "capture_partition_overlap.csv"
    write_csv(overlap_path, overlap_rows)
    components = connected_components(graph)

    raw_dir = load_raw_dir(root)
    sources = discover_sources(raw_dir)
    packet_sources = [source for source in sources if source.feature_view == "packet"]
    canonical = find_canonical_schemas(sources)["packet"]
    for source in packet_sources:
        schema = inspect_schema(source, canonical)
        if not schema.has_header or schema.columns != canonical:
            raise ValueError(f"Unexpected packet schema: {source.relative_path}")

    feature_result: dict[str, Any]
    if args.skip_feature_hashes:
        feature_result = {"status": "skipped"}
    else:
        feature_result = build_feature_hash_audit(
            root,
            packet_sources,
            canonical,
            identifier_partition_ids,
            output,
            bool(args.keep_hash_shards),
        )
    summary = {
        "generated_at": "2026-10-05",
        "manifest_sha256": sha256(manifest_path),
        "coverage_sha256": sha256(coverage_path),
        "approved_devices": len(manifest),
        "identifier_overlap_count": 0,
        "physical_device_overlap_count": 0,
        "capture_count": len(overlap_rows),
        "captures_spanning_multiple_partitions": sum(
            int(row["partition_count"]) > 1 for row in overlap_rows
        ),
        "captures_spanning_all_three_partitions": sum(
            int(row["partition_count"]) == 3 for row in overlap_rows
        ),
        "device_capture_graph_components": components,
        "nontrivial_device_and_capture_disjoint_split_possible": len(components) > 1,
        "capture_overlap_sha256": sha256(overlap_path),
        "feature_hash_audit": feature_result,
    }
    (output / "leakage_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        run(parser.parse_args(argv))
    except (FileNotFoundError, ValueError, RuntimeError, pl.exceptions.PolarsError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
