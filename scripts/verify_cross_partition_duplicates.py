"""Verify cross-partition dual-hash matches against complete raw feature vectors."""

from __future__ import annotations

import csv
import hashlib
import json
import time
import tomllib
from collections import defaultdict
from pathlib import Path

import polars as pl

from iot_diad.metadata import discover_sources
from iot_diad.roles import candidate_feature_columns
from iot_diad.schema import find_canonical_schemas

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/leakage"
PARTITION_IDS = {"train": 0, "calibration": 1, "test": 2}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def vector_digest(values: list[str | None]) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    raw = tomllib.loads((ROOT / "config/audit.toml").read_text(encoding="utf-8"))
    raw_dir = ROOT / str(raw["paths"]["raw_dir"])
    all_sources = discover_sources(raw_dir)
    sources = [source for source in all_sources if source.feature_view == "packet"]
    columns = find_canonical_schemas(all_sources)["packet"]
    features = candidate_feature_columns("packet", columns)
    manifest = read_csv(ROOT / "artifacts/splits/device_split_manifest.csv")
    partitions = {row["local_identifier"]: row["partition"] for row in manifest}
    partition_ids = {identifier: PARTITION_IDS[value] for identifier, value in partitions.items()}
    cross = pl.read_parquet(OUTPUT / "cross_partition_feature_hashes.parquet")
    pairs = [(int(row[0]), int(row[1])) for row in cross.select("hash_a", "hash_b").rows()]
    pair_filter = pl.any_horizontal(
        [(pl.col("hash_a") == first) & (pl.col("hash_b") == second) for first, second in pairs]
    )

    matches: list[dict[str, str]] = []
    started = time.monotonic()
    schema = {column: pl.String for column in columns}
    for index, source in enumerate(sources, start=1):
        partition = pl.col("device_mac").replace_strict(partition_ids).cast(pl.UInt8)
        selected = (
            pl.scan_csv(
                source.path,
                has_header=True,
                schema=schema,
                low_memory=True,
                row_index_name="_row_index",
            )
            .filter(pl.col("device_mac").is_in(partition_ids))
            .with_columns(
                partition.alias("partition_id"),
                pl.struct(features)
                .hash(seed=2027, seed_1=2029, seed_2=2039, seed_3=2053)
                .alias("hash_a"),
                pl.struct(features)
                .hash(seed=2081, seed_1=2083, seed_2=2087, seed_3=2089)
                .alias("hash_b"),
            )
            .filter(pair_filter)
            .select("_row_index", "device_mac", "partition_id", "hash_a", "hash_b", *features)
            .collect(engine="streaming")
        )
        for row in selected.iter_rows(named=True):
            feature_values = [row[name] for name in features]
            partition_name = next(
                name for name, value in PARTITION_IDS.items() if value == row["partition_id"]
            )
            matches.append(
                {
                    "hash_a": str(row["hash_a"]),
                    "hash_b": str(row["hash_b"]),
                    "feature_vector_sha256": vector_digest(feature_values),
                    "partition": partition_name,
                    "coarse_label": source.coarse_label,
                    "device_mac": str(row["device_mac"]),
                    "capture_id": source.capture_id,
                    "row_index": str(row["_row_index"]),
                }
            )
        elapsed = max(time.monotonic() - started, 1e-9)
        eta = (len(sources) - index) / (index / elapsed)
        print(f"Verified {index}/{len(sources)} captures; ETA {eta / 60:.1f} min", flush=True)

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in matches:
        grouped[(row["hash_a"], row["hash_b"])].append(row)
    assert len(grouped) == len(pairs) == 8
    assert len(matches) == 58
    assert all(
        len({row["feature_vector_sha256"] for row in rows}) == 1 for rows in grouped.values()
    )
    assert all(len({row["partition"] for row in rows}) > 1 for rows in grouped.values())
    assert all(len({row["coarse_label"] for row in rows}) == 1 for rows in grouped.values())

    output_path = OUTPUT / "verified_cross_partition_duplicates.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matches[0]))
        writer.writeheader()
        writer.writerows(
            sorted(matches, key=lambda row: (row["hash_a"], row["capture_id"], row["row_index"]))
        )
    summary = {
        "status": "passed",
        "candidate_feature_count": len(features),
        "cross_partition_feature_vector_count": len(grouped),
        "affected_rows": len(matches),
        "cross_label_feature_vector_count": 0,
        "verification": "SHA-256 over canonical JSON arrays of all raw candidate-feature strings",
        "verified_rows_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    (OUTPUT / "duplicate_verification_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    leakage_path = OUTPUT / "leakage_summary.json"
    leakage_summary = json.loads(leakage_path.read_text(encoding="utf-8"))
    leakage_summary["raw_duplicate_verification"] = summary
    leakage_path.write_text(json.dumps(leakage_summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
