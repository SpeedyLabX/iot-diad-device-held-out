"""Independently reconstruct raw row membership, features, and bottom-k sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from iot_diad.metadata import discover_sources
from iot_diad.modeling.baselines import CAP, CHUNK, SEED, sha256, write_json
from iot_diad.modeling.data import METADATA_COLUMNS


def priority_reference(capture: str, indices: np.ndarray) -> np.ndarray:
    salt = int.from_bytes(hashlib.sha256(f"{SEED}\0{capture}".encode()).digest()[:8], "little")
    # Python-integer reference deliberately independent of vectorized production sampling.
    mask = (1 << 64) - 1

    def one(index: int) -> int:
        z = ((index ^ salt) + 0x9E3779B97F4A7C15) & mask
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        return z ^ (z >> 31)

    return np.fromiter((one(int(i)) for i in indices), dtype=np.uint64, count=len(indices))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    started = time.monotonic()
    root = args.root
    output = root / "artifacts/experiments/baselines"
    config = json.loads((output / "config.json").read_text())
    prepared = json.loads((output / "prepared.json").read_text())
    features = config["features"]["numeric"] + config["features"]["categorical"]
    assignments = pl.read_csv(root / "artifacts/splits/device_split_manifest.csv").select(
        pl.col("local_identifier").alias("device_mac"),
        pl.col("canonical_device_id").alias("_canonical_device_id"),
        pl.col("partition").alias("_partition"),
    )
    exclusions = pl.read_csv(root / "artifacts/splits/cross_partition_duplicate_exclusions.csv")
    sources = [
        s
        for s in discover_sources(root / "data/raw/CIC-IoT-IDAD-Dataset-2024")
        if s.feature_view == "packet"
    ]
    assert len(sources) == len(prepared["sources"]) == 180
    all_counts: Counter[str] = Counter()
    raw_total = 0
    pool = None
    for source_index, source in enumerate(sources):
        audit = json.loads((output / "cache" / f"{source_index:03d}" / "complete.json").read_text())
        assert sha256(source.path) == audit["source_sha256"]
        removed = exclusions.filter(pl.col("capture_id") == source.capture_id)["row_index"]
        scan = pl.scan_csv(source.path, infer_schema=False).select("device_mac", *features)
        offset = 0
        expected_chunks = iter(audit["evaluation_chunks"])
        for raw in scan.collect_batches(chunk_size=CHUNK):
            raw = raw.with_row_index("_row_index", offset=offset)
            offset += raw.height
            frame = raw.join(assignments, on="device_mac", how="inner", maintain_order="left")
            frame = (
                frame.filter(~pl.col("_row_index").is_in(removed.implode()))
                .with_columns(
                    pl.col("device_mac").alias("_local_identifier"),
                    pl.lit(source.capture_id).alias("_capture_id"),
                    pl.lit(source.coarse_label).alias("_label"),
                )
                .select(*METADATA_COLUMNS, *features)
            )
            all_counts.update(dict(frame.group_by("_partition").len().rows()))
            evaluation = frame.filter(pl.col("_partition") != "train")
            if evaluation.height:
                chunk = next(expected_chunks)
                cached = pl.read_parquet(output / chunk["path"])
                assert sha256(output / chunk["path"]) == chunk["sha256"]
                assert cached.equals(evaluation), chunk["path"]
            training = frame.filter(pl.col("_partition") == "train").select(METADATA_COLUMNS)
            if training.height:
                training = training.with_columns(
                    pl.Series(
                        "_priority",
                        priority_reference(source.capture_id, training["_row_index"].to_numpy()),
                    )
                )
                pool = pl.concat([pool, training]) if pool is not None else training
                pool = pool.sort("_priority", "_capture_id", "_row_index").filter(
                    pl.int_range(pl.len()).over("_canonical_device_id", "_label") < CAP
                )
        assert next(expected_chunks, None) is None
        assert offset == audit["raw_rows"]
        raw_total += offset
        print(f"raw verification {source_index + 1}/180", flush=True)
    assert pool is not None
    manifest = pl.read_csv(
        output / "training_sample_manifest.csv",
        schema_overrides={"_priority": pl.UInt64, "_row_index": pl.UInt32},
    )
    assert pool.sort("_capture_id", "_row_index").equals(manifest)
    assert dict(all_counts) == prepared["rows_by_partition"]
    assert raw_total == 35_672_839
    write_json(
        output / "input_verification.json",
        {
            "status": "passed",
            "raw_packet_rows": raw_total,
            "rows_by_partition": dict(all_counts),
            "training_manifest_reconstructed_from_raw": True,
            "every_evaluation_feature_and_row_key_compared_to_raw": True,
            "source_sha256_verified": len(sources),
            "runtime_seconds": time.monotonic() - started,
        },
    )


if __name__ == "__main__":
    main()
