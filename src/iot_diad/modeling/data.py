"""Apply approved device partitions and row exclusions to packet data."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from iot_diad.modeling.features import FeatureSpec

METADATA_COLUMNS = (
    "_partition",
    "_canonical_device_id",
    "_local_identifier",
    "_capture_id",
    "_row_index",
    "_label",
)


@dataclass(frozen=True)
class PartitionContract:
    device_by_identifier: dict[str, str]
    partition_by_identifier: dict[str, str]
    excluded_rows_by_capture: dict[str, frozenset[int]]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_partition_contract(manifest_path: Path, exclusion_path: Path) -> PartitionContract:
    manifest = read_csv(manifest_path)
    device_by_identifier = {row["local_identifier"]: row["canonical_device_id"] for row in manifest}
    partition_by_identifier = {row["local_identifier"]: row["partition"] for row in manifest}
    if len(device_by_identifier) != len(manifest):
        raise ValueError("Local packet identifiers must be unique in the split manifest")
    if set(partition_by_identifier.values()) != {"train", "calibration", "test"}:
        raise ValueError("Split manifest must contain train, calibration, and test partitions")

    excluded: dict[str, set[int]] = defaultdict(set)
    for row in read_csv(exclusion_path):
        excluded[row["capture_id"]].add(int(row["row_index"]))
    return PartitionContract(
        device_by_identifier=device_by_identifier,
        partition_by_identifier=partition_by_identifier,
        excluded_rows_by_capture={
            capture: frozenset(indices) for capture, indices in excluded.items()
        },
    )


def select_modeling_rows(
    frame: pl.DataFrame,
    capture_id: str,
    coarse_label: str,
    spec: FeatureSpec,
    contract: PartitionContract,
    row_offset: int = 0,
) -> pl.DataFrame:
    required = {"device_mac", *spec.columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Packet frame is missing required columns: {sorted(missing)}")
    if "_row_index" in frame.columns:
        raise ValueError("Input frame must not define the reserved _row_index column")

    selected = frame.with_row_index("_row_index", offset=row_offset).filter(
        pl.col("device_mac").is_in(contract.device_by_identifier)
    )
    excluded = contract.excluded_rows_by_capture.get(capture_id, frozenset())
    if excluded:
        selected = selected.filter(~pl.col("_row_index").is_in(excluded))
    return selected.select(
        pl.col("device_mac").replace_strict(contract.partition_by_identifier).alias("_partition"),
        pl.col("device_mac")
        .replace_strict(contract.device_by_identifier)
        .alias("_canonical_device_id"),
        pl.col("device_mac").alias("_local_identifier"),
        pl.lit(capture_id).alias("_capture_id"),
        pl.col("_row_index"),
        pl.lit(coarse_label).alias("_label"),
        *spec.columns,
    )
