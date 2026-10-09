from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from iot_diad.metadata import FeatureView, SourceFile

EXPECTED_FIRST_COLUMN: dict[FeatureView, str] = {
    "flow": "Flow ID",
    "packet": "stream",
}


@dataclass(frozen=True, slots=True)
class SchemaInfo:
    has_header: bool
    status: str
    column_count: int
    schema_hash: str
    columns: tuple[str, ...]


def read_first_record(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        try:
            return tuple(next(csv.reader(handle)))
        except StopIteration as exc:
            raise ValueError(f"CSV file is empty: {path}") from exc


def find_canonical_schemas(sources: list[SourceFile]) -> dict[FeatureView, tuple[str, ...]]:
    schemas: dict[FeatureView, tuple[str, ...]] = {}
    for source in sources:
        record = read_first_record(source.path)
        if record and record[0] == EXPECTED_FIRST_COLUMN[source.feature_view]:
            schemas.setdefault(source.feature_view, record)
    missing = set(EXPECTED_FIRST_COLUMN).difference(schemas)
    if missing:
        raise ValueError(f"No canonical header found for feature views: {sorted(missing)}")
    return schemas


def inspect_schema(source: SourceFile, canonical: tuple[str, ...]) -> SchemaInfo:
    first_record = read_first_record(source.path)
    expected = EXPECTED_FIRST_COLUMN[source.feature_view]
    has_header = bool(first_record and first_record[0] == expected)

    if has_header and first_record == canonical:
        status = "canonical"
        columns = first_record
    elif has_header:
        status = "header_mismatch"
        columns = first_record
    elif len(first_record) == len(canonical):
        status = "headerless"
        columns = canonical
    else:
        status = "invalid_width"
        columns = canonical

    return SchemaInfo(
        has_header=has_header,
        status=status,
        column_count=len(first_record),
        schema_hash=hash_columns(columns),
        columns=columns,
    )


def hash_columns(columns: tuple[str, ...]) -> str:
    payload = "\0".join(columns).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
