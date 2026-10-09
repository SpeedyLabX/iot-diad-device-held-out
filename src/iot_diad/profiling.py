from __future__ import annotations

import hashlib
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl

from iot_diad.metadata import FeatureView, SourceFile
from iot_diad.roles import candidate_feature_columns, column_role, is_numeric_column
from iot_diad.schema import SchemaInfo


def scan_strings(source: SourceFile, schema: SchemaInfo, row_index: bool = False) -> pl.LazyFrame:
    options: dict[str, Any] = {
        "has_header": schema.has_header,
        "schema": {column: pl.String for column in schema.columns},
        "low_memory": True,
    }
    if row_index:
        options["row_index_name"] = "_row_index"
    return pl.scan_csv(source.path, **options)


def profile_view(
    sources: Sequence[SourceFile],
    schemas: dict[str, SchemaInfo],
    feature_view: FeatureView,
) -> list[dict[str, Any]]:
    profiles, _ = profile_view_and_duplicates(sources, schemas, feature_view)
    return profiles


def profile_view_and_duplicates(
    sources: Sequence[SourceFile],
    schemas: dict[str, SchemaInfo],
    feature_view: FeatureView,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = [source for source in sources if source.feature_view == feature_view]
    if not selected:
        return [], []
    columns = schemas[selected[0].relative_path].columns
    expressions = profile_expressions(feature_view, columns)
    features = candidate_feature_columns(feature_view, columns)
    capture_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, source in enumerate(selected, start=1):
        frame = (
            scan_strings(source, schemas[source.relative_path])
            .select(
                *expressions,
                pl.struct(columns).n_unique().alias("__raw_unique_rows"),
                pl.struct(features).n_unique().alias("__feature_unique_rows"),
            )
            .collect(engine="streaming")
        )
        capture_rows.extend(
            unpack_profile(frame, feature_view, columns, "capture", source.coarse_label)
        )
        duplicate_rows.append(unpack_duplicate(frame, source))
        show_progress("Exact profile", index, len(selected), started, source.relative_path)
    sys.stdout.write("\n")

    class_rows: list[dict[str, Any]] = []
    labels = sorted({source.coarse_label for source in selected})
    for label in labels:
        class_rows.extend(
            combine_profiles([row for row in capture_rows if row["group"] == label], "class", label)
        )
    profiles = combine_profiles(capture_rows, "overall", "All") + class_rows
    return profiles, duplicate_rows


def profile_expressions(feature_view: FeatureView, columns: tuple[str, ...]) -> list[pl.Expr]:
    expressions: list[pl.Expr] = [pl.len().alias("__row_count")]
    for index, column in enumerate(columns):
        text = pl.col(column).str.strip_chars()
        missing = text.is_null() | (text == "")
        expressions.append(missing.sum().alias(f"c{index}__missing"))
        if not is_numeric_column(feature_view, column):
            continue
        number = text.cast(pl.Float64, strict=False)
        parse_failure = (~missing) & number.is_null()
        finite = number.is_finite()
        valid = pl.when(finite).then(number)
        expressions.extend(
            [
                parse_failure.sum().alias(f"c{index}__parse_failure"),
                (number.is_not_null() & ~finite).sum().alias(f"c{index}__nonfinite"),
                finite.sum().alias(f"c{index}__valid"),
                (finite & (number == 0)).sum().alias(f"c{index}__zero"),
                valid.min().alias(f"c{index}__min"),
                valid.max().alias(f"c{index}__max"),
                valid.mean().alias(f"c{index}__mean"),
                valid.std(ddof=1).alias(f"c{index}__std"),
            ]
        )
    return expressions


def unpack_profile(
    frame: pl.DataFrame,
    feature_view: FeatureView,
    columns: tuple[str, ...],
    scope: str,
    group: str,
) -> list[dict[str, Any]]:
    values = frame.row(0, named=True)
    row_count = int(values["__row_count"])
    rows: list[dict[str, Any]] = []
    for index, column in enumerate(columns):
        numeric = is_numeric_column(feature_view, column)
        missing = int(values[f"c{index}__missing"])
        row = {
            "feature_view": feature_view,
            "scope": scope,
            "group": group,
            "column": column,
            "role": column_role(column),
            "inferred_kind": "numeric" if numeric else "categorical",
            "row_count": row_count,
            "missing_count": missing,
            "missing_rate": missing / row_count if row_count else math.nan,
            "parse_failure_count": int(values.get(f"c{index}__parse_failure", 0) or 0),
            "nonfinite_count": int(values.get(f"c{index}__nonfinite", 0) or 0),
            "valid_numeric_count": int(values.get(f"c{index}__valid", 0) or 0),
            "zero_count": int(values.get(f"c{index}__zero", 0) or 0),
            "min": values.get(f"c{index}__min"),
            "max": values.get(f"c{index}__max"),
            "mean": values.get(f"c{index}__mean"),
            "std": values.get(f"c{index}__std"),
        }
        rows.append(row)
    return rows


def combine_profiles(
    rows: Sequence[dict[str, Any]], scope: str, group: str
) -> list[dict[str, Any]]:
    columns = sorted({str(row["column"]) for row in rows})
    combined: list[dict[str, Any]] = []
    for column in columns:
        groups = [row for row in rows if row["column"] == column]
        row_count = sum(int(row["row_count"]) for row in groups)
        valid_count = sum(int(row["valid_numeric_count"]) for row in groups)
        result = dict(groups[0])
        result.update(
            {
                "scope": scope,
                "group": group,
                "row_count": row_count,
                "missing_count": sum(int(row["missing_count"]) for row in groups),
                "parse_failure_count": sum(int(row["parse_failure_count"]) for row in groups),
                "nonfinite_count": sum(int(row["nonfinite_count"]) for row in groups),
                "valid_numeric_count": valid_count,
                "zero_count": sum(int(row["zero_count"]) for row in groups),
            }
        )
        result["missing_rate"] = ratio(int(result["missing_count"]), row_count)
        result["min"] = optional_extreme(groups, "min", min)
        result["max"] = optional_extreme(groups, "max", max)
        result["mean"] = pooled_mean(groups, valid_count)
        result["std"] = pooled_std(groups, valid_count, result["mean"])
        combined.append(result)
    return combined


def optional_extreme(rows: Sequence[dict[str, Any]], field: str, operation: Any) -> float | None:
    values = [float(row[field]) for row in rows if row[field] is not None]
    return operation(values) if values else None


def pooled_mean(rows: Sequence[dict[str, Any]], valid_count: int) -> float | None:
    if not valid_count:
        return None
    total = sum(
        int(row["valid_numeric_count"]) * float(row["mean"])
        for row in rows
        if row["mean"] is not None
    )
    return total / valid_count


def pooled_std(rows: Sequence[dict[str, Any]], valid_count: int, mean: Any) -> float | None:
    if valid_count < 2 or mean is None:
        return None
    squared_deviations = 0.0
    for row in rows:
        count = int(row["valid_numeric_count"])
        if not count or row["mean"] is None:
            continue
        group_mean = float(row["mean"])
        group_std = float(row["std"] or 0.0)
        squared_deviations += (count - 1) * group_std**2
        squared_deviations += count * (group_mean - float(mean)) ** 2
    return math.sqrt(squared_deviations / (valid_count - 1))


def profile_duplicates(
    sources: Sequence[SourceFile], schemas: dict[str, SchemaInfo]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, source in enumerate(sources, start=1):
        schema = schemas[source.relative_path]
        features = candidate_feature_columns(source.feature_view, schema.columns)
        lazy = scan_strings(source, schema)
        counts = lazy.select(
            pl.len().alias("row_count"),
            pl.struct(schema.columns).n_unique().alias("raw_unique_rows"),
            pl.struct(features).n_unique().alias("feature_unique_rows"),
        ).collect(engine="streaming")
        values = counts.row(0, named=True)
        row_count = int(values["row_count"])
        raw_unique = int(values["raw_unique_rows"])
        feature_unique = int(values["feature_unique_rows"])
        rows.append(
            {
                "relative_path": source.relative_path,
                "feature_view": source.feature_view,
                "coarse_label": source.coarse_label,
                "fine_label": source.fine_label,
                "capture_id": source.capture_id,
                "row_count": row_count,
                "raw_duplicate_rows": row_count - raw_unique,
                "raw_duplicate_rate": ratio(row_count - raw_unique, row_count),
                "feature_duplicate_rows": row_count - feature_unique,
                "feature_duplicate_rate": ratio(row_count - feature_unique, row_count),
            }
        )
        show_progress("Duplicates", index, len(sources), started, source.relative_path)
    sys.stdout.write("\n")
    return rows


def unpack_duplicate(frame: pl.DataFrame, source: SourceFile) -> dict[str, Any]:
    values = frame.row(0, named=True)
    row_count = int(values["__row_count"])
    raw_unique = int(values["__raw_unique_rows"])
    feature_unique = int(values["__feature_unique_rows"])
    return {
        "relative_path": source.relative_path,
        "feature_view": source.feature_view,
        "coarse_label": source.coarse_label,
        "fine_label": source.fine_label,
        "capture_id": source.capture_id,
        "row_count": row_count,
        "raw_duplicate_rows": row_count - raw_unique,
        "raw_duplicate_rate": ratio(row_count - raw_unique, row_count),
        "feature_duplicate_rows": row_count - feature_unique,
        "feature_duplicate_rate": ratio(row_count - feature_unique, row_count),
    }


def sample_sources(
    sources: Sequence[SourceFile],
    schemas: dict[str, SchemaInfo],
    row_counts: dict[str, int],
    sample_dir: Path,
    rows_per_capture: int,
    seed: int,
) -> list[dict[str, Any]]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, source in enumerate(sources, start=1):
        schema = schemas[source.relative_path]
        row_count = row_counts[source.relative_path]
        positions = systematic_positions(row_count, rows_per_capture, source.capture_id, seed)
        sample = (
            scan_strings(source, schema, row_index=True)
            .filter(pl.col("_row_index").is_in(positions))
            .with_columns(sample_casts(source.feature_view, schema.columns))
            .with_columns(
                pl.lit(source.coarse_label).alias("_coarse_label"),
                pl.lit(source.fine_label).alias("_fine_label"),
                pl.lit(source.capture_id).alias("_capture_id"),
                pl.lit(source.relative_path).alias("_source_path"),
            )
            .collect(engine="streaming")
        )
        path = sample_path(sample_dir, source)
        path.parent.mkdir(parents=True, exist_ok=True)
        sample.write_parquet(path, compression="zstd", statistics=True)
        manifest.append(
            {
                "feature_view": source.feature_view,
                "coarse_label": source.coarse_label,
                "capture_id": source.capture_id,
                "source_rows": row_count,
                "sample_rows": sample.height,
                "sample_path": path.as_posix(),
            }
        )
        show_progress("Sampling", index, len(sources), started, source.relative_path)
    sys.stdout.write("\n")
    return manifest


def systematic_positions(row_count: int, limit: int, key: str, seed: int) -> list[int]:
    if row_count <= limit:
        return list(range(row_count))
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    offset_fraction = int.from_bytes(digest[:8], "big") / 2**64
    step = row_count / limit
    return [min(int((index + offset_fraction) * step), row_count - 1) for index in range(limit)]


def sample_casts(feature_view: FeatureView, columns: tuple[str, ...]) -> list[pl.Expr]:
    return [
        pl.col(column).cast(pl.Float64, strict=False).alias(column)
        for column in columns
        if is_numeric_column(feature_view, column)
    ]


def sample_path(sample_dir: Path, source: SourceFile) -> Path:
    digest = hashlib.sha256(source.relative_path.encode()).hexdigest()[:12]
    return sample_dir / source.feature_view / f"{digest}.parquet"


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def show_progress(stage: str, completed: int, total: int, started: float, detail: str) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    remaining = (total - completed) / (completed / elapsed)
    message = f"{stage} {completed:>3}/{total} | ETA {format_duration(remaining)} | {detail}"
    sys.stdout.write("\r\033[K" + message)
    sys.stdout.flush()


def format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"
