from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from iot_diad.metadata import FeatureView
from iot_diad.roles import REPRESENTATIVE_FEATURES, candidate_feature_columns

MAC_ADDRESS = re.compile(r"^(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")


def load_samples(manifest: pl.DataFrame, feature_view: FeatureView) -> pl.DataFrame:
    paths = (
        manifest.filter(pl.col("feature_view") == feature_view)["sample_path"]
        .cast(pl.String)
        .to_list()
    )
    if not paths:
        raise ValueError(f"No sample files found for {feature_view}")
    return pl.concat([pl.read_parquet(path) for path in paths], how="vertical")


def balance_classes(frame: pl.DataFrame, limit: int, seed: int) -> pl.DataFrame:
    groups: list[pl.DataFrame] = []
    labels = frame["_coarse_label"].unique().sort().to_list()
    for index, label in enumerate(labels):
        group = frame.filter(pl.col("_coarse_label") == label)
        if group.height > limit:
            group = group.sample(n=limit, seed=seed + index, shuffle=True)
        groups.append(group)
    return pl.concat(groups, how="vertical")


def summarize_distributions(
    frame: pl.DataFrame,
    feature_view: FeatureView,
    columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    features = candidate_feature_columns(feature_view, columns)
    rows = summarize_group(frame, feature_view, features, "All")
    for label in frame["_coarse_label"].unique().sort().to_list():
        group = frame.filter(pl.col("_coarse_label") == label)
        rows.extend(summarize_group(group, feature_view, features, str(label)))
    return rows


def summarize_group(
    frame: pl.DataFrame,
    feature_view: FeatureView,
    features: Sequence[str],
    label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in features:
        values = finite_values(frame[feature])
        sample_count = len(values)
        missing_count = frame.height - sample_count
        row: dict[str, Any] = {
            "feature_view": feature_view,
            "coarse_label": label,
            "feature": feature,
            "sample_rows": frame.height,
            "valid_rows": sample_count,
            "missing_rate": ratio(missing_count, frame.height),
            "zero_rate": ratio(int(np.count_nonzero(values == 0)), sample_count),
            "sample_unique": int(np.unique(values).size) if sample_count else 0,
        }
        for name, quantile in (
            ("p01", 0.01),
            ("q25", 0.25),
            ("median", 0.5),
            ("q75", 0.75),
            ("p99", 0.99),
        ):
            row[name] = float(np.quantile(values, quantile)) if sample_count else None
        row.update(
            {
                "min": float(values.min()) if sample_count else None,
                "max": float(values.max()) if sample_count else None,
                "mean": float(values.mean()) if sample_count else None,
                "std": float(values.std(ddof=1)) if sample_count > 1 else None,
            }
        )
        rows.append(row)
    return rows


def summarize_histograms(
    frame: pl.DataFrame,
    feature_view: FeatureView,
    bins: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in REPRESENTATIVE_FEATURES[feature_view]:
        overall = finite_values(frame[feature])
        if not len(overall):
            continue
        lower, upper = np.quantile(overall, [0.01, 0.99])
        if not math.isfinite(lower) or not math.isfinite(upper) or lower == upper:
            continue
        edges = np.linspace(lower, upper, bins + 1)
        for label in frame["_coarse_label"].unique().sort().to_list():
            values = finite_values(frame.filter(pl.col("_coarse_label") == label)[feature])
            values = values[(values >= lower) & (values <= upper)]
            counts, _ = np.histogram(values, bins=edges)
            denominator = max(int(counts.sum()), 1)
            for index, count in enumerate(counts):
                rows.append(
                    {
                        "feature_view": feature_view,
                        "feature": feature,
                        "coarse_label": str(label),
                        "bin_left": float(edges[index]),
                        "bin_right": float(edges[index + 1]),
                        "count": int(count),
                        "density": float(count / denominator),
                        "clipped_lower": float(lower),
                        "clipped_upper": float(upper),
                    }
                )
    return rows


def summarize_correlations(
    frame: pl.DataFrame,
    feature_view: FeatureView,
    columns: tuple[str, ...],
    row_limit: int,
    feature_limit: int,
    seed: int,
) -> tuple[list[dict[str, Any]], pl.DataFrame]:
    if frame.height > row_limit:
        frame = frame.sample(n=row_limit, seed=seed, shuffle=True)
    candidates = candidate_feature_columns(feature_view, columns)
    usable = [column for column in candidates if usable_for_correlation(frame[column])]
    ranked = np.column_stack([rank_values(frame[column]) for column in usable])
    matrix = np.corrcoef(ranked, rowvar=False)
    pairs: list[dict[str, Any]] = []
    for left in range(len(usable)):
        for right in range(left + 1, len(usable)):
            correlation = float(matrix[left, right])
            if math.isfinite(correlation):
                pairs.append(
                    {
                        "feature_view": feature_view,
                        "left_feature": usable[left],
                        "right_feature": usable[right],
                        "spearman_correlation": correlation,
                        "absolute_correlation": abs(correlation),
                        "sample_rows": frame.height,
                    }
                )
    pairs.sort(key=lambda row: (-float(row["absolute_correlation"]), str(row["left_feature"])))
    selected = select_correlated_features(matrix, usable, feature_limit)
    indices = [usable.index(column) for column in selected]
    selected_matrix = matrix[np.ix_(indices, indices)]
    heatmap = pl.DataFrame(
        {"feature": selected, **dict(zip(selected, selected_matrix.T, strict=True))}
    )
    return pairs, heatmap


def summarize_sample_duplicates(
    frame: pl.DataFrame,
    feature_view: FeatureView,
    columns: tuple[str, ...],
    seed: int,
) -> dict[str, Any]:
    features = candidate_feature_columns(feature_view, columns)
    groups = (
        frame.select(
            pl.struct(features).hash(seed=seed).alias("feature_hash"),
            "_capture_id",
            "_coarse_label",
        )
        .group_by("feature_hash")
        .agg(
            pl.len().alias("rows"),
            pl.col("_capture_id").n_unique().alias("capture_count"),
            pl.col("_coarse_label").n_unique().alias("class_count"),
        )
    )
    duplicate_rows = int((groups["rows"] - 1).clip(0).sum())
    cross_capture = groups.filter(pl.col("capture_count") > 1)
    cross_class = groups.filter(pl.col("class_count") > 1)
    return {
        "feature_view": feature_view,
        "sample_rows": frame.height,
        "duplicate_rows": duplicate_rows,
        "duplicate_rate": ratio(duplicate_rows, frame.height),
        "rows_in_cross_capture_groups": int(cross_capture["rows"].sum() or 0),
        "cross_capture_group_count": cross_capture.height,
        "rows_in_cross_class_groups": int(cross_class["rows"].sum() or 0),
        "cross_class_group_count": cross_class.height,
        "evidence_scope": "capture-aware sample",
    }


def summarize_exact_duplicates(frame: pl.DataFrame) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for view in frame["feature_view"].unique().sort().to_list():
        view_frame = frame.filter(pl.col("feature_view") == view)
        frames.append(duplicate_aggregate(view_frame, str(view), "All"))
        for label in view_frame["coarse_label"].unique().sort().to_list():
            group = view_frame.filter(pl.col("coarse_label") == label)
            frames.append(duplicate_aggregate(group, str(view), str(label)))
    return pl.concat(frames)


def duplicate_aggregate(frame: pl.DataFrame, view: str, label: str) -> pl.DataFrame:
    rows = int(frame["row_count"].sum())
    raw_duplicates = int(frame["raw_duplicate_rows"].sum())
    feature_duplicates = int(frame["feature_duplicate_rows"].sum())
    return pl.DataFrame(
        {
            "feature_view": [view],
            "coarse_label": [label],
            "capture_count": [frame.height],
            "row_count": [rows],
            "raw_duplicate_rows": [raw_duplicates],
            "raw_duplicate_rate": [ratio(raw_duplicates, rows)],
            "feature_duplicate_rows": [feature_duplicates],
            "feature_duplicate_rate": [ratio(feature_duplicates, rows)],
            "captures_with_raw_duplicates": [int((frame["raw_duplicate_rows"] > 0).sum())],
            "captures_with_feature_duplicates": [int((frame["feature_duplicate_rows"] > 0).sum())],
        }
    )


def summarize_identifiers(frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    packet = frame.filter(
        (pl.col("feature_view") == "packet") & (pl.col("field") == "device_mac")
    ).with_columns(
        pl.col("value")
        .map_elements(identifier_representation, return_dtype=pl.String)
        .alias("representation")
    )
    packet_profile = (
        packet.group_by(["value", "representation"])
        .agg(
            pl.col("coarse_label").n_unique().alias("class_count"),
            pl.col("coarse_label").unique().sort().str.join("|").alias("classes"),
            pl.col("row_count").sum().alias("row_count"),
            pl.col("capture_count").sum().alias("capture_count"),
        )
        .sort(["representation", "value"])
    )
    coverage = (
        frame.group_by(["feature_view", "field", "coarse_label"])
        .agg(
            pl.col("value").n_unique().alias("distinct_values"),
            pl.col("row_count").sum().alias("row_count"),
            pl.col("capture_count").sum().alias("field_capture_occurrences"),
        )
        .sort(["feature_view", "field", "coarse_label"])
    )
    return packet_profile, coverage


def identifier_representation(value: str) -> str:
    if MAC_ADDRESS.fullmatch(value):
        return "MAC address"
    if value.strip().lower() in {"not sure", "unknown", "n/a"}:
        return "ambiguous text"
    return "device name"


def finite_values(series: pl.Series) -> np.ndarray[Any, np.dtype[np.float64]]:
    values = series.cast(pl.Float64, strict=False).to_numpy().astype(np.float64, copy=False)
    return values[np.isfinite(values)]


def usable_for_correlation(series: pl.Series) -> bool:
    values = finite_values(series)
    return len(values) >= len(series) / 2 and np.unique(values).size > 1


def rank_values(series: pl.Series) -> np.ndarray[Any, np.dtype[np.float64]]:
    values = series.cast(pl.Float64, strict=False)
    finite = finite_values(values)
    median = float(np.median(finite)) if len(finite) else 0.0
    values = values.fill_nan(None).fill_null(median)
    values = values.set(values.is_infinite(), median)
    return values.rank(method="average").to_numpy().astype(np.float64, copy=False)


def select_correlated_features(
    matrix: np.ndarray[Any, np.dtype[np.float64]],
    columns: Sequence[str],
    limit: int,
) -> list[str]:
    adjusted = np.abs(matrix.copy())
    np.fill_diagonal(adjusted, np.nan)
    scores = np.nanmax(adjusted, axis=1)
    order = np.argsort(np.nan_to_num(scores, nan=-1.0))[::-1]
    return [columns[int(index)] for index in order[:limit]]


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan
