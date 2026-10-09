from pathlib import Path

import pytest

from iot_diad.metadata import SourceFile
from iot_diad.profiling import (
    combine_profiles,
    profile_duplicates,
    profile_view,
    systematic_positions,
)
from iot_diad.schema import SchemaInfo, hash_columns


def test_systematic_positions_are_deterministic_and_bounded() -> None:
    first = systematic_positions(10_000, 100, "capture-a", 2027)
    second = systematic_positions(10_000, 100, "capture-a", 2027)

    assert first == second
    assert len(first) == len(set(first)) == 100
    assert min(first) >= 0
    assert max(first) < 10_000


def test_exact_profile_counts_missing_invalid_and_nonfinite_values(tmp_path: Path) -> None:
    source, schema = make_flow_source(
        tmp_path,
        "Flow ID,Flow Duration,Label\na,1,x\nb,,x\nc,bad,x\nd,inf,x\n",
    )

    rows = profile_view([source], {source.relative_path: schema}, "flow")
    duration = next(
        row for row in rows if row["scope"] == "overall" and row["column"] == "Flow Duration"
    )

    assert duration["row_count"] == 4
    assert duration["missing_count"] == 1
    assert duration["parse_failure_count"] == 1
    assert duration["nonfinite_count"] == 1
    assert duration["valid_numeric_count"] == 1


def test_duplicate_profile_separates_raw_and_candidate_features(tmp_path: Path) -> None:
    source, schema = make_flow_source(
        tmp_path,
        "Flow ID,Flow Duration,Label\na,1,x\nb,1,x\nc,2,x\nd,2,x\n",
    )

    rows = profile_duplicates([source], {source.relative_path: schema})

    assert rows[0]["raw_duplicate_rows"] == 0
    assert rows[0]["feature_duplicate_rows"] == 2


def test_profile_pooling_preserves_global_mean_and_sample_std() -> None:
    base = {
        "feature_view": "flow",
        "scope": "capture",
        "column": "Flow Duration",
        "role": "candidate_feature",
        "inferred_kind": "numeric",
        "row_count": 2,
        "missing_count": 0,
        "missing_rate": 0.0,
        "parse_failure_count": 0,
        "nonfinite_count": 0,
        "valid_numeric_count": 2,
        "zero_count": 0,
    }
    rows = [
        {**base, "group": "A", "min": 1.0, "max": 2.0, "mean": 1.5, "std": 2**-0.5},
        {**base, "group": "B", "min": 3.0, "max": 4.0, "mean": 3.5, "std": 2**-0.5},
    ]

    combined = combine_profiles(rows, "overall", "All")[0]

    assert combined["row_count"] == 4
    assert combined["mean"] == 2.5
    assert combined["std"] == pytest.approx((5 / 3) ** 0.5)


def make_flow_source(tmp_path: Path, content: str) -> tuple[SourceFile, SchemaInfo]:
    path = tmp_path / "capture.csv"
    path.write_text(content, encoding="utf-8")
    columns = ("Flow ID", "Flow Duration", "Label")
    source = SourceFile(
        path=path,
        relative_path="Anomaly Detection - Flow Based features/DoS/capture.csv",
        feature_view="flow",
        coarse_label="DoS",
        fine_label="DoS",
        capture_id="flow/DoS/capture",
        size_bytes=path.stat().st_size,
    )
    schema = SchemaInfo(
        has_header=True,
        status="canonical",
        column_count=3,
        schema_hash=hash_columns(columns),
        columns=columns,
    )
    return source, schema
