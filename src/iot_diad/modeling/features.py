"""Model feature contracts derived from the audited column-role artifact."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]

    @property
    def columns(self) -> tuple[str, ...]:
        return self.numeric + self.categorical

    def validate(self) -> None:
        if not self.columns:
            raise ValueError("At least one model feature is required")
        if len(self.columns) != len(set(self.columns)):
            raise ValueError("Model features must be unique")


def load_packet_feature_spec(path: Path) -> FeatureSpec:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["feature_view"] == "packet"]
    numeric = tuple(
        row["column"] for row in rows if row["primary_baseline_action"] == "include_numeric"
    )
    categorical = tuple(
        row["column"] for row in rows if row["primary_baseline_action"] == "encode_after_validation"
    )
    spec = FeatureSpec(numeric=numeric, categorical=categorical)
    spec.validate()
    selected = set(spec.columns)
    forbidden = {
        row["column"]
        for row in rows
        if row["role"]
        in {"group_identifier", "session_identifier", "source_label", "high_risk_metadata"}
    }
    overlap = selected & forbidden
    if overlap:
        raise ValueError(f"Forbidden columns selected as model features: {sorted(overlap)}")
    return spec
