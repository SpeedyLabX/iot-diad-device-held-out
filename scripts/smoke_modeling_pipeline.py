"""Run the train-only preprocessor against deterministic raw packet rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from iot_diad.metadata import discover_sources
from iot_diad.modeling import (
    TrainOnlyPreprocessor,
    load_packet_feature_spec,
    load_partition_contract,
    select_modeling_rows,
)
from iot_diad.schema import find_canonical_schemas

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    raw_dir = ROOT / "data/raw/CIC-IoT-IDAD-Dataset-2024"
    sources = discover_sources(raw_dir)
    packet_sources = [source for source in sources if source.feature_view == "packet"]
    columns = find_canonical_schemas(sources)["packet"]
    roles_path = ROOT / "artifacts/eda/column_roles.csv"
    manifest_path = ROOT / "artifacts/splits/device_split_manifest.csv"
    exclusions_path = ROOT / "artifacts/splits/cross_partition_duplicate_exclusions.csv"
    spec = load_packet_feature_spec(roles_path)
    contract = load_partition_contract(manifest_path, exclusions_path)

    selected_source = None
    selected_frame = None
    training_frame = None
    for source in packet_sources:
        frame = pl.read_csv(
            source.path,
            has_header=True,
            schema={column: pl.String for column in columns},
            n_rows=20_000,
        )
        selected = select_modeling_rows(
            frame, source.capture_id, source.coarse_label, spec, contract
        )
        training = selected.filter(pl.col("_partition") == "train")
        if training.height >= 100:
            selected_source = source
            selected_frame = selected
            training_frame = training
            break
    if selected_source is None or selected_frame is None or training_frame is None:
        raise RuntimeError("No packet capture supplied enough training rows for the smoke test")

    processor = TrainOnlyPreprocessor(spec).fit(training_frame)
    transformed = processor.transform(selected_frame.head(1000))
    result = {
        "status": "passed",
        "source": selected_source.relative_path,
        "raw_prefix_rows": 20_000,
        "eligible_prefix_rows": selected_frame.height,
        "training_prefix_rows": training_frame.height,
        "input_features": len(spec.columns),
        "output_rows": transformed.shape[0],
        "output_features": transformed.shape[1],
        "all_outputs_finite": bool(np.isfinite(transformed).all()),
        "preprocessor_audit": processor.audit(),
        "roles_sha256": sha256(roles_path),
        "manifest_sha256": sha256(manifest_path),
        "exclusions_sha256": sha256(exclusions_path),
    }
    output = ROOT / "artifacts/modeling"
    output.mkdir(parents=True, exist_ok=True)
    (output / "preprocessing_smoke.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
