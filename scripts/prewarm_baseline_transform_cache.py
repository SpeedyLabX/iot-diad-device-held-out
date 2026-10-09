"""Precompute a disjoint evaluation-transform range for an active BASE-002 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from iot_diad.modeling.baselines import (
    cached_evaluation_features,
    make_transform_contract,
    object_hash,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", default="LR")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    args = parser.parse_args()

    output = args.root / "artifacts/experiments/baselines"
    prepared = json.loads((output / "prepared.json").read_text())
    run = output / args.run_id
    config = json.loads((run / "run_config.json").read_text())
    processor, _ = joblib.load(run / "model.joblib")
    transform_contract = make_transform_contract(
        args.root,
        processor,
        config["sample_manifest_sha256"],
        prepared["feature_contract_sha256"],
    )
    cache_key = object_hash(transform_contract)
    if cache_key != json.loads((run / "transform_cache.json").read_text())["cache_key"]:
        raise ValueError("Transform cache key differs from active run")
    sources = prepared["evaluation_chunks"]
    if not 0 <= args.start < args.stop <= len(sources):
        raise ValueError("Invalid half-open chunk range")
    for index in range(args.start, args.stop):
        cached_evaluation_features(output, sources[index], processor, cache_key)
        if index % 25 == 0:
            print(f"prewarmed {index}/{args.stop - 1}", flush=True)


if __name__ == "__main__":
    main()
