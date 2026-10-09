"""Export descriptive CONF-001 CSV tables from hash-verified artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from iot_diad.modeling.baselines import sha256
from iot_diad.modeling.conformal import BASE, METHODS, OUTPUT, read_json, require


def tables(root: Path) -> dict[str, list[dict[str, Any]]]:
    registry = read_json(root / BASE / "accepted_model_registry.json")
    output = root / OUTPUT
    result: dict[str, list[dict[str, Any]]] = {
        "model_summary": [],
        "class_coverage": [],
        "device_class_coverage": [],
        "calibration_support": [],
        "threshold_summary": [],
    }
    for model in registry["models"]:
        if not model["primary_learned_model"]:
            continue
        name = model["model_id"]
        directory = output / name
        manifest = read_json(directory / "manifest.json")
        for filename, key in [
            ("metrics.json", "metrics_sha256"),
            ("thresholds.json", "thresholds_sha256"),
        ]:
            require(sha256(directory / filename) == manifest[key], "Source artifact hash")
        metrics = read_json(directory / "metrics.json")
        thresholds = read_json(directory / "thresholds.json")
        for method in METHODS:
            prefix = {"model_id": name, "run_id": model["run_id"], "method": method}
            value = metrics[method]
            result["model_summary"].append(
                {**prefix, **{k: v for k, v in value.items() if k not in ("classes", "cells")}}
            )
            result["class_coverage"].extend({**prefix, **row} for row in value["classes"])
            result["device_class_coverage"].extend({**prefix, **row} for row in value["cells"])
            result["device_class_coverage"].extend(
                {
                    **prefix,
                    "device": row["device"],
                    "label": row["label"],
                    "rows": 0,
                    "covered": 0,
                    "coverage": None,
                }
                for row in metrics["absent_test_device_class_cells"]
            )
        for row in thresholds["classes"]:
            result["threshold_summary"].append(
                {
                    "model_id": name,
                    "label": row["label"],
                    "K_y": row["K_y"],
                    "N_y": row["N_y"],
                    "infinity_weight": row["infinity_weight"],
                    "hierarchical": row["hierarchical"],
                    "equal_row_diagnostic": row["equal_row_diagnostic"],
                }
            )
            for device in row["devices"]:
                result["calibration_support"].append(
                    {
                        "model_id": name,
                        "label": row["label"],
                        "K_y": row["K_y"],
                        "device": device["device"],
                        "N_ky": device["N_ky"],
                        "device_weight": device["device_weight"],
                        "score_weight": device["score_weight"],
                    }
                )
            for device in row["absent_calibration_devices"]:
                result["calibration_support"].append(
                    {
                        "model_id": name,
                        "label": row["label"],
                        "K_y": row["K_y"],
                        "device": device,
                        "N_ky": 0,
                        "device_weight": "0",
                        "score_weight": None,
                    }
                )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    for name, rows in tables(args.root).items():
        path = args.root / OUTPUT / f"{name}.csv"
        if args.verify_only:
            with path.open(newline="") as handle:
                actual = list(csv.DictReader(handle))
            expected = [{k: "" if v is None else str(v) for k, v in r.items()} for r in rows]
            require(actual == expected, f"CSV reconstruction mismatch: {path}")
        else:
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        print(f"{name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
