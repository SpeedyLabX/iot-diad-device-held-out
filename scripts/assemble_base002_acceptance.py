"""Assemble and verify the accepted BASE-002 model registry and point summary."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, cast

from iot_diad.modeling.baselines import CLASSES, sha256, write_json

EXPECTED_ROWS = 14_533_155
EXPECTED_SAMPLE = "099f821eb38f657c38dcf0f3b0cb060614417374339a43f7454d49e1ed07162c"
SOURCES = (
    ("LR", "lr_solver_amendment/LR-amended-newton-cholesky-maxiter200", 2000, True),
    ("ET", "ET", 2000, True),
    ("HGB", "HGB", 2000, True),
    ("RF", "RF", 2000, True),
    ("MLP", "amended_rerun/MLP-amended-maxiter200", 2000, True),
    ("DUMMY", "DUMMY", 2000, False),
    ("HGB", "HGB-cap1000", 1000, False),
)
FIELDS = (
    "run_id",
    "model_id",
    "role",
    "cap",
    "fit_status",
    "training_rows",
    "calibration_mean_device_macro_f1",
    "test_mean_device_macro_f1",
    "test_median_device_macro_f1",
    "test_q25",
    "test_q75",
    "test_iqr",
    "test_min",
    "test_max",
)


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def assemble(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline = root / "artifacts/experiments/baselines"
    models = []
    rows = []
    for model_id, relative, cap, primary in SOURCES:
        directory = baseline / relative
        run_config = read_json(directory / "run_config.json")
        fit = read_json(directory / "fit.json")
        exports = read_json(directory / "exports.json")
        metrics = read_json(directory / "metrics.json")
        if run_config["model_id"] != model_id or run_config["cap"] != cap:
            raise ValueError(f"Unexpected model identity or cap: {relative}")
        if tuple(run_config["class_order"]) != CLASSES:
            raise ValueError(f"Unexpected class order: {relative}")
        if cap == 2000 and run_config["sample_manifest_sha256"] != EXPECTED_SAMPLE:
            raise ValueError(f"Unexpected primary training sample: {relative}")
        if fit["status"] != "passed" or exports["fit_status"] != "passed":
            raise ValueError(f"Unaccepted fit selected: {relative}")
        if exports["status"] != "complete" or exports["rows"] != EXPECTED_ROWS:
            raise ValueError(f"Incomplete evaluation export: {relative}")
        expected_training = 446_363 if cap == 2000 else 258_876
        if fit["training_rows"] != expected_training:
            raise ValueError(f"Unexpected training row count: {relative}")
        run_id = run_config["run_id"]
        models.append(
            {
                "artifact_directory": str(directory.relative_to(root)),
                "cap": cap,
                "class_order": list(CLASSES),
                "exports_sha256": sha256(directory / "exports.json"),
                "fit_sha256": sha256(directory / "fit.json"),
                "metrics_sha256": sha256(directory / "metrics.json"),
                "model_id": model_id,
                "model_sha256": fit["model_sha256"],
                "primary_learned_model": primary,
                "run_config_sha256": sha256(directory / "run_config.json"),
                "run_id": run_id,
                "sample_manifest_sha256": run_config["sample_manifest_sha256"],
                "training_rows": fit["training_rows"],
            }
        )
        calibration = metrics["aggregates"]["calibration"]
        test = metrics["aggregates"]["test"]
        rows.append(
            {
                "run_id": run_id,
                "model_id": model_id,
                "role": "primary" if primary else "descriptive_appendix",
                "cap": cap,
                "fit_status": fit["status"],
                "training_rows": fit["training_rows"],
                "calibration_mean_device_macro_f1": calibration["mean_device_macro_f1"],
                "test_mean_device_macro_f1": test["mean_device_macro_f1"],
                "test_median_device_macro_f1": test["median_device_macro_f1"],
                "test_q25": test["q25_device_macro_f1"],
                "test_q75": test["q75_device_macro_f1"],
                "test_iqr": test["iqr_device_macro_f1"],
                "test_min": test["min_device_macro_f1"],
                "test_max": test["max_device_macro_f1"],
            }
        )
    primary_ids = [model["model_id"] for model in models if model["primary_learned_model"]]
    if primary_ids != ["LR", "ET", "HGB", "RF", "MLP"]:
        raise ValueError("Accepted primary learned-model family changed")
    registry = {
        "class_order": list(CLASSES),
        "confirmatory_model_family": ["LR", "ET", "HGB"],
        "evaluation_rows": {"calibration": 6_620_817, "test": 7_912_338},
        "models": models,
        "primary_metric": "equal-device mean per-device macro-F1",
        "status": "passed",
    }
    return registry, rows


def main() -> None:
    root = Path.cwd()
    baseline = root / "artifacts/experiments/baselines"
    registry, rows = assemble(root)
    registry_path = baseline / "accepted_model_registry.json"
    summary_path = baseline / "accepted_point_metric_summary.csv"
    write_json(registry_path, registry)
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "models": len(registry["models"]),
                "registry_sha256": sha256(registry_path),
                "status": registry["status"],
                "summary_sha256": sha256(summary_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
