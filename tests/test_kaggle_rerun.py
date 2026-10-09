import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import polars as pl

from iot_diad.modeling.baselines import (
    CLASSES,
    CONTRACT_PATHS,
    estimators,
    object_hash,
    sha256,
    write_json,
)
from iot_diad.modeling.data import METADATA_COLUMNS
from iot_diad.modeling.kaggle_rerun import (
    AMENDMENT_RELATIVE,
    BASELINE_RELATIVE,
    amended_estimator,
    run,
    validate_bundle,
    verify_outputs,
)


def _versions() -> dict[str, str]:
    names = ("joblib", "numpy", "polars", "scikit-learn", "scipy", "threadpoolctl")
    return {name: importlib.metadata.version(name) for name in names}


def _write_contracts(root: Path) -> dict[str, str]:
    roles = root / CONTRACT_PATHS[0]
    roles.parent.mkdir(parents=True)
    roles.write_text(
        "feature_view,column,role,primary_baseline_action\n"
        "packet,x,candidate_feature,include_numeric\n"
    )
    for relative in CONTRACT_PATHS[1:4]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic {relative}\n")
    protocol = root / CONTRACT_PATHS[4]
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_text("synthetic amended Gate 3 protocol\n")
    return {relative: sha256(root / relative) for relative in CONTRACT_PATHS}


def _metadata(partition: str, rows: int, offset: int) -> dict[str, list[object]]:
    return {
        "_partition": [partition] * rows,
        "_canonical_device_id": [f"{partition}-d{i % 2}" for i in range(rows)],
        "_local_identifier": [f"{partition}-m{i % 2}" for i in range(rows)],
        "_capture_id": ["synthetic"] * rows,
        "_row_index": list(range(offset, offset + rows)),
        "_label": [CLASSES[i % len(CLASSES)] for i in range(rows)],
    }


def _make_bundle(root: Path) -> Path:
    bundle = root / "bundle"
    baseline = bundle / BASELINE_RELATIVE
    baseline.mkdir(parents=True)
    contracts = _write_contracts(bundle)
    train_rows = 160
    train = pl.DataFrame(
        {
            **_metadata("train", train_rows, 0),
            "x": np.arange(train_rows, dtype=np.float64),
            "_priority": np.arange(train_rows, dtype=np.uint64),
        }
    )
    train.write_parquet(baseline / "training_sample.parquet")
    train.select(*METADATA_COLUMNS, "_priority").write_csv(
        baseline / "training_sample_manifest.csv"
    )
    evaluation = []
    for index, partition in enumerate(("calibration", "test")):
        frame = pl.DataFrame(
            {
                **_metadata(partition, 16, train_rows + index * 16),
                "x": np.arange(index * 16, (index + 1) * 16, dtype=np.float64),
            }
        )
        path = baseline / "cache/000" / f"eval-{index:05d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
        evaluation.append(
            {"path": str(path.relative_to(baseline)), "rows": frame.height, "sha256": sha256(path)}
        )
    features = {"numeric": ["x"], "categorical": []}
    config = {
        "cap": 2000,
        "class_order": list(CLASSES),
        "contracts": contracts,
        "estimators": {
            name: json.loads(json.dumps(estimators()[name].get_params())) for name in ("LR", "MLP")
        },
        "features": features,
        "seed": 2027,
    }
    write_json(baseline / "config.json", config)
    prepared = {
        "config_sha256": sha256(baseline / "config.json"),
        "evaluation_chunks": evaluation,
        "feature_contract_sha256": object_hash(features),
        "rows_by_partition": {"calibration": 16, "test": 16, "train": train_rows},
        "sample_manifest_sha256": sha256(baseline / "training_sample_manifest.csv"),
        "training_rows": train_rows,
        "training_sample_sha256": sha256(baseline / "training_sample.parquet"),
    }
    write_json(baseline / "prepared.json", prepared)
    amendment = {
        "amendment_id": "synthetic-test",
        "amended_gate3_protocol_sha256": contracts[CONTRACT_PATHS[4]],
        "class_order": list(CLASSES),
        "expected_feature_contract_sha256": prepared["feature_contract_sha256"],
        "expected_rows_by_partition": prepared["rows_by_partition"],
        "expected_sample_manifest_sha256": prepared["sample_manifest_sha256"],
        "expected_training_rows": train_rows,
        "models": {
            "LR": {
                "amended_max_iter": 1000,
                "original_max_iter": 300,
                "run_id": "LR-amended-maxiter1000",
            },
            "MLP": {
                "amended_max_iter": 200,
                "original_max_iter": 50,
                "run_id": "MLP-amended-maxiter200",
            },
        },
        "minimum_output_free_bytes": 5 * 1024**3,
        "original_gate3_protocol_sha256": contracts[CONTRACT_PATHS[4]],
        "required_package_versions": _versions(),
        "required_python_major_minor": ".".join(platform.python_version_tuple()[:2]),
        "seed": 2027,
        "training_cap_per_device_class": 2000,
    }
    amendment_path = bundle / AMENDMENT_RELATIVE
    amendment_path.parent.mkdir(parents=True)
    write_json(amendment_path, amendment)
    requirements = bundle / "config/requirements-kaggle-base002.txt"
    requirements.write_text("\n".join(f"{k}=={v}" for k, v in _versions().items()) + "\n")
    source = bundle / "src/iot_diad/modeling/preprocessing.py"
    source.parent.mkdir(parents=True)
    source.write_text("# synthetic source hash fixture\n")
    files = []
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name != "kaggle_bundle_manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(bundle)),
                    "sha256": sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    write_json(bundle / "kaggle_bundle_manifest.json", {"files": files})
    return bundle


def test_amendment_changes_only_max_iter() -> None:
    config = {
        "estimators": {
            name: json.loads(json.dumps(estimators()[name].get_params())) for name in ("LR", "MLP")
        }
    }
    amendment = {"models": {"LR": {"amended_max_iter": 1000}, "MLP": {"amended_max_iter": 200}}}
    for name, limit in (("LR", 1000), ("MLP", 200)):
        _, params = amended_estimator(name, amendment, config)
        differences = {
            key for key, value in params.items() if value != config["estimators"][name][key]
        }
        assert differences == {"max_iter"}
        assert params["max_iter"] == limit


def test_synthetic_rerun_is_complete_resumable_and_self_verifying(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    output = tmp_path / "output"
    first = run(bundle, output)
    assert first["verification_status"] == "passed"
    assert set(first["fit_statuses"]) == {"LR", "MLP"}
    probability = output / "LR-amended-maxiter1000/probabilities/00000.parquet"
    digest = hashlib.sha256(probability.read_bytes()).hexdigest()
    second = run(bundle, output)
    assert second == first
    assert hashlib.sha256(probability.read_bytes()).hexdigest() == digest
    verification = json.loads((output / "rerun_verification.json").read_text())
    assert verification["rows_by_partition"] == {"calibration": 16, "test": 16}
    for run_id in ("LR-amended-maxiter1000", "MLP-amended-maxiter200"):
        metrics = json.loads((output / run_id / "metrics.json").read_text())
        assert metrics["aggregates"]["test"]["device_count"] == 2

    amendment, prepared, config = validate_bundle(bundle)
    metrics_path = output / "LR-amended-maxiter1000/metrics.json"
    metrics = metrics_path.read_text()
    metrics_path.write_text('{"corrupt": true}\n')
    try:
        verify_outputs(bundle, output, amendment, prepared, config)
    except ValueError as error:
        assert "point metrics mismatch" in str(error)
    else:
        raise AssertionError("corrupt metrics were accepted")
    metrics_path.write_text(metrics)

    fit_path = output / "LR-amended-maxiter1000/fit.json"
    fit = json.loads(fit_path.read_text())
    fit["status"] = "passed"
    fit["warnings"] = [{"category": "ConvergenceWarning", "message": "synthetic"}]
    write_json(fit_path, fit)
    try:
        verify_outputs(bundle, output, amendment, prepared, config)
    except ValueError as error:
        assert "fit status does not match its warnings" in str(error)
    else:
        raise AssertionError("inconsistent convergence status was accepted")


def test_bundle_corruption_is_rejected(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    target = bundle / BASELINE_RELATIVE / "cache/000/eval-00000.parquet"
    target.write_bytes(target.read_bytes() + b"corrupt")
    try:
        run(bundle, tmp_path / "output")
    except ValueError as error:
        assert "Bundle file integrity mismatch" in str(error)
    else:
        raise AssertionError("corrupt bundle was accepted")
