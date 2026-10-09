import json
from pathlib import Path

from test_kaggle_rerun import _make_bundle

from iot_diad.modeling.baselines import CAP, CLASSES, SEED, sha256, write_json
from iot_diad.modeling.kaggle_rerun import read_json, validate_bundle
from iot_diad.modeling.lr_solver_rerun import (
    AMENDMENT_RELATIVE,
    AMENDMENT_REPORT_RELATIVE,
    SOURCE_RELATIVE,
    amended_lr,
    run,
    verify,
)


def _make_repo(root: Path, bundle: Path) -> Path:
    repo = root / "repo"
    report = repo / AMENDMENT_REPORT_RELATIVE
    report.parent.mkdir(parents=True)
    report.write_text("synthetic predeclared LR solver amendment\n")
    source = repo / SOURCE_RELATIVE
    source.parent.mkdir(parents=True)
    source.write_text(Path(SOURCE_RELATIVE).read_text())
    _, prepared, _ = validate_bundle(bundle)
    amendment = {
        "amendment_id": "synthetic-lr-solver-test",
        "amendment_report_sha256": sha256(report),
        "base_bundle_manifest_sha256": sha256(bundle / "kaggle_bundle_manifest.json"),
        "class_order": list(CLASSES),
        "expected_feature_contract_sha256": prepared["feature_contract_sha256"],
        "expected_rows_by_partition": prepared["rows_by_partition"],
        "expected_sample_manifest_sha256": prepared["sample_manifest_sha256"],
        "expected_training_rows": prepared["training_rows"],
        "max_iter": 200,
        "minimum_output_free_bytes": 2_684_354_560,
        "model_id": "LR",
        "run_id": "LR-amended-newton-cholesky-maxiter200",
        "seed": SEED,
        "solver": "newton-cholesky",
        "training_cap_per_device_class": CAP,
    }
    write_json(repo / AMENDMENT_RELATIVE, amendment)
    return repo


def test_solver_amendment_changes_only_solver_and_limit(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    _, _, config = validate_bundle(bundle)
    amendment = {
        "solver": "newton-cholesky",
        "max_iter": 200,
    }
    _, params = amended_lr(amendment, config)
    differences = {key for key, value in params.items() if value != config["estimators"]["LR"][key]}
    assert differences == {"max_iter", "solver"}
    assert params["solver"] == "newton-cholesky"
    assert params["max_iter"] == 200


def test_solver_rerun_passes_training_gate_exports_and_resumes(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    repo = _make_repo(tmp_path, bundle)
    output = tmp_path / "output"
    first = run(repo, bundle, output)
    assert first["acceptance_status"] == "passed"
    fit = read_json(output / "LR-amended-newton-cholesky-maxiter200/fit.json")
    assert fit["status"] == "passed"
    assert all(fit["acceptance_checks"].values())
    assert fit["iterations"] and max(fit["iterations"]) < 200
    digest = sha256(output / "LR-amended-newton-cholesky-maxiter200/probabilities/00000.parquet")
    assert run(repo, bundle, output) == first
    assert digest == sha256(
        output / "LR-amended-newton-cholesky-maxiter200/probabilities/00000.parquet"
    )
    _, prepared, config = validate_bundle(bundle)
    amendment = json.loads((repo / AMENDMENT_RELATIVE).read_text())
    metrics = output / "LR-amended-newton-cholesky-maxiter200/metrics.json"
    original = metrics.read_text()
    metrics.write_text('{"corrupt": true}\n')
    try:
        verify(repo, bundle, output, amendment, prepared, config)
    except ValueError as error:
        assert "point metrics mismatch" in str(error)
    else:
        raise AssertionError("corrupt point metrics were accepted")
    metrics.write_text(original)


def test_solver_amendment_rejects_changed_bundle_hash(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    repo = _make_repo(tmp_path, bundle)
    amendment_path = repo / AMENDMENT_RELATIVE
    amendment = json.loads(amendment_path.read_text())
    amendment["base_bundle_manifest_sha256"] = "0" * 64
    write_json(amendment_path, amendment)
    try:
        run(repo, bundle, tmp_path / "output")
    except ValueError as error:
        assert "base_bundle_manifest_sha256" in str(error)
    else:
        raise AssertionError("changed bundle hash was accepted")
