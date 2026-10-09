from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import f1_score

from iot_diad.modeling.baselines import (
    bottom_k,
    estimators,
    guard_manifest,
    metric_rows,
    priorities,
    validate_probabilities,
)
from iot_diad.modeling.data import PartitionContract, select_modeling_rows
from iot_diad.modeling.features import FeatureSpec


def test_streaming_selection_matches_full_selection_and_preserves_small_cells() -> None:
    frame = pl.DataFrame(
        {
            "_canonical_device_id": ["d1"] * 40 + ["d2"] * 3,
            "_label": ["Benign"] * 20 + ["DDoS"] * 20 + ["Benign"] * 3,
            "_capture_id": ["cap"] * 43,
            "_row_index": list(range(43)),
            "_priority": priorities("cap", np.arange(43, dtype=np.uint64)),
        }
    )
    frame = frame.select("_row_index", pl.exclude("_row_index"))
    full = bottom_k(frame, 7)
    assert full.columns == frame.columns
    chunks = [bottom_k(c, 7) for c in frame.iter_slices(6)]
    streamed = bottom_k(pl.concat(chunks).reverse(), 7)
    assert full.equals(streamed)
    assert full.filter(pl.col("_canonical_device_id") == "d2").height == 3
    assert full.height == 17
    assert bottom_k(full, 4).equals(bottom_k(frame, 4))
    np.testing.assert_array_equal(
        priorities("cap", np.arange(43, dtype=np.uint64)), frame["_priority"].to_numpy()
    )


def test_chunk_offsets_apply_duplicate_exclusion_at_raw_position() -> None:
    contract = PartitionContract({"a": "d1"}, {"a": "test"}, {"cap": frozenset({4})})
    spec = FeatureSpec(("x",), ())
    raw = pl.DataFrame({"device_mac": ["a", "unknown", "a", "a", "a", "a"], "x": range(6)})
    full = select_modeling_rows(raw, "cap", "Benign", spec, contract)
    chunks = [
        select_modeling_rows(raw.slice(i, 3), "cap", "Benign", spec, contract, i) for i in (0, 3)
    ]
    assert full.equals(pl.concat(chunks))
    assert full["_row_index"].to_list() == [0, 2, 3, 5]


def test_resume_refuses_changed_contract(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    guard_manifest(path, {"seed": 2027})
    guard_manifest(path, {"seed": 2027})
    with pytest.raises(ValueError, match="Resume contract changed"):
        guard_manifest(path, {"seed": 3})


def test_metrics_present_true_classes_and_missing_recall() -> None:
    truth = np.array([0, 0, 1, 1, 1])
    predicted = np.array([0, 2, 0, 1, 2])
    matrix = np.zeros((8, 8), dtype=np.int64)
    np.add.at(matrix, (truth, predicted), 1)
    result = metric_rows(matrix)
    assert result["macro_f1"] == pytest.approx(
        f1_score(truth, predicted, labels=[0, 1], average="macro")
    )
    assert result["per_class"]["DDoS"]["recall"] is None
    assert result["accuracy"] == 0.4


def test_invalid_probabilities_rejected() -> None:
    valid = np.full((2, 8), 0.125)
    validate_probabilities(valid, 2)
    for bad in (valid * 0.9, np.full((2, 8), np.nan), np.zeros((2, 7))):
        with pytest.raises(ValueError):
            validate_probabilities(bad, 2)
    bad = valid.copy()
    bad[0, 0], bad[0, 1] = -0.1, 0.35
    with pytest.raises(ValueError):
        validate_probabilities(bad, 2)


def test_frozen_configuration_prevents_evaluation_early_stopping() -> None:
    models = estimators()
    assert models["HGB"].early_stopping is False
    assert models["MLP"].early_stopping is True
    assert models["MLP"].validation_fraction == 0.1
    assert models["RF"].class_weight == "balanced_subsample"
    assert models["ET"].class_weight == "balanced"
    assert models["LR"].max_iter == 300
    assert all(m.random_state == 2027 for m in models.values())


def test_end_to_end_chunked_dummy_resume_and_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    import iot_diad.modeling.baselines as baseline
    from iot_diad.metadata import PACKET_DIRECTORY

    monkeypatch.setattr(baseline, "CAP", 3)
    monkeypatch.setattr(baseline, "CHUNK", 5)
    roles = tmp_path / baseline.CONTRACT_PATHS[0]
    roles.parent.mkdir(parents=True)
    roles.write_text(
        "feature_view,column,role,primary_baseline_action\n"
        "packet,x,candidate_feature,include_numeric\n"
    )
    split = tmp_path / baseline.CONTRACT_PATHS[1]
    split.parent.mkdir(parents=True)
    split.write_text(
        "local_identifier,canonical_device_id,partition\n"
        "a,d1,train\nb,d2,train\nc,d3,calibration\nd,d4,test\n"
    )
    (tmp_path / baseline.CONTRACT_PATHS[2]).write_text("capture_id,row_index\n")
    (tmp_path / baseline.CONTRACT_PATHS[3]).write_text(
        json.dumps({"filtered_rows_by_partition": {"train": 128, "calibration": 32, "test": 32}})
    )
    protocol = tmp_path / baseline.CONTRACT_PATHS[4]
    protocol.parent.mkdir(parents=True)
    protocol.write_text("test-only synthetic protocol fixture")
    directories = ("Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web-Based")
    for label in directories:
        path = (
            tmp_path / "data/raw/CIC-IoT-IDAD-Dataset-2024" / PACKET_DIRECTORY / label / "cap.csv"
        )
        path.parent.mkdir(parents=True)
        pl.DataFrame(
            {"device_mac": ["a", "b", "c", "d", "a", "b"] * 4, "x": list(range(24))}
        ).write_csv(path)
    output = tmp_path / "output"
    baseline.prepare(tmp_path, output)
    prepared = json.loads((output / "prepared.json").read_text())
    assert prepared["training_rows"] == 48
    baseline.run_model(tmp_path, output, "DUMMY", cap=3)
    first = json.loads((output / "DUMMY/exports.json").read_text())
    model_hash = baseline.sha256(output / "DUMMY/model.joblib")
    # Simulate missing last receipt after interruption; only that chunk is regenerated.
    last = output / first["chunks"][-1]["path"]
    last.with_suffix(".json").unlink()
    baseline.run_model(tmp_path, output, "DUMMY", cap=3)
    second = json.loads((output / "DUMMY/exports.json").read_text())
    assert first["chunks"] == second["chunks"]
    assert baseline.sha256(output / "DUMMY/model.joblib") == model_hash
    assert second["rows"] == 64
    last.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Resume chunk hash mismatch"):
        baseline.run_model(tmp_path, output, "DUMMY", cap=3)


def test_cached_transform_exact_reuse_and_corruption(tmp_path: Path) -> None:
    import joblib

    from iot_diad.modeling.baselines import (
        cached_evaluation_features,
        make_transform_contract,
        sha256,
    )
    from iot_diad.modeling.preprocessing import TrainOnlyPreprocessor

    frame = pl.DataFrame(
        {
            "_partition": ["train"] * 4,
            "_canonical_device_id": ["d1"] * 4,
            "_local_identifier": ["a"] * 4,
            "_capture_id": ["cap"] * 4,
            "_row_index": range(4),
            "_label": ["Benign"] * 4,
            "numeric": [1.0, 2.0, None, float("inf")],
            "category": ["a", "a", "b", None],
        }
    )
    processor = TrainOnlyPreprocessor(FeatureSpec(("numeric",), ("category",))).fit(frame)
    before = make_transform_contract(tmp_path, processor, "sample", "feature")
    serialized = tmp_path / "processor.joblib"
    joblib.dump(processor, serialized)
    restored = joblib.load(serialized)
    assert make_transform_contract(tmp_path, restored, "sample", "feature") == before
    heldout = frame.with_columns(
        pl.lit("test").alias("_partition"),
        pl.lit("d2").alias("_canonical_device_id"),
        pl.lit("unknown").alias("category"),
    )
    path = tmp_path / "source.parquet"
    heldout.write_parquet(path)
    source = {"path": path.name, "rows": 4, "sha256": sha256(path)}
    reference = processor.transform(heldout)
    _, first = cached_evaluation_features(tmp_path, source, processor, "key")
    _, second = cached_evaluation_features(tmp_path, source, processor, "key")
    np.testing.assert_array_equal(first, reference)
    np.testing.assert_array_equal(second, reference)
    cache = next((tmp_path / "transformed/key").glob("*.npy"))
    with cache.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="Transformed cache hash mismatch"):
        cached_evaluation_features(tmp_path, source, processor, "key")
