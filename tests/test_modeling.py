from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression

from iot_diad.modeling.data import load_partition_contract, select_modeling_rows
from iot_diad.modeling.features import FeatureSpec, load_packet_feature_spec
from iot_diad.modeling.pipeline import ProbabilityPipeline
from iot_diad.modeling.preprocessing import TrainOnlyPreprocessor


def modeling_frame(partition: str = "train") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "_partition": [partition] * 6,
            "_canonical_device_id": ["d1"] * 3 + ["d2"] * 3,
            "_local_identifier": ["a"] * 3 + ["b"] * 3,
            "_capture_id": ["capture"] * 6,
            "_row_index": list(range(6)),
            "_label": ["Benign", "Attack", "Benign", "Attack", "Benign", "Attack"],
            "numeric": [1.0, 2.0, np.inf, 4.0, None, 6.0],
            "constant": [7.0] * 6,
            "category": ["x", "x", None, "y", "y", "x"],
        }
    )


def test_train_only_preprocessor_handles_nonfinite_missing_constant_and_unknown() -> None:
    spec = FeatureSpec(numeric=("numeric", "constant"), categorical=("category",))
    train = modeling_frame()
    processor = TrainOnlyPreprocessor(spec).fit(train)
    transformed = processor.transform(
        pl.DataFrame(
            {
                "_partition": ["calibration"],
                "_canonical_device_id": ["d3"],
                "numeric": [np.nan],
                "constant": [1000.0],
                "category": ["unseen"],
            }
        )
    )

    assert transformed.shape[0] == 1
    assert np.isfinite(transformed).all()
    audit = processor.audit()
    assert audit["fit_partition"] == "train"
    assert audit["training_device_count"] == 2
    assert audit["output_features"] == 3


def test_preprocessor_rejects_nontraining_fit() -> None:
    spec = FeatureSpec(numeric=("numeric",), categorical=())
    frame = modeling_frame().with_columns(pl.lit("calibration").alias("_partition"))
    with pytest.raises(ValueError, match="only training rows"):
        TrainOnlyPreprocessor(spec).fit(frame)


def test_feature_contract_excludes_identifier_and_high_risk_roles(tmp_path: Path) -> None:
    path = tmp_path / "roles.csv"
    path.write_text(
        "feature_view,column,role,primary_baseline_action\n"
        "packet,safe,candidate_feature,include_numeric\n"
        "packet,category,candidate_feature,encode_after_validation\n"
        "packet,device_mac,group_identifier,exclude\n"
        "packet,http_host,high_risk_metadata,exclude\n",
        encoding="utf-8",
    )
    spec = load_packet_feature_spec(path)
    assert spec.numeric == ("safe",)
    assert spec.categorical == ("category",)
    assert "device_mac" not in spec.columns
    assert "http_host" not in spec.columns


def test_partition_contract_filters_unknown_and_approved_duplicate_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "local_identifier,canonical_device_id,partition\n"
        "known,T1-01-left,train\n"
        "heldout,T1-02-left,test\n"
        "calibration-device,T1-03-left,calibration\n",
        encoding="utf-8",
    )
    exclusions = tmp_path / "exclusions.csv"
    exclusions.write_text(
        "capture_id,row_index\ncap,1\n",
        encoding="utf-8",
    )
    contract = load_partition_contract(manifest, exclusions)
    frame = pl.DataFrame(
        {
            "device_mac": ["known", "known", "unknown", "heldout"],
            "feature": [1.0, 2.0, 3.0, 4.0],
        }
    )
    selected = select_modeling_rows(
        frame,
        "cap",
        "Benign",
        FeatureSpec(numeric=("feature",), categorical=()),
        contract,
    )
    assert selected["_local_identifier"].to_list() == ["known", "heldout"]
    assert selected["_partition"].to_list() == ["train", "test"]
    assert selected["_row_index"].to_list() == [0, 3]


def test_probability_pipeline_exports_required_metadata_and_class_order() -> None:
    frame = modeling_frame()
    pipeline = ProbabilityPipeline(
        FeatureSpec(numeric=("numeric",), categorical=("category",)),
        LogisticRegression(max_iter=200, random_state=2027),
    ).fit(frame)
    output = pipeline.probability_frame(frame)

    assert output.columns[:6] == [
        "_partition",
        "_canonical_device_id",
        "_local_identifier",
        "_capture_id",
        "_row_index",
        "_label",
    ]
    assert output.columns[6:] == ["prob__Attack", "prob__Benign"]
    assert np.allclose(output.select(output.columns[6:]).to_numpy().sum(axis=1), 1.0)
