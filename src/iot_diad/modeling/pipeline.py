"""Estimator-neutral probability pipeline with auditable metadata export."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder

from iot_diad.modeling.data import METADATA_COLUMNS
from iot_diad.modeling.features import FeatureSpec
from iot_diad.modeling.preprocessing import TrainOnlyPreprocessor


class ProbabilityPipeline:
    def __init__(self, spec: FeatureSpec, estimator: Any):
        self.preprocessor = TrainOnlyPreprocessor(spec)
        self.estimator = clone(estimator)
        self.label_encoder = LabelEncoder()
        self.classes_: tuple[str, ...] | None = None

    def fit(self, train: pl.DataFrame) -> ProbabilityPipeline:
        if "_label" not in train.columns:
            raise ValueError("Training frame is missing _label")
        features = self.preprocessor.fit_transform(train)
        labels = self.label_encoder.fit_transform(train["_label"].to_numpy())
        self.estimator.fit(features, labels)
        if not hasattr(self.estimator, "predict_proba"):
            raise TypeError("Baseline estimator must implement predict_proba")
        self.classes_ = tuple(str(value) for value in self.label_encoder.classes_)
        return self

    def predict_proba(self, frame: pl.DataFrame) -> NDArray[np.float64]:
        if self.classes_ is None:
            raise RuntimeError("Probability pipeline must be fit before prediction")
        probabilities = np.asarray(
            self.estimator.predict_proba(self.preprocessor.transform(frame)), dtype=np.float64
        )
        if probabilities.shape != (frame.height, len(self.classes_)):
            raise ValueError("Estimator probability shape does not match rows and fitted classes")
        if not np.isfinite(probabilities).all():
            raise ValueError("Estimator produced non-finite probabilities")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("Estimator probabilities do not sum to one")
        return probabilities

    def probability_frame(self, frame: pl.DataFrame) -> pl.DataFrame:
        missing = set(METADATA_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"Probability export is missing metadata: {sorted(missing)}")
        probabilities = self.predict_proba(frame)
        probability_columns = {
            f"prob__{label}": probabilities[:, index]
            for index, label in enumerate(self.classes_ or ())
        }
        return frame.select(METADATA_COLUMNS).with_columns(**probability_columns)
