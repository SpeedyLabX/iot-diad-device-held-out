"""Train-only preprocessing for mixed packet features."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from iot_diad.modeling.features import FeatureSpec


def finite_to_nan(values: Any) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64).copy()
    array[~np.isfinite(array)] = np.nan
    return array


def empty_to_nan(values: Any) -> NDArray[np.object_]:
    array = np.asarray(values, dtype=object).copy()
    for index in np.ndindex(array.shape):
        value = array[index]
        if value is None or value == "":
            array[index] = np.nan
    return array


def device_set_hash(device_ids: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(device_ids)).encode()).hexdigest()


class TrainOnlyPreprocessor:
    def __init__(self, spec: FeatureSpec):
        spec.validate()
        self.spec = spec
        self.transformer = self._build_transformer()
        self.training_devices_: frozenset[str] | None = None

    def _build_transformer(self) -> ColumnTransformer:
        numeric_indices = list(range(len(self.spec.numeric)))
        categorical_indices = list(range(len(self.spec.numeric), len(self.spec.columns)))
        numeric = Pipeline(
            [
                (
                    "finite_to_nan",
                    FunctionTransformer(finite_to_nan, feature_names_out="one-to-one"),
                ),
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("remove_constant", VarianceThreshold(threshold=0.0)),
                ("scale", StandardScaler()),
            ]
        )
        transformers: list[tuple[str, Any, list[int]]] = [("numeric", numeric, numeric_indices)]
        if categorical_indices:
            categorical = Pipeline(
                [
                    (
                        "empty_to_nan",
                        FunctionTransformer(empty_to_nan, feature_names_out="one-to-one"),
                    ),
                    (
                        "impute",
                        SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                    ),
                    (
                        "encode",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )
            transformers.append(("categorical", categorical, categorical_indices))
        return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)

    def fit(self, frame: pl.DataFrame) -> TrainOnlyPreprocessor:
        self._validate_frame(frame)
        partitions = set(frame["_partition"].to_list())
        if partitions != {"train"}:
            raise ValueError(
                f"Preprocessing fit requires only training rows, received {partitions}"
            )
        devices = set(frame["_canonical_device_id"].to_list())
        if not devices:
            raise ValueError("Preprocessing fit requires at least one training device")
        self.transformer.fit(frame.select(self.spec.columns).to_numpy())
        self.training_devices_ = frozenset(devices)
        return self

    def transform(self, frame: pl.DataFrame) -> NDArray[np.float64]:
        self._validate_frame(frame)
        if self.training_devices_ is None:
            raise RuntimeError("Preprocessor must be fit before transform")
        transformed = self.transformer.transform(frame.select(self.spec.columns).to_numpy())
        result = np.asarray(transformed, dtype=np.float64)
        if not np.isfinite(result).all():
            raise ValueError("Preprocessing produced non-finite model inputs")
        return result

    def fit_transform(self, frame: pl.DataFrame) -> NDArray[np.float64]:
        return self.fit(frame).transform(frame)

    def audit(self) -> dict[str, str | int]:
        if self.training_devices_ is None:
            raise RuntimeError("Preprocessor must be fit before audit")
        return {
            "fit_partition": "train",
            "training_device_count": len(self.training_devices_),
            "training_device_set_sha256": device_set_hash(set(self.training_devices_)),
            "input_numeric_features": len(self.spec.numeric),
            "input_categorical_features": len(self.spec.categorical),
            "output_features": len(self.transformer.get_feature_names_out()),
        }

    def _validate_frame(self, frame: pl.DataFrame) -> None:
        required = {"_partition", "_canonical_device_id", *self.spec.columns}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Modeling frame is missing required columns: {sorted(missing)}")
