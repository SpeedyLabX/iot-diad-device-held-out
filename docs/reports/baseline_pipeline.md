# Train-only baseline pipeline

**Task:** BASE-001  
**Date:** 2026-10-05 (Asia/Ho_Chi_Minh)  
**Status:** implemented and verified; classifier protocol remains at Human Gate 3

## Data contract

The modeling loader accepts only the 68 Gate 1-approved packet identifiers, attaches their frozen
train/calibration/test device assignment, and removes the 58 Gate 2-approved cross-partition
duplicate rows by `(capture_id, zero-based raw row index)`. Unknown and unresolved identifiers do
not enter the modeling frame.

Every output row retains partition, canonical physical-device ID, local identifier, capture, raw
row index, and coarse label as metadata. These fields are never passed to the estimator.

## Feature contract

The audited column-role artifact selects 116 numeric and four categorical packet features. Group
identifiers, session identifiers, source labels, and seven high-risk metadata fields remain
excluded. The categorical candidates are `handshake_version`, `http_request_method`,
`highest_layer`, and `http_content_type`.

## Train-only transforms

`TrainOnlyPreprocessor.fit` rejects any frame containing calibration or test rows. Using training
devices only, it:

1. converts numeric non-finite values to missing;
2. learns numeric medians;
3. removes zero-variance numeric columns;
4. learns standard-scaling parameters;
5. learns most-frequent categorical imputations; and
6. learns one-hot categories with unknown-category handling.

The preprocessor records the training-device count and a deterministic hash of the training-device
set. Transform output is rejected if any non-finite value remains.

`ProbabilityPipeline` clones a supplied probabilistic classifier, label-encodes training labels,
and exports ordered class probabilities with the row-level metadata needed for conformal
calibration and device-level paired inference. The classifier choice is intentionally not frozen
inside BASE-001; Human Gate 3 controls the compact model matrix and compute budget.

## Verification

Focused tests cover:

- non-finite and missing numeric values;
- training-derived imputation and scaling;
- constant-column removal;
- missing and previously unseen categorical values;
- rejection of non-training fit data;
- exclusion of identifiers and high-risk metadata;
- filtering of unresolved identifiers and approved duplicate rows; and
- probability column order, metadata preservation, finiteness, and normalization.

The raw-data smoke test reads the first 20,000 rows of a deterministic packet capture, selects
18,759 eligible rows and 11,564 training rows, fits on 36 training devices, and transforms 1,000
mixed-partition rows. It starts from 120 audited features and produces 139 finite columns after
training-only constant removal and categorical expansion. Exact evidence is in
`artifacts/modeling/preprocessing_smoke.json`.

## Reproduction

```bash
python scripts/smoke_modeling_pipeline.py
pytest tests/test_modeling.py
```

The environment pins scikit-learn 1.9.x in `pyproject.toml`; the resolved environment records
scikit-learn 1.9.1, SciPy 1.18.1, and joblib 1.6.0 in `requirements.lock`.
