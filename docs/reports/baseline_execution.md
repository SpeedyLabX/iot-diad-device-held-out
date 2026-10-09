# BASE-002 execution and audit contract

Gate 3 was explicitly approved on 2026-10-05. The scientific settings remain frozen in
`docs/reports/gate3_protocol.md`. This report describes implementation; outcomes and runtime
are recorded in the BASE-002 handoff and machine artifacts.

## Deterministic training sample

For each eligible training row, seed 2027 and its capture identifier define a SHA256-derived
64-bit salt. A SplitMix64 permutation of the salt XOR zero-based raw row index gives a priority.
The 2,000 smallest priorities per physical-device/coarse-class cell are retained, with capture
and raw row index breaking ties. Cells with fewer rows retain every row. The algorithm is
independent of chunk boundaries and processing order. The manifest is sorted by raw row key.
The required 1,000-row-cap sensitivity is the nested bottom-1,000 subset for the model with the
highest primary point metric (model-ID tie break). This selection is descriptive and does not
change any primary model or configuration.

## Streaming and resume

Raw packet CSVs are read in at most 50,000-row batches. The loader applies approved identities,
physical-device partitions, and all 58 Gate 2 row exclusions using absolute raw offsets. Each
capture completion record includes its raw SHA256, counts, per-device/class counts, candidate
sample hash, and evaluation-cache chunk hashes. Incomplete captures can be replayed. Completed
captures are reused; the independent input audit checks all raw contents before acceptance.

Every model fits its own preprocessing on the same selected training rows. No calibration/test
value enters preprocessing, fitting, validation-based early stopping, or model configuration.
MLP's internal early stopping uses only the training sample. Numeric non-finite values become
missing before training-derived imputation; unknown categories are handled by training-fitted
encoding. ET/RF keep `n_jobs=-1`; numerical thread pools are capped at eight for resource control.

Because the five primary fits use the identical training manifest and deterministic preprocessing,
evaluation transforms are cached as finite float64 arrays and shared only when a restart-stable
contract matches. That contract includes the training-manifest and feature-contract hashes, the
fitted preprocessor audit, preprocessing source hash, and software versions. Each cached array has
an atomic receipt containing its source-chunk hash, shape, dtype, output hash, and cache key.
The 1,000-row sensitivity fit necessarily has a different manifest and therefore a separate cache.

The runner serializes each trained model before evaluation. Parquet outputs and completion
receipts are written atomically. Resume checks model, sample, input, output, configuration,
software-version, and execution-source hashes; it refuses inconsistent completed work. A process
lock prevents concurrent writers. Interrupted model fits must restart; completed fits and export
chunks are reused. Exceptions and convergence warnings remain explicit; no configuration is
silently changed. Convergence-warning exports, if any, are diagnostic pending human review.

Two development attempts are preserved under `artifacts/experiments/baselines/attempts/`: the
first motivated the exact shared-transform cache, and the second exposed that a library object
hash changed after serialization. Neither attempt is accepted as a primary result. The final cache
uses the deterministic contract above; no model setting, row, or transformed value changed.

## Probability artifacts and metrics

Each float64 probability row carries the eight frozen class columns in order, argmax label,
true coarse label, physical device, local identifier, capture, partition, zero-based raw row
index, model/run ID, seed, feature-contract and sample-manifest hashes, and software versions.
Class order: Benign; Brute Force; DDoS; DoS; Mirai; Recon; Spoofing; Web-based.

Each run exports per-device confusion matrices, present-true-class macro-F1, accuracy and
per-class support/recall/F1. Missing true classes are null for class-specific calculations.
Partition summaries contain equal-device mean, median, quartiles, IQR and range of macro-F1,
device-balanced class recall, global row-weighted metrics, and leave-one-device-out means.
Calibration point metrics are descriptive diagnostics only. There are no prediction sets,
conformal thresholds, p-values, multiplicity corrections, or bootstrap intervals in BASE-002.

## Reproduction

Use the `slx07-iot` environment and run from the repository root:

```bash
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 POLARS_MAX_THREADS=8
python -m iot_diad.modeling.baselines prepare
python -m iot_diad.modeling.baselines run
python -m iot_diad.modeling.baselines sensitivity
python scripts/verify_baseline_inputs.py
python scripts/verify_baselines.py
pytest
ruff check src tests scripts
mypy src/iot_diad scripts/verify_baselines.py scripts/verify_baseline_inputs.py
iot-diad-state
```

`run --model LR` (or ET/HGB/RF/MLP/DUMMY) resumes an individual run. DUMMY is an appendix-only
pipeline sanity check. Verification independently reconstructs the sample using scalar integer
priority arithmetic, joins raw rows to the approved split, compares every cached evaluation
feature and row key to raw CSVs, and recomputes metrics from all probability exports using
scikit-learn's metric routines. Artifact integrity and estimator convergence are separate checks.
