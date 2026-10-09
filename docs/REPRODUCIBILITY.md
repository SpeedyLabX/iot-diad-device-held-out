# Reproducibility Guide

This guide separates lightweight verification from the full-data experiment. Run commands from
the repository root with Python 3.12.

## 1. Install the environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

The direct dependency ranges are declared in `pyproject.toml`; `requirements.lock` records the
complete verified environment.

## 2. Run lightweight verification

The test suite uses synthetic fixtures and does not require the dataset:

```bash
ruff format --check src tests scripts
ruff check src tests scripts
mypy src
bandit -q -r src scripts -ll
pytest
```

Compact published artifacts can be inspected without regenerating the row-level outputs. Exact
full-artifact verifiers require the locally regenerated files described below.

## 3. Prepare the data

Download the official release and place it under
`data/raw/CIC-IoT-IDAD-Dataset-2024/`. The complete extracted dataset is approximately 51 GB.
Confirm the two expected feature-view directories listed in [`data/README.md`](../data/README.md).

Before a full scan, run a four-file smoke test:

```bash
iot-diad-audit --stage all --limit-files 4 \
  --output-dir /tmp/iot-diad-audit \
  --report-path /tmp/iot-diad-data-summary.md

iot-diad-eda --stage all --limit-files 4 \
  --output-dir /tmp/iot-diad-eda \
  --sample-dir /tmp/iot-diad-samples \
  --report-path /tmp/iot-diad-eda-report.md
```

## 4. Audit and partition the dataset

```bash
iot-diad-audit --stage all
iot-diad-eda --stage all
python -m iot_diad.device_mapping
python scripts/reconcile_published_devices.py
python scripts/resolve_published_aliases.py
python scripts/verify_device_mapping.py
python -m iot_diad.splitting
python scripts/verify_device_split.py
python -m iot_diad.leakage
python scripts/verify_leakage_audit.py
python scripts/verify_cross_partition_duplicates.py
```

The repository includes the approved compact mapping and split contracts. Re-running mapping
creates larger source-row evidence files that remain local.

## 5. Train and evaluate baselines

The accepted workflow uses deterministic seed 2027, caps training at 2,000 rows per
device/class cell, and streams complete calibration and test partitions. Expect tens of gigabytes
of temporary and row-level outputs.

```bash
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export POLARS_MAX_THREADS=8

python -m iot_diad.modeling.baselines prepare
python -m iot_diad.modeling.baselines run
python -m iot_diad.modeling.baselines sensitivity
python scripts/verify_baseline_inputs.py
python scripts/verify_baselines.py
```

Completed model runs resume from hash-bound receipts. A changed contract, source file, sample,
model, or software version is rejected instead of silently reusing incompatible output.

## 6. Reproduce conformal and statistical analyses

```bash
POLARS_MAX_THREADS=4 python -m iot_diad.modeling.conformal --root .
POLARS_MAX_THREADS=4 python scripts/verify_conformal.py --root .
python scripts/summarize_conformal.py --root .
python scripts/summarize_conformal.py --root . --verify-only

python -m iot_diad.modeling.statistics --root .
python scripts/verify_statistics.py --root .
```

The conformal stage requires accepted baseline probability exports. The statistical stage uses
paired physical-device outcomes and resamples complete devices, not packet rows.

## 7. Interpret the outputs

The primary public summaries are:

- `artifacts/experiments/baselines/accepted_point_metric_summary.csv`;
- `artifacts/experiments/conformal/model_summary.csv`;
- `artifacts/experiments/conformal/class_coverage.csv`;
- `artifacts/experiments/statistics/confirmatory_comparisons.csv`;
- `artifacts/experiments/statistics/bootstrap_summary.csv`.

Interpret these together with the corresponding reports in `docs/reports/`. The benchmark uses
physical-device-disjoint but capture-shared partitions, and the 14 test devices—not individual
packet rows—form the inferential sample.

## Determinism and expected differences

Data selection, splitting, sampling, and bootstrap indices are deterministic. Runtime and logs
may vary by platform. Exact model outputs require the recorded Python and package versions;
different BLAS implementations or dependency versions may change floating-point results.
