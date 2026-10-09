# Device-Held-Out Evaluation for CIC IoT-DIAD 2024

Reproducible data auditing, physical-device-held-out evaluation, conformal prediction, and
device-level statistical inference for the CIC IoT-DIAD 2024 intrusion-detection dataset.

## Overview

This repository evaluates packet-based IoT intrusion detectors on physical devices that are not
used for model training. It provides the code and compact evidence needed to audit the dataset,
construct deterministic device partitions, detect leakage, train baseline classifiers, calibrate
class-conditional prediction sets, and compare models at the device level.

The evaluation has four important boundaries:

- preprocessing is fitted only on training devices;
- training, calibration, and test partitions contain disjoint physical devices;
- the accepted packet split is capture-shared, so conclusions are limited to the benchmark's
  observed capture conditions;
- the Flow feature view is not used for unseen-physical-device claims because an authoritative
  Flow-IP-to-device mapping is not available.

Raw traffic, licensed papers, model binaries, row-level probability exports, and pre-release
manuscript material are not distributed in this repository.

## Repository structure

```text
artifacts/   Compact manifests, summaries, figures, and verification records
config/      Auditing, split, and experiment contracts
data/        Expected local dataset layout
docs/        Reproducibility guide and evidence reports
notebooks/   Artifact-based exploratory analysis and rerun notebooks
scripts/     Data preparation, verification, and artifact inventory tools
src/         Installable Python package
tests/       Unit and integration tests
```

See [the artifact policy](artifacts/README.md) for the distinction between published evidence and
large outputs that must be regenerated locally.

## Quick start

Python 3.12 is the reference runtime.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest
```

Conda can be used instead:

```bash
conda create -y -n iot-diad python=3.12 pip
conda activate iot-diad
python -m pip install -e '.[dev]'
```

`requirements.lock` records the complete package set used for the verified study environment.

## Dataset

The dataset is not redistributed. Download CIC IoT-DIAD 2024 from the
[Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/iot-diad-2024.html), then
place the extracted release at:

```text
data/raw/CIC-IoT-IDAD-Dataset-2024/
├── Anomaly Detection - Flow Based features/
└── Device Identification_Anomaly Detection - Packet Based Features/
```

The upstream download path uses `IDAD`; the dataset and paper use `DIAD`. The optional downloader
preserves the upstream directory structure and resumes partial downloads:

```bash
python scripts/download_cic_iot_diad.py
```

## Reproducing the study

Start with a small audit that writes only to `/tmp`:

```bash
iot-diad-audit --stage all --limit-files 4 \
  --output-dir /tmp/iot-diad-audit \
  --report-path /tmp/iot-diad-data-summary.md
```

Run the complete dataset audit and exploratory analysis:

```bash
iot-diad-audit --stage all
iot-diad-eda --stage all
```

The full modeling workflow is intentionally explicit because it scans approximately 51 GB of
source data and produces large local intermediates:

```bash
python -m iot_diad.modeling.baselines prepare
python -m iot_diad.modeling.baselines run
python -m iot_diad.modeling.baselines sensitivity
python scripts/verify_baseline_inputs.py
python scripts/verify_baselines.py

python -m iot_diad.modeling.conformal --root .
python scripts/verify_conformal.py --root .
python scripts/summarize_conformal.py --root . --verify-only

python -m iot_diad.modeling.statistics --root .
python scripts/verify_statistics.py --root .
```

The complete ordered workflow, resource expectations, and artifact requirements are documented in
[Reproducibility](docs/REPRODUCIBILITY.md).

## Validation

```bash
ruff format --check src tests scripts
ruff check src tests scripts
mypy src
bandit -q -r src scripts -ll
pytest
```

These checks cover schema handling, deterministic sampling, device splitting, train-only
preprocessing, conformal threshold construction, paired inference, resume contracts, and
independent artifact verification logic.

## Documentation

- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Dataset audit](docs/reports/data_summary.md)
- [Exploratory analysis](docs/reports/eda_report.md)
- [Device mapping](docs/reports/device_mapping_resolution.md)
- [Split and leakage audit](docs/reports/split_leakage_report.md)
- [Baseline execution](docs/reports/baseline_execution.md)
- [Conformal evaluation](docs/reports/conformal_evaluation.md)
- [Statistical inference](docs/reports/statistical_inference.md)

## Citation

Repository citation metadata is provided in [`CITATION.cff`](CITATION.cff). When using the
dataset, also cite:

> M. Rabbani et al., “Device Identification and Anomaly Detection in IoT Environments,”
> *IEEE Internet of Things Journal*, vol. 12, no. 10, pp. 13625–13643, 2025.
> https://doi.org/10.1109/JIOT.2024.3522863

## Contributing and security

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Please report security issues
according to [SECURITY.md](SECURITY.md), not through a public issue.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
