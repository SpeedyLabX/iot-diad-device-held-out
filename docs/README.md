# Documentation

This directory contains the stable technical documentation for the device-held-out CIC IoT-DIAD
2024 evaluation.

## Reproduction

- [End-to-end reproducibility guide](REPRODUCIBILITY.md)
- [Raw-data layout](../data/README.md)
- [Published artifact policy](../artifacts/README.md)

## Data and evaluation design

- [Dataset audit summary](reports/data_summary.md)
- [Exploratory analysis](reports/eda_report.md)
- [Device-mapping audit](reports/device_mapping_report.md)
- [Device-mapping resolution](reports/device_mapping_resolution.md)
- [Physical-device split](reports/device_split_proposal.md)
- [Split leakage audit](reports/split_leakage_report.md)

## Modeling and inference

- [Feature and preprocessing pipeline](reports/baseline_pipeline.md)
- [Baseline execution contract](reports/baseline_execution.md)
- [Effective evaluation protocol](reports/gate3_effective_protocol.md)
- [Conformal evaluation](reports/conformal_evaluation.md)
- [Device-level statistical inference](reports/statistical_inference.md)

Reports distinguish full-data evidence from sampled estimates and retain the limitations attached
to each result. Machine-readable counterparts are stored under [`artifacts/`](../artifacts/).
