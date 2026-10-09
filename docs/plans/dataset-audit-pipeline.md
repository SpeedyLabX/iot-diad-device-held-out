# Dataset Audit Pipeline

## Goal
Build a reproducible, memory-safe audit and EDA workflow for the 51 GB CIC IoT-DIAD 2024 feature dataset.

## Tasks
- [x] Define the Python package, locked dependencies, configuration, and raw-data contract.
- [x] Implement path-based metadata extraction and schema normalization for both feature views.
- [x] Implement streaming file, schema, row-count, label, and device-identity audits.
- [x] Write compact CSV/JSON artifacts and a generated `docs/reports/data_summary.md`.
- [x] Add an EDA notebook that reads audit artifacts rather than raw CSV files.
- [x] Add focused tests for metadata parsing, schema drift, aggregation, and report generation.
- [x] Run formatting, linting, type checking, security checks, tests, and a real-data smoke audit.

## Done When
- [x] A documented command audits either a small subset or the complete dataset without loading it into memory.
- [x] Generated outputs expose class imbalance, capture counts, schema variants, and device-identity limitations.
- [x] The notebook executes from compact audit artifacts.
- [x] All automated checks pass.

## Notes
- Raw files are immutable and excluded from Git.
- Coarse labels come from directory names because source CSV rows are unlabeled.
- An unseen-device claim is not made until device identifiers are validated for both feature views.
