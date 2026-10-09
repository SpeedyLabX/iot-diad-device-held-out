# In-Depth Dataset EDA

## Goal
Produce paper-ready descriptive evidence and split-feasibility diagnostics for both CIC IoT-DIAD feature views without loading the complete dataset into memory.

## Tasks
- [x] Define feature roles, exact-profile metrics, deterministic sampling, and release artifacts.
- [x] Compute exact missingness, parse failures, non-finite values, ranges, and within-capture duplicates.
- [x] Build a capture-aware sample and summarize distributions, redundancy, and cross-capture duplicate risk.
- [x] Quantify packet identifier representations, class coverage, and flow-IP limitations.
- [x] Generate publication-quality figures, a findings report, and an expanded artifact-only notebook.
- [x] Add focused tests and document commands, assumptions, and readiness gates.
- [x] Run the complete 51 GB pipeline and all release checks.

## Done When
- [x] Exact and sample-based results are clearly distinguished in every artifact.
- [x] Future split/model code can consume stable column roles and data-quality outputs.
- [x] No physical-device or leakage claim exceeds the available evidence.
- [x] Tests, lint, typing, security checks, CLI smoke tests, and notebook execution pass.

## Notes
- Exact profiles cover all source rows; distribution and correlation estimates use a fixed-seed sample.
- Raw files remain immutable, and sampled row data is excluded from the release.
