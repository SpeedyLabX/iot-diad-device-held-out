# Artifact Policy

The repository publishes compact evidence that is useful for inspecting and interpreting the
study while keeping large regenerable files out of Git.

## Included

- dataset inventory and exploratory summaries;
- approved device mappings and split manifests;
- leakage summaries and duplicate exclusions;
- model configurations, fit receipts, aggregate metrics, and verification records;
- conformal thresholds and aggregate coverage summaries;
- device-level statistical tables and bootstrap summaries;
- publication-quality exploratory figures.

## Regenerated locally

- the raw CIC IoT-DIAD dataset;
- sampled training rows and transformed feature caches;
- fitted model binaries;
- calibration and test probability shards;
- sorted conformal scores and row-level prediction-set masks;
- licensed literature files and pre-release manuscript workspaces.

These files remain in their expected paths after local generation but are excluded by
`.gitignore`. This preserves the same execution layout without turning the Git repository into an
artifact store. See [`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) for regeneration steps.
