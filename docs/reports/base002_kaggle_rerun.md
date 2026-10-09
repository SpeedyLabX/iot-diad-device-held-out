# BASE-002 Kaggle convergence rerun

**Scope:** approved Gate 3 amendment for LR and MLP only  
**Execution:** Kaggle CPU; accelerator disabled  
**Status:** verified fallback; amended execution completed locally instead

The project lead chose local execution after this fallback was prepared. The exact bundle ran
locally and produced the outcome in `docs/reports/base002_convergence_rerun.md`. Uploading and
rerunning it on Kaggle is unnecessary unless a later reproducibility check explicitly requests it.

## What this rerun changes

The original BASE-002 execution reached the frozen iteration limit for LR (`300`) and MLP (`50`).
The project lead approved fresh fits with LR `max_iter=1000` and MLP `max_iter=200`. Every other
model parameter, seed, training row, feature, preprocessing rule, class order, and evaluation row
remains fixed. The original failed fits and outputs remain preserved.

This notebook does not run ET, HGB, RF, the dummy model, conformal prediction, inferential tests,
Flow experiments, or manuscript analysis.

## Local preparation

The input bundle has already been generated at:

`artifacts/experiments/baselines/kaggle_rerun_bundle/`

It contains 797 inventoried inputs: the exact 446,363-row training sample, its manifest, all 765
post-Gate-2 calibration/test shards, frozen contracts, the approved amendment, exact package pins,
and the execution source. Its manifest SHA256 is
`b73090234b76a504a55f2a7281da71681eaa04abddbae35c32a1d247592ce0c4`.

The bundle occupies about 3.5 GiB. The local copy uses hard links for large immutable inputs, so it
does not duplicate their storage. Upload the directory as a private Kaggle Dataset. Do not edit a
file after upload.

To reproduce the bundle from the verified local BASE-002 artifacts:

```bash
conda run -n slx07-iot python scripts/build_kaggle_base002_bundle.py
```

## Kaggle execution

1. Create a Kaggle notebook and import
   `notebooks/kaggle_base002_lr_mlp_rerun.ipynb`.
2. Attach the uploaded bundle Dataset.
3. Set **Accelerator** to **None** and enable Internet for the exact pinned package installation.
4. Choose **Save Version → Save & Run All** so Kaggle retains the output files.
5. Do not change a cell, parameter, input Dataset, or output file after the run begins.

The notebook locates the unique `kaggle_bundle_manifest.json`, checks Python and package versions,
validates every input hash, requires at least 5 GiB of free output space, fits both models from
scratch, streams every calibration/test shard, and independently rereads every output probability
file. Each transformed shard is used for both models and released; transformed features are not
retained. Valid probability chunks are reused if the same working directory survives an
interruption. The two original probability directories occupy about 1.9 GiB combined, which gives
a concrete reference for the amended output size before saved models and audit metadata.

The final cell must show `acceptance_status: passed`, `verification_status: passed`, and fit status
`passed` for both LR and MLP. If either fit still reports a convergence warning, preserve all files
and return them unchanged; Gate 3 must reopen.

## Return artifacts

Return the complete `/kaggle/working/base002_kaggle_rerun/` directory. It includes model files,
fit records, class-ordered probability chunks, receipts, per-device and aggregate point metrics,
the full output verification record, and `rerun_summary.json`. Kaggle's saved-version output is
preferred over creating another archive inside the limited working disk.

After download, verify it locally against the same bundle:

```bash
PYTHONPATH=artifacts/experiments/baselines/kaggle_rerun_bundle/src \
python -m iot_diad.modeling.kaggle_rerun \
  --bundle-root artifacts/experiments/baselines/kaggle_rerun_bundle \
  --output-root /path/to/base002_kaggle_rerun \
  --verify-only
```

The local amended artifacts passed complete output verification. BASE-002 is now `BLOCKED` because
LR still failed convergence at 1,000 iterations, not because Kaggle execution is pending.
