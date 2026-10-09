# BASE-002 convergence-amendment execution

> **Historical first-amendment record.** This report accurately records the L-BFGS-1,000 failure
> and MLP-200 success at that checkpoint. Its BLOCKED/BACKLOG disposition was later superseded by
> the approved LR Newton–Cholesky amendment in `base002_lr_solver_amendment.md`. The current model
> routing is `artifacts/experiments/baselines/accepted_model_registry.json`.

**Date:** 2026-10-06 (Asia/Ho_Chi_Minh)  
**Execution environment:** local CPU, pinned `slx07-iot` environment  
**Scope:** approved LR and MLP fresh reruns only  
**Outcome:** artifact verification passed; BASE-002 acceptance blocked by LR convergence

## Frozen change

The rerun changed only LR `max_iter` from 300 to 1,000 and MLP `max_iter` from 50 to 200. It used
seed 2027, the exact 446,363-row training sample, train-only preprocessing, the eight-class order,
and all approved post-Gate-2 calibration and test rows. No calibration or test result selected a
setting. ET, HGB, RF, the dummy model, conformal calibration, statistical inference, manuscript
Results, and Flow experiments were not rerun.

The input bundle manifest SHA256 was
`b73090234b76a504a55f2a7281da71681eaa04abddbae35c32a1d247592ce0c4`.

## Fit outcomes

| Model | Approved limit | Actual iterations | Fit time (s) | Status |
|---|---:|---:|---:|---|
| LR | 1,000 | 1,000 | 446.88 | convergence failure |
| MLP | 200 | 79 | 257.28 | passed |

LR emitted the scikit-learn `lbfgs` warning that the total iteration limit was reached. MLP emitted
no warning. The original failed fits at LR 300 and MLP 50 remain preserved separately.

## Complete streamed outputs

Each amended model exported 765 atomic Parquet chunks covering exactly 6,620,817 calibration rows
and 7,912,338 test rows. Combined amended output occupies about 1.7 GiB. Transformed feature arrays
were held one shard at a time and were not retained.

Descriptive equal-device mean macro-F1 values are:

| Model | Calibration | Test |
|---|---:|---:|
| LR amended | 0.4913756615211687 | 0.4739045405556249 |
| MLP amended | 0.5952462155839362 | 0.5969631658447377 |

These are point metrics only. They were not used for tuning and carry no conformal or inferential
claim.

## Verification

`rerun_verification.json` passed. The verifier loaded each saved estimator and checked the approved
parameter delta, class order, preprocessor audit, training rows, warning/status consistency,
iteration record, producing software versions, input lineage, all output receipts and hashes,
row-by-row metadata identity, probability ranges and sums, predicted labels, partition totals, and
reconstructed per-device and aggregate metrics.

- LR model SHA256: `42cb85b38ce10bdf0326a656744961b6a513788b6c1ff5cfa5661e8557381388`
- MLP model SHA256: `ddb3cd2ceef6b07f5fc12f0045adc0662c5f20c71da730df4dd722bbf321c232`
- Verification SHA256: `02059b1d559143cf9b079192f6ef7b3386de1b5fad6aed28efd0e7e57df47d52`
- Summary SHA256: `f1e1641ae50e7adeae83c917b97fcfc18b128bf26247fe5b1d717e98e3c2253a`

## Disposition

The approved amendment says another convergence failure reopens Gate 3. The complete LR output is
valid evidence of a failed fit criterion, not an accepted learned model. BASE-002 is BLOCKED;
CONF-001 and STAT-001 remain BACKLOG. The next chat must obtain a project-lead decision on LR before
changing a setting or accepting a fixed-budget result.

This was the disposition at the end of the first amendment. BASE-002 later reached DONE after the
separately predeclared Newton–Cholesky LR fit passed; see the historical note above.
