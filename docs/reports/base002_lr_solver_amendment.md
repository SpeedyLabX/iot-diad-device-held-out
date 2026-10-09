# BASE-002 LR numerical-optimizer amendment

**Approved:** 2026-10-06 (Asia/Ho_Chi_Minh)  
**Authority:** project lead  
**Timing:** declared before the amended fit and before CONF-001 or STAT-001

## Motivation

The frozen LR with L-BFGS reached both approved iteration limits: 300 in the original run and
1,000 in the first convergence amendment. Both failures and their complete outputs remain
preserved. The training design has 446,363 sampled rows and 244 transformed columns across eight
classes. Scikit-learn documents Newton–Cholesky as suitable when the number of samples greatly
exceeds the number of features and notes that it directly computes the Hessian; its quadratic
memory cost in features times classes is acceptable for this matrix.

## Frozen amendment

Only these LR implementation fields change:

| Field | Previous | Approved amendment |
|---|---|---|
| `solver` | `lbfgs` | `newton-cholesky` |
| `max_iter` | 1,000 in the prior amendment | 200 |

The multinomial L2-regularized logistic objective, `C=1`, `tol=1e-4`, seed metadata, exact
seed-2027 training-sample manifest, cap of 2,000 rows per physical-device/class cell, train-only
preprocessing, packet features, class order, and all post-Gate-2 calibration/test rows remain
fixed. The fit starts from scratch. Calibration and test labels or metrics cannot influence the
configuration.

## Acceptance rule

Before evaluation export, the LR fit must:

1. emit no `ConvergenceWarning`;
2. stop with every recorded iteration count below 200;
3. contain finite coefficients and intercepts;
4. produce finite, normalized probabilities on the complete training sample; and
5. match the frozen sample, feature, preprocessing, class, software, and population contracts.

If the fit passes, stream and audit every approved calibration and test row, then replace only the
accepted LR fit in the BASE-002 matrix. If it fails, preserve the failure and reopen Human Gate 3.
No further solver search, tolerance change, regularization change, or evaluation-based selection
is authorized.

## Scope boundary

This amendment remains inside BASE-002. It does not authorize conformal calibration, inferential
tests, manuscript Results, or Flow experiments.
