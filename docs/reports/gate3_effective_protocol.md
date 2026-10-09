# Gate 3 effective protocol after convergence amendments

**Effective date:** 2026-10-06 (Asia/Ho_Chi_Minh)  
**Status:** approved and executed for BASE-002  
**Purpose:** current routing document for CONF-001 and STAT-001

This document composes the approved Gate 3 protocol with its two recorded convergence amendments.
It does not replace or rewrite `docs/reports/gate3_protocol.md`, whose exact historical bytes are
part of the BASE-002 execution provenance. All scientific choices outside the listed numerical
fit amendments remain exactly as approved in that protocol.

## Effective learned-model matrix

| ID | Effective accepted fit | Role |
|---|---|---|
| `LR` | multinomial L2 logistic regression, `C=1`, `tol=1e-4`, `solver=newton-cholesky`, `max_iter=200`, seed metadata 2027; converged at iteration 69 | confirmatory |
| `ET` | original frozen Gate 3 configuration | confirmatory |
| `HGB` | original frozen Gate 3 configuration | confirmatory |
| `RF` | original frozen Gate 3 configuration | supplementary |
| `MLP` | original frozen configuration with approved `max_iter=200`; converged at iteration 79 | supplementary |

The accepted artifact paths and hashes are machine-readable in
`artifacts/experiments/baselines/accepted_model_registry.json`. Failed LR fits at L-BFGS limits
300 and 1,000 and the original MLP fit at limit 50 remain preserved as failed attempts; downstream
tasks must not route to them.

## Unchanged protocol

- The primary estimand is equal-device mean per-device macro-F1 on 14 test physical devices.
- The deterministic training sample uses seed 2027 and caps each physical-device/class cell at
  2,000 rows.
- Packet is the sole primary unseen-physical-device feature view.
- Calibration and test include every approved post-Gate-2 row.
- LR, ET, and HGB remain the three-model confirmatory family.
- Class-conditional hierarchical conformal prediction is device-balanced at alpha 0.10.
- Holm applies to the three confirmatory model comparisons. BH applies only to the separate
  exploratory per-class-recall family.
- The bootstrap resamples complete physical devices and is reported with the 14-cluster
  limitation.
- Undefined `S1`/`S2`/`S3` labels receive no invented interpretation.

No calibration/test result selected either convergence amendment. The first amendment and its
outcome are recorded in `docs/reports/base002_convergence_rerun.md`; the final LR amendment and its
predeclared acceptance rule are in `docs/reports/base002_lr_solver_amendment.md`. The authoritative
approval chronology is in `docs/decisions/decision_log.md`.
