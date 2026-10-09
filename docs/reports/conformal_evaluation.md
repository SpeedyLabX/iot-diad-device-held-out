# CONF-001: frozen class-conditional conformal evaluation

**Scope:** descriptive conformal evaluation at alpha 0.10. This report documents the executed
evaluation; model-comparison inference is reported separately.

## Routing and population

The sole model-output routing source is
`artifacts/experiments/baselines/accepted_model_registry.json`; only entries marked
`primary_learned_model` are processed, with an exact LR, ET, HGB, RF, MLP identity check.
The failed historical fits, DUMMY, and HGB-cap1000 are excluded from this task.
Each accepted model uses all 6,620,817 calibration rows and all 7,912,338 test rows.
Physical devices remain disjoint (14 calibration and 14 test), while captures are shared.
The fixed class order is Benign, Brute Force, DDoS, DoS, Mirai, Recon, Spoofing, Web-based.

`docs/reports/gate3_effective_protocol.md` routes the approved amendments. Historical
`docs/reports/gate3_protocol.md` retains SHA256
`3fa6ba8c8c900cbc4b26a7674fefddbcb1095454718a31bcbc4de1a0bc951bb4`.

## Calibration and arithmetic

Only true-class calibration scores `1 - p_hat(y|x)` determine thresholds. For class `y`, each
of its `K_y` supporting devices contributes mass `1/(K_y+1)`; each of its `N_ky` rows has mass
`1/((K_y+1) N_ky)`. Positive infinity has mass `1/(K_y+1)`. The threshold is the smallest score
with CDF at least exactly `9/10`; membership uses `score <= threshold`.

Scores are saved by original source shard, then accumulated into per-device/class little-endian
float64 files and sorted in place through disk maps. Quantile inversion binary-searches ordered
nonnegative IEEE-754 float64 values. Each CDF comparison sums exact rational counts/weights,
including at tied scores. Insufficient finite mass returns positive infinity, serialized as the
string `infinity`. No test support, labels, coverage, set sizes, or metrics enter quantile inversion.

The separately labeled `equal_row_diagnostic` gives each calibration row and the infinity atom
mass `1/(N_y+1)`. Its finite threshold is the usual order statistic at rank
`ceil(0.9*(N_y+1))`. This comparator has no clustered-device validity claim.

## Artifact layout and reconstruction

All outputs are under `artifacts/experiments/conformal/`:

- `contract.json` binds source, protocol, registry, runner, software, class-order, and probability
  column provenance. Each `prob__<class>` column corresponds to the same-index class.
- `<model>/calibration/*.parquet` retains exact row metadata and every true-class score.
- `<model>/sorted_scores/*.f64` stores reconstructible sorted per-device/class scores.
- `<model>/thresholds.json` records every `K_y`, `N_y`, `N_ky`, score/device/infinity weight,
  finite or infinite threshold, absent calibration device, and sorted-score hash.
- `<model>/sets/*.parquet` retains exact test row metadata and two uint8 membership masks.
  Bit `j` is set exactly when class `j` is included; empty sets are zero and all-class sets are 255.
- Chunk JSON receipts and `<model>/manifest.json` link each output to its original probability
  shard index/hash and to threshold, metrics, and execution-contract hashes.
- `<model>/metrics.json` contains complete cell, class, and overall aggregates. Zero-support
  test cells are explicitly listed with null coverage; they are excluded only from the
  mathematically undefined cell coverage averages and the worst *observed* cell statistic.
- Root CSV files expose model summaries, all class coverage values, all device/class cells,
  calibration device support, and both threshold constructions.
- `verification.json` records independent full-row reconstruction and exact rational CDF brackets.
  `file_inventory.csv` inventories generated files; `logs/` retains commands' outputs.

For each class, device-balanced coverage averages the observed device/class coverage values
with equal device weights. Row-weighted coverage divides all covered rows of that class by all
its test rows. Overall mean set size, singleton rate, and empty-set rate use every test row with
equal row weight. Worst observed device/class coverage is the minimum over nonempty cells.
No comparison reorders models or changes the accepted baseline ranking.

## Descriptive hierarchical results

These are empirical diagnostics on the frozen test partition. They are not model-ranking or
statistical-inference conclusions.

| Model | Mean set size | Singleton rate | Empty-set rate | Worst observed device/class coverage |
|---|---:|---:|---:|---:|
| LR | 1.842310 | 0.662141 | 0.000000 | 0.682468 |
| ET | 1.754960 | 0.659745 | 0.000000 | 0.334034 |
| HGB | 1.207114 | 0.867973 | 0.000000 | 0.800000 |
| RF | 1.621452 | 0.723642 | 0.000000 | 0.821216 |
| MLP | 1.477659 | 0.798212 | 0.000000 | 0.487529 |

All 40 hierarchical thresholds and all 40 equal-row diagnostic thresholds are finite. The full
classwise device-balanced and row-weighted coverage values are in `class_coverage.csv`; all 560
method/model/test-device/class cells, including zero-support cells, are in
`device_class_coverage.csv`. These observed values were computed only after every calibration
threshold was frozen.

## Execution and resume

Use the existing `slx07-iot` environment:

```bash
POLARS_MAX_THREADS=4 python -m iot_diad.modeling.conformal --root .
POLARS_MAX_THREADS=4 python scripts/verify_conformal.py --root .
python scripts/summarize_conformal.py --root .
python scripts/summarize_conformal.py --root . --verify-only
```

A rerun checks the frozen execution contract, source hashes, existing chunk receipts, and exact
chunk contents. It rebuilds derived sorted scores, checks unchanged thresholds and metrics, and
resumes atomic chunk writes. A receipt with corrupt content is rejected. Temporary writes without
a committed receipt may be replaced. `execution.json` preserves the first completed run; a later
completed invocation writes `resume_execution.json`. Memory use is bounded by source shards,
per-class disk maps, and small count tables; no full probability matrix is retained.

The first real startup stopped before writing a contract or thresholds because the runner initially
looked for a training-sample manifest inside amended model directories. The corrected runner
validates the shared canonical training-sample manifest. Its failed startup log remains preserved.

## Limits and scope

Hierarchical exchangeability is an assumption, not an audited property. The deterministic split,
temporal packet structure, and shared captures prevent an unconditional distribution-free coverage
claim for this fixed benchmark. The approved theoretical target is a class-specific row sampled
uniformly within a new class-supporting device under hierarchical exchangeability. It is not a
per-device conditional coverage guarantee. Empirical cells below 90% remain reported unchanged.

No class pooling, alpha change, fallback weighting, row/device exclusion, calibration on test,
undefined S1/S2/S3 scenario, training, tuning, p-value, Holm/BH correction, bootstrap, model ranking
change, Flow experiment, or manuscript interpretation is part of CONF-001. Project-specific
proof and scientific interpretation remain subject to later writing/review and Human Gate 4.
