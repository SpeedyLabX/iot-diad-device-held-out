# STAT-001 statistical inference execution report

**Execution date:** 2026-10-07 (Asia/Ho_Chi_Minh)  
**Scope:** frozen Gate 3 paired-device comparisons, multiplicity correction, and device bootstrap  
**Status:** execution and independent artifact verification passed

## Frozen routing and population

The sole accepted-output router was
`artifacts/experiments/baselines/accepted_model_registry.json`, SHA256
`bbc59f587588f9c4e3b238d224a5ad60b49cbfb4804cada1c039ba4cdde4a6fe`. The runner used exactly
LR `LR-amended-newton-cholesky-maxiter200`, ET `ET`, HGB `HGB`, RF `RF`, and MLP
`MLP-amended-maxiter200`. Failed historical LR/MLP runs, DUMMY, HGB-cap1000, and `Thu_version/`
were not routed into any result.

The fixed test population contains 7,912,338 rows from 14 physical devices. Before inference, the
runner reconstructed every per-device macro-F1 and class recall directly from the accepted 8-by-8
confusion matrices. It matched the stored metrics, class order, row counts, per-class supports,
accepted metric hashes, split membership, and the same 14 paired device identities across models.
The resulting audit tables contain 70 model-device macro-F1 records and 560
model-device-class-recall records.

The hash-preserved Gate 3 protocol remained unchanged at SHA256
`3fa6ba8c8c900cbc4b26a7674fefddbcb1095454718a31bcbc4de1a0bc951bb4`.

## Confirmatory family

The device-blocked Friedman statistic for LR, ET, and HGB is 22.285714 with two degrees of freedom
and an asymptotic chi-square p-value of `1.4478335303304164e-05`. This omnibus calculation did not
select or suppress any pairwise test.

| Frozen paired contrast | Mean device difference, A-B | Exact raw p | Holm p | Decision at 0.05 |
|---|---:|---:|---:|---|
| LR versus ET | -0.021541 | 0.325806 | 0.325806 | insufficient evidence of a difference |
| LR versus HGB | -0.288570 | 0.000122 | 0.000366 | reject equal paired location |
| ET versus HGB | -0.267029 | 0.000122 | 0.000366 | reject equal paired location |

All comparisons used the 14 full-precision paired device outcomes. Exact zero differences were
removed before ranking, exact absolute ties received midranks, and the two-sided tail was obtained
by enumerating all signs of the nonzero paired differences. There were no zero differences or
absolute-rank ties in these three contrasts. Holm adjustment was applied only to these three
predeclared hypotheses. Non-rejection for LR versus ET is not evidence of equivalence.

## Separately labeled exploratory family

The frozen exploratory family contains exactly 24 per-class-recall hypotheses: the same three
model pairs across the eight coarse classes. Tests retain only physical devices supporting the
relevant class, yielding 11 to 14 paired devices depending on class. Benjamini-Hochberg adjustment
was applied once within these 24 hypotheses and nowhere else. Fifteen adjusted p-values are at or
below 0.05 and nine are above 0.05. Complete raw/adjusted p-values, device counts, paired effects,
zero/tie diagnostics, and every contributing device difference are exported. These findings remain
exploratory and do not enlarge the confirmatory family.

## Complete-device bootstrap

The runner used NumPy `default_rng` seed 2027 to generate exactly 50,000 samples, each containing
14 complete physical-device records drawn with replacement. The same sampled indices were used
across models, preserving pairing. Exported estimands are the five accepted models' equal-device
mean macro-F1 values and the three frozen confirmatory paired mean differences. The saved sample
matrix, replicate-estimate matrix, estimand map, percentile intervals, BCa sensitivity intervals,
and all 14 leave-one-device-out estimates make the results reconstructible.

| Estimand | Estimate | 95% percentile interval | 95% BCa sensitivity interval |
|---|---:|---:|---:|
| LR mean macro-F1 | 0.465529 | [0.403278, 0.522350] | [0.396326, 0.517650] |
| ET mean macro-F1 | 0.487070 | [0.401652, 0.564072] | [0.393320, 0.558033] |
| HGB mean macro-F1 | 0.754099 | [0.682686, 0.822804] | [0.678910, 0.819385] |
| RF mean macro-F1 | 0.600022 | [0.513135, 0.680777] | [0.507126, 0.675812] |
| MLP mean macro-F1 | 0.596963 | [0.518582, 0.672015] | [0.514749, 0.669036] |
| LR-ET mean difference | -0.021541 | [-0.058696, 0.021157] | [-0.055496, 0.026861] |
| LR-HGB mean difference | -0.288570 | [-0.318900, -0.251506] | [-0.314424, -0.241639] |
| ET-HGB mean difference | -0.267029 | [-0.325139, -0.214009] | [-0.329439, -0.216810] |

All leave-one-device-out accelerations were finite, so all eight BCa sensitivity intervals are
available. The percentile intervals are the primary bootstrap summaries. Bootstrap intervals are
approximate with only 14 outer devices and are not sole support for a conclusion.

## Independent verification and runtime

The independent verifier imports no runner calculation. It separately reconstructed confusion-
matrix metrics; used SciPy for Friedman and rank construction; independently enumerated signed-rank
sign assignments; recomputed Holm and Benjamini-Hochberg adjustments; regenerated every seed-2027
device sample; and recomputed percentile, BCa, and leave-one-device-out results. Verification passed
for five accepted models, three confirmatory hypotheses, 24 exploratory hypotheses, 14 devices,
and 50,000 bootstrap replicates.

- Runner SHA256: `888ef433fe842dcf7f94fad3bfd5427a2394b5ec707237a0fe6ebc350bff66f0`.
- Verifier SHA256: `5d5770f20933cf0ba23be435c4370718cbe59bbbcd6acafd7a43eec2cf4c2d86`.
- Contract SHA256: `56a26f02161047a652fb90c244ff0a5480b2a91d0afb9b3392abdc45fcf0f0e4`.
- Manifest SHA256: `b488dc9a5327588d64095505e56eaba221e6b82fca7711e9ca06e057c6a0cc4e`.
- Verification SHA256: `3c1d269f2089dc87be3570f8175e8a8f5937c380f36a5660d942beb5ffed7609`.
- File-inventory SHA256: `2aaa95f13ae92ed71e6be60945c5da53d6c9f2569c657636e3e7a30f9b3fc9a2`.
- Fresh execution: 0.606 seconds; actual resume validation: 0.611 seconds.
- Final artifact footprint: approximately 4.6 MiB.

Validation commands included `pytest -q`, Ruff with the explicitly noncanonical `Thu_version/`
tree excluded, strict mypy across 28 relevant source files, Bandit at medium/high severity, mapping,
device-split, Gate 2 exclusion, leakage, frozen training-sample, and STAT-001 artifact verifiers.
The full suite passed 63 tests. An unscoped `ruff check .` reported 386 pre-existing errors only
inside `Thu_version/`; those external comparison files were neither modified nor used.

## Limitations and scope boundary

- Fourteen physical test devices, not packet rows, are the inferential sample.
- The benchmark is deterministic, temporally structured, and capture-shared. Device outcomes may
  remain associated through shared capture conditions, so population-level interpretation is
  bounded to the observed benchmark design.
- Complete-device bootstrap consistency is a many-cluster result; intervals with 14 clusters are
  approximate sensitivity summaries.
- The Friedman p-value uses its standard chi-square approximation. Exact exhaustive enumeration is
  used for the predeclared paired signed-rank tests when defined.
- Exploratory class-recall findings are BH-controlled only within their own frozen family and are
  not confirmatory model-ranking evidence.
- No conformal result was reinterpreted, no model was trained or tuned, no Flow experiment or
  undefined S1/S2/S3 scenario was created, and no manuscript Results/Discussion text was drafted.

Human Gate 4 must approve the main tables, statistical conclusions, and these limitation statements
before WRITE-003 can leave BACKLOG.
