# Human Gate 3: model, metric, conformal, and inference protocol

**Task:** PROT-001  
**Date:** 2026-10-05 (Asia/Ho_Chi_Minh)  
**Status:** approved 2026-10-05; convergence amendment approved 2026-10-06

## Decision requested

Approve the frozen protocol below for BASE-002, CONF-001, and STAT-001. The protocol preserves the
accepted physical-device-disjoint, capture-shared scope, retains the original five-model request,
and changes the original row-level conformal idea to a device-aware construction. Three
representative models form the confirmatory comparison family; two additional learned models are
reported descriptively so that model breadth does not inflate the primary multiplicity burden.

## Primary scientific target

The primary point-performance estimand is the mean across held-out physical devices of macro-F1
computed over the coarse classes present on each device. Each of the 14 test devices receives equal
weight, regardless of its packet count. The target population is an unseen physical device observed
within the same capture collection and attack scenarios as the training devices.

The confirmatory question is whether the frozen detectors differ in their distribution of
device-level macro-F1. Global row-weighted macro-F1, accuracy, per-class recall, and fine-label
results are secondary or exploratory.

## Evaluation scenarios and feature-view boundary

The advisor guide names `S1`, `S2`, and `S3` but does not define them. No project record provides an
authoritative definition, so this protocol does not invent or use those labels. The frozen primary
scenario is the approved packet-view split over 40 training, 14 calibration, and 14 test physical
devices. It is physical-device-disjoint and capture-shared.

Flow rows do not enter the primary experiment because no authoritative Flow-IP-to-physical-device
crosswalk has been verified. Any later Flow analysis must use a separately stated estimand and
cannot support unseen-physical-device or paired packet-versus-Flow claims. Fine-grained attack
labels are also outside the primary matrix and remain optional after all coarse-label results pass
verification.

## Frozen classifier matrix

Five learned classifiers retain the breadth requested in the advisor guide. `LR`, `ET`, and `HGB`
are frozen as the three representative confirmatory models. `RF` and `MLP` are supplementary
models: they receive the same split, training sample, preprocessing discipline, probability export,
and descriptive metrics, but do not enlarge the confirmatory pairwise family.

| ID | Estimator | Frozen configuration |
|---|---|---|
| `LR` | multinomial logistic regression | `C=1`, L2 penalty, `lbfgs`, `max_iter=1000`, seed 2027 |
| `ET` | extremely randomized trees | 300 trees, `max_features=sqrt`, `max_leaf_nodes=256`, `min_samples_leaf=5`, balanced class weights, all CPU cores, seed 2027 |
| `HGB` | histogram gradient boosting | 200 iterations, learning rate 0.1, 31 leaves, L2=1, balanced class weights, no row-random early stopping, seed 2027 |
| `RF` | random forest | 300 trees, `max_features=sqrt`, `max_leaf_nodes=256`, `min_samples_leaf=5`, balanced-subsample class weights, all CPU cores, seed 2027 |
| `MLP` | multilayer perceptron | one 128-unit ReLU layer, Adam, L2 coefficient `1e-4`, batch size 2048, `max_iter=200`, training-only early stopping with fraction 0.1 and patience 5, seed 2027 |

A prior-only dummy classifier is a pipeline sanity check and appears only in an appendix; it is not
one of the five learned models and is not part of confirmatory pairwise testing. No test or
calibration result may tune these configurations. The tree leaf caps are computational safeguards
for the 27 GiB target machine. If an estimator fails its convergence or memory criterion, the
failure is reported and the matrix is reopened rather than silently changing parameters.

### Approved convergence amendment

The original 2026-10-05 execution used `max_iter=300` for LR and `max_iter=50` for MLP. Both
produced complete finite exports but reached those iteration limits, so their fit statuses failed
the frozen convergence criterion. On 2026-10-06, before conformal calibration or inferential
testing, the project lead approved fresh reruns with `max_iter=1000` for LR and `max_iter=200` for
MLP. No other setting changes. The original models and warnings remain archived as failed attempts.
The amended fits may replace their statuses only after complete independent verification.

Kaggle is approved only as the CPU execution environment for these reruns. The notebook must use
the same seed-2027 sample, fit preprocessing only on training devices, validate manifest and
feature-contract hashes, stream the full unchanged calibration/test population, and record exact
software versions. Calibration or test values cannot select further settings. A new convergence
failure reopens the matrix again.

## Training and prediction budget

- Train from a deterministic, seed-2027 sample capped at 2,000 rows per `(physical device, coarse
  class)` cell, at most 640,000 rows.
- Preserve every available cell with fewer than 2,000 rows. Record selected row keys and a config
  hash so the sample is exactly reproducible.
- Fit imputation, constant removal, scaling, categorical encoding, and the classifier only on this
  training-device sample.
- Do not rebalance calibration or test rows. Stream all 6,620,817 post-duplicate calibration rows
  and all 7,912,338 post-duplicate test rows in bounded chunks. These counts follow the Gate 2
  exclusion totals and must be reconciled by the run manifest.
- Export float probabilities, predicted label, true label, physical device, capture, partition, raw
  row key, model ID, seed, feature-contract hash, sample-manifest hash, and software versions.

The cap is an explicit computational design choice. A sensitivity run at 1,000 rows per cell is
required for the leading model; a 5,000-row cap is optional only if the primary run finishes early.

## Point metrics

Primary:

1. compute macro-F1 separately for every test device over the true classes present on that device;
2. report all 14 values, their equal-device mean, median, interquartile range, and range; and
3. rank all models only from these paired 14-value vectors and restrict confirmatory tests to the
   predeclared `LR`, `ET`, and `HGB` vectors.

Secondary:

- device-balanced per-class recall: average recall equally over test devices that exhibit the class;
- global row-weighted macro-F1 and accuracy for comparison with conventional benchmark reporting;
- per-device confusion matrices and leave-one-device-out influence summaries.

Missing classes on a device are marked not applicable for that device's class-specific calculation;
they are not inserted as zero-recall pseudo-observations.

## Device-balanced class-conditional conformal sets

For candidate class `y`, use nonconformity score

`s(x, y) = 1 - p_hat(y | x)`.

Within each calibration device that exhibits `y`, retain its class-`y` scores. If there are `K_y`
such devices and device `k` has `N_ky` scores, set the class threshold to the `(1-alpha)` weighted
quantile of

`sum_k sum_i [1 / ((K_y + 1) N_ky)] delta_(s_ki) + [1 / (K_y + 1)] delta_(+infinity)`.

Use `alpha=0.10` as the sole confirmatory coverage level. Include `y` in the prediction set when
`s(x,y)` does not exceed its class threshold. All eight coarse classes have 13 or 14 calibration
devices before the 58-row exclusion, so the construction is nontrivial at this alpha; exact post-filter
counts must be verified before calibration.

The appendix will state and prove the project-specific corollary: under hierarchical exchangeability,
after symmetrically retaining the class-`y` observations, HCP provides at least 90% coverage for the
device-balanced target that first draws an unseen device from devices exhibiting `y` and then draws
a class-`y` row uniformly within that device. This is different from ordinary row-weighted
class-conditional coverage.

The paper must say that hierarchical exchangeability is an assumption, not an audited property.
The deterministic split, temporal packet structure, and shared captures prevent an unconditional
distribution-free claim for this fixed benchmark. Results therefore include empirical coverage for
every `(class, test device)` cell, device-balanced class coverage, row-weighted class coverage, mean
set size, singleton rate, empty-set rate, and worst observed device-class coverage.

Ordinary equal-row classwise conformal is included only as a diagnostic comparator and receives no
validity claim under device clustering.

## Paired comparisons and multiplicity

The inferential unit is the physical device (`n=14`). Packet rows never enter a p-value as independent
observations.

1. Run a device-blocked Friedman omnibus test across `LR`, `ET`, and `HGB` using the primary
   per-device macro-F1 values.
2. Regardless of whether the omnibus result is highlighted, report the three predeclared two-sided
   paired Wilcoxon signed-rank comparisons using full-precision device values and exhaustive exact
   sign enumeration when computationally defined.
3. Apply Holm's step-down correction at family-wise alpha 0.05 to exactly these three comparisons.
4. Report paired effect distributions and adjusted p-values. A failure to reject is “insufficient
   evidence of a difference,” never equivalence.

Per-class recall comparisons among the three primary models form one separately labeled exploratory
family of at most 24 hypotheses (three model pairs across eight classes). Apply Benjamini-Hochberg
at false-discovery rate 0.05 to that complete family and report the number of contributing devices
for every test. Conformal efficiency comparisons and training-cap sensitivity remain descriptive
unless a separate family is defined before inspecting results. Holm and BH are never applied to the
same p-values.

## Device bootstrap

Use 50,000 seed-2027 nonparametric bootstrap samples of 14 complete device records, sampled with
replacement. Compute intervals for each model's equal-device mean macro-F1 and each pairwise mean
difference. Preserve the paired model outcomes whenever a device is sampled.

Report percentile intervals as the main bootstrap summary and BCa intervals as a sensitivity check
if the leave-one-device-out acceleration is finite. Also publish the leave-one-device-out estimates.
Because cluster-bootstrap consistency is a many-cluster result and the project has only 14 devices,
call these intervals approximate and do not use them as the sole support for a conclusion.

## Claim changes forced by the literature review

The registered abstract's phrase “bound the miss rate of each attack category without distributional
assumptions” is too broad for this dataset. The manuscript should instead say:

> We construct device-balanced, class-conditional hierarchical conformal sets. They attain
> finite-sample marginal coverage for a class-specific row from a new device under hierarchical
> exchangeability; on this fixed capture-shared benchmark, we report the corresponding empirical
> device-level coverage and state the exchangeability limitation explicitly.

Any sentence claiming that the certified sets kept their promised coverage or that detector gaps lost
statistical support remains a planned result until verified experiments exist.

## Primary sources controlling this protocol

- `docs/research/papers/LEE-2026-HIERARCHICAL.md`
- `docs/research/papers/DING-2023-CLASS-CONDITIONAL.md`
- `docs/research/papers/DUNN-2023-TWO-LAYER.md`
- `docs/research/papers/DEMSAR-2006-COMPARISONS.md`
- `docs/research/papers/HOLM-1979-MULTIPLE-TESTING.md`
- `docs/research/papers/FIELD-2007-CLUSTER-BOOTSTRAP.md`

## Acceptance checks after approval

- Training sample, preprocessing fit, and probability exports reconcile exactly to manifests.
- No calibration or test value influences model fitting or configuration.
- Every conformal threshold can be independently reconstructed from scores and device weights.
- Every metric, p-value, correction, and bootstrap interval can be recomputed from the probability
  exports without refitting a model.
- All claims remain inside the accepted physical-device-disjoint, capture-shared boundary.
- The five learned models all produce complete, class-ordered probability exports, while only the
  three predeclared primary models enter the confirmatory Holm family.
