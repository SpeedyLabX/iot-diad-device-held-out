# Device-held-out packet split proposal

**Task:** SPLIT-001  
**Date:** 2026-10-05 (Asia/Ho_Chi_Minh)  
**Status:** approved at Human Gate 2 with the capture-shared claim boundary

## Proposed split

The proposal assigns each of the 68 Gate 1-approved physical devices to exactly one partition.
The target device counts are 40 training, 14 calibration, and 14 test devices, corresponding to a
60/20/20 allocation rounded to whole devices. The eligible population contains 34,627,307 packet
rows. The 47 unresolved or ambiguous identifiers and their 1,045,532 rows remain excluded with an
explicit audit trail.

| Partition | Devices | Rows | Row share | Minimum devices supporting any coarse class |
| --- | ---: | ---: | ---: | ---: |
| Training | 40 | 20,094,118 | 58.03% | 37 |
| Calibration | 14 | 6,620,830 | 19.12% | 13 |
| Test | 14 | 7,912,359 | 22.85% | 11 |

## Coarse-class support

Every coarse class occurs in all three partitions. Counts below are exact aggregates from the
packet identifier-by-capture evidence.

| Class | Train rows / devices | Calibration rows / devices | Test rows / devices |
| --- | ---: | ---: | ---: |
| Benign | 643,323 / 38 | 214,377 / 13 | 230,804 / 12 |
| Brute Force | 67,884 / 37 | 23,690 / 13 | 18,587 / 11 |
| DDoS | 12,364,141 / 40 | 4,246,015 / 14 | 5,396,809 / 14 |
| DoS | 2,465,872 / 39 | 528,379 / 14 | 487,000 / 14 |
| Mirai | 3,402,130 / 40 | 1,241,725 / 13 | 1,382,097 / 12 |
| Recon | 640,857 / 39 | 199,232 / 14 | 215,221 / 14 |
| Spoofing | 401,879 / 40 | 123,468 / 14 | 135,709 / 13 |
| Web-based | 108,032 / 40 | 43,944 / 14 | 46,132 / 13 |

## Construction rule

The fixed configuration is in `config/split.toml`. Seed 2027 generates 50,000 candidate
assignments with exact device counts. The selected candidate is improved by deterministic
cross-partition swaps. The objective minimizes deviations in total rows, per-class rows, and
per-class device support from the device-count target, while enforcing at least 10 supporting
devices per training class and five per calibration/test class.

This is a proposal rather than a claim of an optimal split. The objective, weights, seed, input
hashes, selected manifest hash, and final score are recorded in `split_summary.json`.

## Material limitation

The source population is highly concentrated by device. A single approved device contains
69.3405% of all eligible DoS rows. Assigning that device wholly to one partition makes an exact
60/20/20 DoS row balance impossible. It is assigned to training, producing DoS shares of 70.83%,
15.18%, and 13.99% across training, calibration, and test. Device isolation takes precedence over
row balance. This concentration must be reported with the evaluation results and considered when
interpreting device-level uncertainty.

## Artifacts

- `artifacts/splits/device_split_manifest.csv`: one row per approved physical device.
- `artifacts/splits/class_partition_counts.csv`: row and source-device counts by coarse class and
  partition.
- `artifacts/splits/excluded_packet_identifiers.csv`: the Gate 1-excluded population.
- `artifacts/splits/split_summary.json`: configuration, hashes, objective, and partition totals.
- `src/iot_diad/splitting.py`: deterministic generator.
- `scripts/verify_device_split.py`: independent conservation and isolation verifier.

The verifier confirms 68 unique canonical IDs, partition sizes 40/14/14, row conservation,
exact agreement with the approved mapping, all exclusion totals, minimum class-device support,
and output hashes. Capture overlap and candidate-feature duplicate leakage remain LEAK-001 work.
Human Gate 2 accepted the split as physical-device-disjoint and capture-shared. The approved
duplicate filter removes 58 rows before modeling and is recorded in
`artifacts/splits/cross_partition_duplicate_exclusions.csv`.
