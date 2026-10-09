# Packet split leakage audit

**Task:** LEAK-001  
**Date:** 2026-10-05 (Asia/Ho_Chi_Minh)  
**Status:** exact audit complete; recommendation accepted at Human Gate 2

## Finding

The proposed split is strictly physical-device-disjoint but not capture-disjoint. All 68 approved
canonical device IDs occur in exactly one partition, so identifier and physical-device overlap are
both zero. However, each of the 180 packet captures contains eligible rows from training,
calibration, and test devices.

The bipartite device-capture graph has one connected component containing all 68 devices and all
180 captures. Consequently, this release cannot produce a nontrivial split that is simultaneously
physical-device-disjoint and capture-disjoint. Assigning any capture to one partition would force
every connected device and capture into that partition.

This does not erase the unseen-device property: test physical devices remain absent from training.
It narrows the interpretation to unseen devices observed under shared collection sessions and
attack scenarios. It cannot support a stronger claim about generalization to unseen captures,
environments, or collection periods.

## Exact candidate-feature duplicates

The audit hashed all 34,627,307 eligible packet rows over the 116 raw candidate-feature strings
using two independently seeded 64-bit hashes. Eight feature vectors appeared in more than one
partition. A second full 39.8 GB scan recovered every matching row and verified each vector with a
SHA-256 digest over the complete ordered feature-string array.

| Partition span | Feature vectors | Training rows | Calibration rows | Test rows |
| --- | ---: | ---: | ---: | ---: |
| Training and test | 3 | 3 | 0 | 4 |
| All three partitions | 5 | 21 | 13 | 17 |
| **Total** | **8** | **24** | **13** | **21** |

The eight vectors affect 58 rows, or 0.0001675% of the eligible population. None spans multiple
coarse labels. The exact row locations and SHA-256 vector digests are retained in
`verified_cross_partition_duplicates.csv`.

## Gate 2 decision

The project lead accepted the split as **physical-device-disjoint and capture-shared**, with that phrase and its
scope stated in Methods and Limitations. Before modeling, remove all 58 rows belonging to a
feature vector observed in more than one partition using one symmetric, predeclared rule applied
to training, calibration, and test. Retain the unfiltered counts in the audit trail.

This option preserves the paper's central unseen-device question and removes the observed exact
feature duplicates. It does not remove shared capture context. Achieving both device and capture
isolation would require a different collection or authoritative independent capture units; no
reassignment of this connected release can do it.

The executable exclusion manifest removes 24 training, 13 calibration, and 21 test rows, leaving
34,627,249 eligible rows. The accepted claim is generalization to unseen physical devices within
shared captures/scenarios.

## Evidence and reproduction

- `artifacts/leakage/capture_partition_overlap.csv`: exact per-capture partition counts.
- `artifacts/leakage/cross_partition_feature_hashes.parquet`: the eight dual-hash groups.
- `artifacts/leakage/verified_cross_partition_duplicates.csv`: exact matching row locations and
  feature-vector SHA-256 values.
- `artifacts/leakage/leakage_summary.json`: complete counts, graph result, and hashes.
- `artifacts/leakage/duplicate_verification_summary.json`: raw-vector verification result.
- `artifacts/splits/cross_partition_duplicate_exclusions.csv`: approved symmetric row filter.
- `artifacts/splits/gate2_decision_summary.json`: accepted protocol and filtered counts.
- `src/iot_diad/leakage.py`: full-data hash and capture audit.
- `scripts/verify_cross_partition_duplicates.py`: independent raw-vector recovery and verification.

Commands:

```bash
POLARS_MAX_THREADS=4 python -m iot_diad.leakage
POLARS_MAX_THREADS=4 python scripts/verify_cross_partition_duplicates.py
```

The two commands each scan all 180 packet files. Temporary hash shards are deleted after global
aggregation. No model or paper-facing result was produced.
