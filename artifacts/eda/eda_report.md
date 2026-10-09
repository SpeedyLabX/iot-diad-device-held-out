# CIC IoT-DIAD 2024 In-Depth EDA

## Scope and Evidence

The EDA covers all 312 CSV files, 63,482,843 rows, and 50.8 GiB. Missingness, parse failures, non-finite values, numeric ranges, and within-capture duplicate counts are exact. Distribution, correlation, and cross-capture duplicate results are estimates from the fixed-seed, capture-aware sample.

## Class and Capture Structure

| View | Largest class | Rows | Share | Captures |
| --- | --- | ---: | ---: | ---: |
| flow | DoS | 23,143,929 | 83.2% | 132 |
| packet | DDoS | 22,471,481 | 63.0% | 180 |

## Exact Data-Quality Profile

### Flow view

- Columns with at least one missing value: 0 of 84.
- Numeric parse failures: 0.
- Non-finite numeric values: 179,466.
- Constant numeric columns: 7.
- Numeric columns with at least 99% zeros: 10.
- Non-finite values occur in: `Flow Bytes/s` (89,733), `Flow Packets/s` (89,733).
- Constant columns: `Bwd PSH Flags`, `Bwd URG Flags`, `Fwd Bulk Rate Avg`, `Fwd Bytes/Bulk Avg`, `Fwd Packet/Bulk Avg`, `Fwd URG Flags`, `URG Flag Count`.

### Packet view

- Columns with at least one missing value: 49 of 135.
- Numeric parse failures: 0.
- Non-finite numeric values: 0.
- Constant numeric columns: 1.
- Numeric columns with at least 99% zeros: 9.
- Highest missingness: `dns_query_type` (99.2%), `stream_jitter_1_var` (44.3%), `stream_1_var` (43.4%), `stream_jitter_5_var` (43.1%), `stream_jitter_10_var` (42.4%).
- Constant columns: `handshake_ciphersuites`.

## Exact Within-Capture Duplicates

| View | Raw duplicate rows | Raw rate | Candidate-feature duplicates | Feature rate |
| --- | ---: | ---: | ---: | ---: |
| flow | 33,091 | 0.12% | 3,310,893 | 11.91% |
| packet | 0 | 0.00% | 0 | 0.00% |

Candidate-feature duplicates exclude group identifiers, session identifiers, source labels, and high-risk metadata. They are therefore more relevant to model leakage than duplicates of complete source rows.

The highest Flow candidate-feature duplicate rates are DDoS (16.54%), DoS (11.58%), Mirai (7.61%).

## Sample-Based Cross-Capture Duplicate Risk

| View | Sample rows | Duplicate rate | Rows in cross-capture groups | Rows in cross-class groups |
| --- | ---: | ---: | ---: | ---: |
| flow | 127,005 | 1.32% | 1,461 | 1,160 |
| packet | 180,000 | 0.00% | 0 | 0 |

## Device-Identity Feasibility

- Packet `device_mac` MAC address: 55 distinct values covering 28,422,182 rows.
- Packet `device_mac` ambiguous text: 1 distinct values covering 220,842 rows.
- Packet `device_mac` device name: 59 distinct values covering 7,029,815 rows.
- 63 of 115 packet identifier values occur in all eight coarse classes.
- Flow data exposes private IP endpoints rather than a validated device key.
- The current evidence cannot establish a one-to-one mapping among textual device names, MAC addresses, and flow IPs.

## Feature Roles

| View | Role | Columns |
| --- | --- | ---: |
| flow | candidate_feature | 77 |
| flow | group_identifier | 2 |
| flow | session_identifier | 4 |
| flow | source_label | 1 |
| packet | candidate_feature | 120 |
| packet | group_identifier | 5 |
| packet | high_risk_metadata | 7 |
| packet | session_identifier | 3 |

## Strongest Sample Correlations

### Flow view

30 feature pairs have |ρ| ≥ 0.95 in the sample.

- `Bwd Packet Length Mean` and `Bwd Segment Size Avg`: ρ = 1.000.
- `Fwd Packet Length Mean` and `Fwd Segment Size Avg`: ρ = 1.000.
- `Packet Length Std` and `Packet Length Variance`: ρ = 1.000.
- `Active Mean` and `Active Max`: ρ = 0.999.
- `Bwd Bytes/Bulk Avg` and `Bwd Packet/Bulk Avg`: ρ = 0.998.
- `Bwd Bytes/Bulk Avg` and `Bwd Bulk Rate Avg`: ρ = 0.998.
- `Bwd IAT Total` and `Bwd IAT Max`: ρ = 0.998.
- `Active Mean` and `Active Min`: ρ = 0.998.

### Packet view

133 feature pairs have |ρ| ≥ 0.95 in the sample.

- `q1` and `q1_e`: ρ = 1.000.
- `sum_p` and `average_p`: ρ = 1.000.
- `min_et` and `min_e`: ρ = 1.000.
- `handshake_cipher_suites_length` and `handshake_extensions_length`: ρ = 1.000.
- `icmp_type` and `icmp_checksum_status`: ρ = 1.000.
- `channel_30_mean` and `channel_60_mean`: ρ = 0.998.
- `stream_30_mean` and `stream_60_mean`: ρ = 0.997.
- `stream_30_count` and `stream_60_count`: ρ = 0.997.

## Readiness Gates

| Gate | Status | Evidence |
| --- | --- | --- |
| Schema normalization | Ready | Both canonical schemas are identified; five headerless Flow rows are retained. |
| Label provenance | Ready | Coarse and fine labels are derived from paths, not the unreliable source label field. |
| Feature exclusion policy | Ready for baseline | Identifier and high-risk metadata roles are explicit in `column_roles.csv`. |
| Physical-device mapping | Blocked | Packet identifiers mix representations and Flow uses IP endpoints. A mapping table is still required. |
| Device-held-out split | Not started | It must wait for a validated physical-device mapping. |
| Train/test leakage rate | Not measurable yet | It requires the final grouped split; the present duplicate analysis identifies the risk beforehand. |

## Implications for the Evaluation Protocol

1. Build and manually validate a canonical physical-device table before constructing S1/S2/S3 or train/calibration/test partitions.
2. Exclude all identifier, session, source-label, and high-risk metadata columns from the primary baseline; introduce high-risk metadata only through a named ablation.
3. Deduplicate within the training partition and audit candidate-feature hashes across partitions after grouped splitting.
4. Convert non-finite Flow rates to missing values, remove training-constant columns, and learn every imputation rule from training devices only.
5. Fit scaling, sampling, encoding, and feature selection on training devices only.
6. Include a redundancy-reduced feature ablation because many windowed features are nearly deterministic transformations of one another.
7. Preserve class and device counts in calibration because rare-class conformal coverage depends on the available calibration examples.

## Limitations

Sample-derived distributions and correlations are descriptive rather than inferential. Within-capture duplicate counts are exact, while cross-capture duplicate findings are a risk diagnostic until the full split-aware leakage audit is run. Source identifiers are treated conservatively and are not assumed to be physical devices.
