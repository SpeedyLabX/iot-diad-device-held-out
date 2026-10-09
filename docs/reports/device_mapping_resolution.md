# Published-source device mapping resolution

**Task:** MAP-004  
**Date:** 2026-10-05 (Asia/Ho_Chi_Minh)  
**Disposition:** Gate 1 approved for the packet-primary evaluation

## Approved decision

The full associated paper establishes that the IoT-DIAD feature files were extracted from CICIoT2023 PCAP files. This resolves dataset-level release continuity and strengthens use of the published CICIoT2023 device inventory. It does not provide a row-level or physical-instance crosswalk, does not resolve Flow IPs, and does not distinguish eight same-model Raspberry Pi units represented by one local textual label.

The project lead approved the recommended Gate 1 disposition on 2026-10-05:

1. Use 68 local packet identifiers as uniquely inventory-backed mappings: 64 prior unique exact/MAC candidates plus four deterministic Amazon/Alexa name aliases. These cover 34,627,307 of 35,672,839 packet rows (97.0691%).
2. Keep `Raspberry Pi 4-2 GB` unresolved and exclude it from the primary evaluation because its 35,755 rows map to eight published physical entries.
3. Keep 45 unmatched MAC-form identifiers (788,935 rows) and `not sure` (220,842 rows) unresolved and exclude them from the primary evaluation. Fourteen of those MAC values are locally administered, covering 292,966 rows; this property does not identify a physical device.
4. Make the packet view the primary physical-device-held-out evaluation. Flow cannot support this claim unless authoritative correspondence is later verified.

The decision is materialized separately in `artifacts/device_mapping/approved_packet_device_mapping.csv` so that evidence candidates remain immutable. The excluded population totals 47 identifiers and 1,045,532 rows.

## Full-paper provenance

- Paper: Mahdi Rabbani et al., “Device Identification and Anomaly Detection in IoT Environments.”
- DOI: 10.1109/JIOT.2024.3522863.
- Local source: `artifacts/research/sources/Device_Identification_and_Anomaly_Detection_in_IoT_Environments.pdf`.
- Provenance: supplied by the project lead; the file identifies itself as an IEEE licensed copy downloaded through RMIT University Library.
- PDF: 19 pages; SHA-256 `6e9971aab17a22733d2a62d92d31fa9e75550e1bd12922ee386dba18048bbb18`.
- Extracted text and hashes: `artifacts/device_mapping/literature/full_text_resolution_summary.json`.

The licensed PDF is retained as research evidence and should not be redistributed.

## What the full paper establishes

### Dataset continuity

The paper states in its contributions that experiments use PCAP files from CICIoT2023 (p. 13627). Section IV again says that the experiments use CICIoT2023 network-traffic PCAPs (p. 13636), and Table X identifies the proposed device-identification and anomaly-detection results as CICIoT2023 (p. 13641). IoT-DIAD is therefore a feature extraction and task framing built from the CICIoT2023 captures, rather than an independent physical collection.

This supports using the CICIoT2023 Table 1 inventory as candidate mapping evidence. It does not show how every local `device_mac` value maps to a Table 1 physical instance.

### Evaluation unit and leakage boundary

Section IV says all experiments use a 70% train-test split (p. 13636). For device identification, `device_mac` is the multiclass target; labels with at least 10,000 samples are retained, common labels are capped at 100,000 samples, and 2,723,778 samples across 45 labels are split 70:30 (pp. 13639–13640, Table VIII).

The paper does not report grouping this split by physical device, capture, time, or source identifier. It also does not report that imputation, MinMax scaling, ANOVA, correlation filtering, genetic search, or RFE were fit only on the training partition. Absence of that documentation is not proof of leakage, but the reported scores cannot establish leakage-safe generalization to unseen physical devices.

The phrase “previously unseen devices not included in the training phase” on p. 13636 introduces the zero-day attack experiment. Its operational protocol withholds one attack class and tests that attack class against benign traffic (pp. 13636 and 13639, Fig. 6 and Table VII). It is an unseen-attack-class evaluation, not a physical-device holdout.

### Internal count inconsistency

Page 13636 reports 40 device classes with sufficient representation. Pages 13639–13640 and Table VIII report 45 devices/classes and 2,723,778 samples. The paper provides no reconciliation. Local audited evidence contains 115 raw identifier values, showing that the paper's filtered target classes must not be treated as a complete physical-device inventory.

## Residual deterministic aliases

The only additional deterministic name rule found among the 50 previously unmatched identifiers removes the single published token `Alexa` while preserving all other ordered tokens. It yields four unique candidates:

| Local identifier | Published entry | Rows |
| --- | --- | ---: |
| Amazon Echo Dot 1 | Amazon Alexa Echo Dot 1 | 333,389 |
| Amazon Echo Dot 2 | Amazon Alexa Echo Dot 2 | 115,129 |
| Amazon Echo Spot | Amazon Alexa Echo Spot | 354,970 |
| Amazon Echo Studio | Amazon Alexa Echo Studio | 365,943 |

The four aliases cover 1,169,431 rows. The original evidence artifact retains its pre-decision
`deterministic_name_alias_candidate_not_approved` status and blank canonical fields; the approved
dispositions are recorded separately in `approved_packet_device_mapping.csv`. No fuzzy or
nearest-name match is accepted.

## Evidence and verification

- Base candidates: `artifacts/device_mapping/literature/table1_identifier_candidates.csv`
- Residual aliases: `artifacts/device_mapping/literature/residual_alias_candidates.csv`
- Full-text findings and hashes: `artifacts/device_mapping/literature/full_text_resolution_summary.json`
- Independent artifact/page checks: `artifacts/device_mapping/literature/full_text_coordinator_verification.json`
- Reproduction: `scripts/resolve_published_aliases.py`
- Verification: `scripts/verify_full_text_resolution.py`

The evidence verifier checked the PDF signature and hash, all 19 extracted pages, evidence on
article pages 13627/13636/13639/13640, each alias against the published inventory, row-count
conservation, and the absence of decisions inside the candidate artifacts. The separate Gate 1
verifier checks the approved mapping, excluded population, and conservation totals.
