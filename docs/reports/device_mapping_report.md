# Candidate physical-device mapping review

**Task:** MAP-001, MAP-003, and MAP-004 evidence; MAP-002 approved  
**Date:** 2026-10-05 (Asia/Ho_Chi_Minh)  
**Disposition:** Gate 1 approved for the packet-primary evaluation

## Decision for the project lead

The extraction, published-inventory reconciliation, and full associated-paper review are complete. The paper establishes that IoT-DIAD features derive from CICIoT2023 PCAP files, but it does not supply a physical-instance or Flow-IP crosswalk. The project lead approved 68 unique packet-side inventory mappings covering 97.0691% of packet rows and excluded all ambiguous or unresolved packet identifiers from the primary evaluation. SPLIT-001 is authorized for this packet population.

This review documents the approved canonical device mapping, unresolved identities, accepted
scope, and exclusion rules for the packet-primary evaluation.

## Exact evidence and provenance

The extractor scanned identifier columns in all 312 CSV files, covering 63,482,843 rows: 35,672,839 packet rows and 27,810,004 Flow rows. The five headerless Flow files each retain their single data row. All per-capture row totals, source byte sizes, schema hashes, and audited class-level identifier row/capture counts reconcile with the earlier audit.

| Artifact | Purpose |
| --- | --- |
| `candidate_mapping.csv` (regenerated locally) | 130,860 candidate records with source path, capture, endpoint orientation, evidence, confidence, and unresolved status |
| [packet_identifier_summary.csv](../../artifacts/device_mapping/packet_identifier_summary.csv) | Every one of the 115 packet identifiers, counts, candidate private IPs, and unresolved disposition |
| [packet_identifier_coverage.csv](../../artifacts/device_mapping/packet_identifier_coverage.csv) | Identifier counts and orientation per capture |
| `packet_endpoint_evidence.csv` (regenerated locally) | Exact grouped packet identifier/endpoint tuples |
| `flow_endpoint_evidence.csv` (regenerated locally) | Exact grouped Flow endpoint pairs |
| [source_manifest.csv](../../artifacts/device_mapping/source_manifest.csv) | Source paths, sizes, modification times, schema hashes, row counts, and projected-tuple fingerprints |
| [run_summary.json](../../artifacts/device_mapping/run_summary.json) | Run environment, code/input/artifact hashes, counting rules, and full-data summaries |
| [independent mapping verification](../../artifacts/device_mapping/coordinator_verification.json) | Artifact reconciliation and seven raw-capture checks |
| [table1_identifier_candidates.csv](../../artifacts/device_mapping/literature/table1_identifier_candidates.csv) | Local packet identifiers reconciled against the published CICIoT2023 Table 1 inventory |
| [reconciliation_summary.json](../../artifacts/device_mapping/literature/reconciliation_summary.json) | Exact match coverage, ambiguity, discrepancy, and source hashes |
| [independent literature verification](../../artifacts/device_mapping/literature/coordinator_verification.json) | Candidate-rule, provenance, conservation, and hash checks |
| [device_mapping_resolution.md](device_mapping_resolution.md) | Full-paper continuity evidence, residual aliases, protocol findings, and Gate 1 recommendation |
| [full_text_resolution_summary.json](../../artifacts/device_mapping/literature/full_text_resolution_summary.json) | Full-paper hashes, page-bounded findings, and updated coverage |
| [residual_alias_candidates.csv](../../artifacts/device_mapping/literature/residual_alias_candidates.csv) | Four deterministic Amazon/Alexa evidence candidates; approval is recorded only in the separate decision artifact |
| [approved_packet_device_mapping.csv](../../artifacts/device_mapping/approved_packet_device_mapping.csv) | Approved mapping or exclusion disposition for all 115 packet identifiers |
| [gate1_decision_summary.json](../../artifacts/device_mapping/gate1_decision_summary.json) | Gate 1 population totals and Flow claim boundary |

These are exact full-data identifier observations, not sampled estimates or physical-device counts. Source fingerprints cover metadata and projected values; they are not full-file content hashes for every raw CSV. The verifier separately records full-file hashes for the seven captures it parses.

## Candidate rules

Packet candidates link `device_mac` to `src_ip` only when it equals that row's `src_mac`, and to `dst_ip` only when it equals `dst_mac`. Valid MAC-form values receive case/separator normalization; textual label case is preserved. Neither endpoint matching, both endpoints matching, and missing IPs remain explicit. A name in `src_mac` is still a textual label, not a recovered MAC address.

Flow candidates require an exact coarse label, fine label, and capture basename match after removing `.pcap_Flow`, plus an IP observed on an identifier-matching packet endpoint. Filename correspondence is tentative and does not establish a shared original capture. There is no global-IP fallback or automatic name-to-MAC alias merge. All IP scopes are retained; the RFC1918 subset is identified explicitly.

`direct_endpoint_observation` describes packet-field agreement only. Flow links have `tentative` confidence. The original candidate artifacts remain unchanged and unresolved; the separate approved mapping artifact records the human decision without rewriting source evidence.

Candidate `row_count` values must not be summed to count unique source rows: Flow counts repeat across alternative candidate identifiers, and packet rows matching both endpoints appear once per side. Use endpoint evidence and identifier coverage for row totals.

## Findings and unresolved dispositions

- The 115 packet identifiers comprise 55 MAC-form values and 60 textual values. The earlier EDA classified the latter as 59 device-name values and the ambiguous string `not sure`; the extractor intentionally uses the broader `text_label` category.
- Packet endpoint orientation is source only for 30,299,051 rows, destination only for 5,147,553, both for 5,393, and neither for 220,842. All neither-endpoint rows carry `not sure`; no endpoint is assigned to that label.
- Packet endpoint candidates contain 257 distinct private IPs. Across captures, 251 occur with multiple identifier strings. There are also 117 capture/IP combinations with multiple strings among 9,513 observed private capture/IP combinations. These associations do not determine whether the cause is aliasing, address reuse, traffic behavior, or another mechanism.
- Twenty-eight packet identifiers have no candidate private IP. Their identities remain unresolved; absence of a private IP is not an exclusion decision. Individual values and counts are in the identifier summary.
- Flow evidence contains 266 private IP values. Exact naming rules yield 50 tentative cross-view capture pairs; 82 Flow files and 130 packet files have no counterpart under this rule. For example, Flow `sqlinjection` and packet `SqlInjection` fine-label directory names differ in case and are not automatically equated.
- All Flow endpoints remain unresolved, including those with a single candidate packet string. The output retains alternatives and unmatched endpoints rather than selecting a physical device.

The following rationale applies to every row of the linked candidate and identifier tables:

| Identifier group | Disposition and rationale |
| --- | --- |
| MAC-form packet values | Unresolved: observed address syntax does not establish a physical unit, alias relationships, or role in the experiment |
| Textual packet values other than `not sure` | Unresolved: names/categories do not establish unique physical instances or their actual MAC addresses |
| `not sure` | Unresolved: no matching packet endpoint; retain the observed rows pending an explicit decision |
| Flow IP values with candidate links | Unresolved: address assignment, capture correspondence, alternative identifiers, and physical endpoint attribution require validation |
| Flow IP values without candidate links | Unresolved: no evidence satisfying the conservative cross-view join; no inferred global mapping |

The packet README describes approximately 45 device categories and supplies no physical-instance lookup. Neither local README resolves these cases.

## Published-inventory follow-up

The full IoT-DIAD article states that its feature extraction and experiments use CICIoT2023 PCAP files (article pp. 13627 and 13636). The CICIoT2023 primary paper publishes a device-name/MAC inventory in Table 1. Exact and separator/case-only reconciliation produces candidate matches for 65 of 115 local packet identifiers and 33,493,631 of 35,672,839 packet rows. A deterministic rule that removes only the published token `Alexa` adds four unique Amazon candidates and 1,169,431 rows.

Together, 69 identifiers covering 34,663,062 rows have inventory candidates. Sixty-eight identifiers map uniquely and cover 34,627,307 rows (97.0691%); these mappings are approved for the packet-primary evaluation. `Raspberry Pi 4-2 GB` maps to eight distinct published entries. Forty-five MAC-form identifiers covering 788,935 rows and `not sure` with 220,842 rows remain unmatched. These 47 identifiers and 1,045,532 rows are excluded from the primary evaluation. The published narrative says 105 devices while Table 1 contains 110 entries.

The associated paper also uses a 70:30 sample split over 45 retained `device_mac` labels for device identification and does not report a physical-device-held-out split. Its zero-day experiment withholds attack classes, not physical devices. These results do not resolve Flow endpoint attribution or unlock SPLIT-001.

## Approved scope and effects

1. The 68 unique Table 1-backed packet mappings are the eligible population for the primary physical-device-held-out evaluation.
2. The ambiguous Raspberry Pi label and 46 unmatched identifiers remain unresolved and are excluded from that evaluation; their counts must be reported as a limitation.
3. Flow remains outside unseen-physical-device claims. It may receive a separately framed analysis, while MAP-005 continues the search for authoritative correspondence.
4. SPLIT-001 may construct train, calibration, and test manifests from the approved packet population. Its output still requires Gate 2 review before modeling results are reported.

## Reproduction and verification

From the repository root, using the `slx07-iot` environment:

```bash
POLARS_MAX_THREADS=4 python -m iot_diad.device_mapping
python scripts/verify_device_mapping.py
python scripts/reconcile_published_devices.py
python scripts/verify_published_reconciliation.py
python scripts/resolve_published_aliases.py
python scripts/verify_full_text_resolution.py
python -m pytest
ruff check src tests scripts/verify_device_mapping.py
mypy src/iot_diad scripts/verify_device_mapping.py
bandit -r src/iot_diad scripts/verify_device_mapping.py -ll
iot-diad-state
```

The independent verifier reconciles all 312 captures, 2,624 audited identifier groups, and all candidate dispositions and Flow-link provenance. It separately parses two complete packet captures (Uploading_Attack and Recon-PingSweep) and all five headerless Flow files with Python's CSV reader. That independent raw check covers seven captures, not a second independent full-data parser. The test suite has 23 passing tests, including five focused mapping tests. Scientific identity approval is outside these software checks.
