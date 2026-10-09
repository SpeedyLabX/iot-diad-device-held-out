# Dataset and device-generalization literature review

**Task:** LIT-001  
**Date:** 2026-10-04 (Asia/Ho_Chi_Minh)  
**Scope:** targeted primary-source review of CIC IoT-DIAD 2024, its upstream inventory, and the closest device-aware evaluation studies

## Findings that change the project

The official CIC page describes a dual-purpose release with packet and Flow feature views. Its packet schema calls `device_mac` the device-identification label, while its Flow schema is described for anomaly detection. The full associated IEEE article, supplied by the project lead on 2026-10-05, establishes that the features were extracted from CICIoT2023 PCAP files. It does not expose a row-level crosswalk from local identifiers to physical device instances.

The CICIoT2023 primary paper publishes a name/MAC inventory for the 105-device testbed. Exact reconciliation plus four deterministic Amazon/Alexa aliases yields candidates for 69 of 115 local packet identifier values and 34,663,062 of 35,672,839 packet rows (97.1693%). Of these, 68 identifiers map uniquely, covering 34,627,307 rows; `Raspberry Pi 4-2 GB` corresponds to eight published entries. The published Table 1 contains 110 entries despite the narrative count of 105 devices.

The original IoT-DIAD paper evaluates device identification with a 70:30 sample split over 45 retained `device_mac` labels and does not report grouping by physical device, capture, time, or identifier. Its “unseen” experiment withholds attack classes, not physical devices. The paper reports 40 sufficiently represented device classes on p. 13636 but 45 devices/classes on pp. 13639–13640 and Table VIII. Its near-perfect device-label scores therefore cannot support unseen-physical-device generalization.

A 2026 study on CIC IoT-DIAD already develops device-disjoint split governance, train-only preprocessing, OUI enrichment, and leakage diagnostics. It reports row-level fallback for designated attack classes and explicitly limits strict device-novel claims. This removes any defensible broad novelty claim based only on leakage-aware preprocessing or grouped split construction.

A 2024 device-specific anomaly-detection study reconstructs device-aware CICIoT2023 data from PCAPs, but divides records within devices. It therefore provides reconstruction precedent, not evidence for generalization to wholly unseen physical devices.

## Plan consequence

The research question remains viable, but the contribution must be narrower and stronger: independently verified physical-device grouping, evaluation on wholly held-out physical devices, class-conditional conformal analysis, and device-level statistical inference. These are intended contributions until verified experiments support them.

MAP-004 now establishes source-capture continuity and resolves the only deterministic residual name aliases found without collapsing repeated physical instances. Gate 1 is ready for project-lead review, but all mappings remain candidates and SPLIT-001 stays blocked. Flow IPs, the repeated Raspberry Pi label, 45 unmatched MAC-form identifiers, and `not sure` remain unresolved.

## Source and access limits

- Original dataset page: https://www.unb.ca/cic/datasets/iot-diad-2024.html
- Associated IEEE record: https://ieeexplore.ieee.org/document/10816028/
- Upstream full-text paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC10346235/
- Closest device-disjoint study: https://www.mdpi.com/1999-5903/18/6/287
- Device-specific comparator: https://www.cs.usask.ca/faculty/makaroff/papers/Device-Specific-Anomaly-Detection-Models-for-IoT-Systems.pdf

The licensed IEEE full text was supplied by the project lead and inspected with page-level
provenance; it is retained for research use and is not redistributed. Search-result snippets and
third-party dataset summaries were not used as evidence. The literature search was targeted rather
than systematic.
