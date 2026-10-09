# CIC IoT-DIAD 2024 Data Summary

## Inventory

- CSV files audited: 312 of 312
- Complete inventory: yes
- Rows: 63,482,843
- Size: 50.8 GiB
- Headerless files: 5

## Class Distribution

| Feature view | Class | Files | Rows | Size |
| --- | --- | ---: | ---: | ---: |
| flow | Benign | 4 | 398,330 | 207.2 MiB |
| flow | Brute Force | 1 | 3,619 | 1.7 MiB |
| flow | DDoS | 61 | 3,478,814 | 1.6 GiB |
| flow | DoS | 30 | 23,143,929 | 11.4 GiB |
| flow | Mirai | 29 | 174,588 | 86.7 MiB |
| flow | Recon | 1 | 442,158 | 216.8 MiB |
| flow | Spoofing | 3 | 157,238 | 80.0 MiB |
| flow | Web-based | 3 | 11,328 | 5.8 MiB |
| packet | Benign | 4 | 1,182,798 | 1.4 GiB |
| packet | Brute Force | 1 | 131,477 | 157.3 MiB |
| packet | DDoS | 101 | 22,471,481 | 25.2 GiB |
| packet | DoS | 38 | 3,503,680 | 3.1 GiB |
| packet | Mirai | 22 | 6,097,219 | 4.6 GiB |
| packet | Recon | 5 | 1,148,459 | 1.1 GiB |
| packet | Spoofing | 3 | 883,823 | 1.2 GiB |
| packet | Web-based | 6 | 253,902 | 295.6 MiB |

## Schema Audit

| Feature view | Status | Columns | Files | Rows |
| --- | --- | ---: | ---: | ---: |
| flow | canonical | 84 | 127 | 27,809,999 |
| flow | headerless | 84 | 5 | 5 |
| packet | canonical | 135 | 180 | 35,672,839 |

## Device Identity

- Distinct packet `device_mac` values: 115
  - MAC-form values: 55
  - Device-name or other textual values: 60
- Distinct private IP values in flow data: 266
- The mixed representation in `device_mac` does not support treating every value as a physical-device identifier without additional validation.
- A physical-device split also requires mapping flow IPs to the same validated device units.

## Label Contract

Source rows are not reliably labeled. Coarse and fine labels are derived from the directory hierarchy and capture filename. Source identifier columns remain available for grouping and leakage audits, but must not be model features.

## Counting Method

Row totals use bounded-memory physical line counts after detecting whether each file has a header. Identifier counts are parsed from projected columns with Polars.
