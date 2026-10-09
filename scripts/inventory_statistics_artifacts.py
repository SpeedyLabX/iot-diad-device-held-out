"""Hash and inventory every STAT-001 artifact for closeout accounting."""

from __future__ import annotations

import csv
import os
from pathlib import Path

from iot_diad.modeling.baselines import sha256


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact_root = root / "artifacts/experiments/statistics"
    output = artifact_root / "file_inventory.csv"
    paths = sorted(path for path in artifact_root.rglob("*") if path.is_file() and path != output)
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("relative_path", "size_bytes", "sha256"))
        for path in paths:
            writer.writerow((path.relative_to(root).as_posix(), path.stat().st_size, sha256(path)))
    os.replace(temporary, output)
    print(f"Inventoried and hashed {len(paths)} artifact files; inventory excludes itself.")


if __name__ == "__main__":
    main()
