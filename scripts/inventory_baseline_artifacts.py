"""List every BASE-002 artifact file for handoff accounting."""

from __future__ import annotations

import csv
import os
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact_root = root / "artifacts/experiments/baselines"
    output = artifact_root / "file_inventory.csv"
    paths = sorted(path for path in artifact_root.rglob("*") if path.is_file() and path != output)
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("relative_path", "size_bytes"))
        for path in paths:
            writer.writerow((path.relative_to(root).as_posix(), path.stat().st_size))
    os.replace(temporary, output)
    print(f"Inventoried {len(paths)} artifact files; inventory excludes itself.")


if __name__ == "__main__":
    main()
