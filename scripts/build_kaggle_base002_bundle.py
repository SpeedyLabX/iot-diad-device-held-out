"""Build the audited, minimal input bundle for the BASE-002 Kaggle rerun."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, cast

from iot_diad.modeling.baselines import CONTRACT_PATHS, sha256, write_json

BASELINE_RELATIVE = Path("artifacts/experiments/baselines")
STATIC_INPUTS = (
    "config/base002_kaggle_rerun.json",
    "config/requirements-kaggle-base002.txt",
    *CONTRACT_PATHS,
)


def materialize(source: Path, destination: Path, *, link: bool, replace: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == source.stat().st_size and sha256(destination) == sha256(
            source
        ):
            return
        if not replace:
            raise ValueError(f"Existing bundle file differs: {destination}")
        destination.unlink()
    if link:
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def inventory_record(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def validate_existing(output: Path) -> bool:
    marker = output / "kaggle_bundle_manifest.json"
    if not marker.exists():
        return False
    manifest = json.loads(marker.read_text())
    for record in manifest["files"]:
        path = output / record["path"]
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or sha256(path) != record["sha256"]
        ):
            return False
    return True


def build(root: Path, output: Path, *, refresh: bool = False) -> dict[str, Any]:
    if not refresh and validate_existing(output):
        return cast(
            dict[str, Any],
            json.loads((output / "kaggle_bundle_manifest.json").read_text()),
        )
    if not refresh and output.exists() and any(output.iterdir()):
        raise ValueError(
            f"Incomplete or changed bundle exists at {output}; remove it before rebuilding"
        )
    baseline = root / BASELINE_RELATIVE
    prepared = json.loads((baseline / "prepared.json").read_text())
    config = json.loads((baseline / "config.json").read_text())
    required: list[tuple[Path, bool]] = [(root / relative, False) for relative in STATIC_INPUTS]
    required.extend((path, False) for path in sorted((root / "src/iot_diad").rglob("*.py")))
    required.extend(
        (
            baseline / name,
            name in {"training_sample.parquet", "training_sample_manifest.csv"},
        )
        for name in (
            "config.json",
            "prepared.json",
            "training_sample.parquet",
            "training_sample_manifest.csv",
        )
    )
    required.extend((baseline / source["path"], True) for source in prepared["evaluation_chunks"])
    seen: set[Path] = set()
    for source, link in required:
        if source in seen:
            continue
        seen.add(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        materialize(source, output / source.relative_to(root), link=link, replace=refresh)

    bundled_baseline = output / BASELINE_RELATIVE
    if sha256(bundled_baseline / "config.json") != prepared["config_sha256"]:
        raise ValueError("BASE-002 configuration hash mismatch")
    if sha256(bundled_baseline / "training_sample.parquet") != prepared["training_sample_sha256"]:
        raise ValueError("Training sample hash mismatch")
    if (
        sha256(bundled_baseline / "training_sample_manifest.csv")
        != prepared["sample_manifest_sha256"]
    ):
        raise ValueError("Training sample manifest hash mismatch")
    for source in prepared["evaluation_chunks"]:
        if sha256(bundled_baseline / source["path"]) != source["sha256"]:
            raise ValueError(f"Evaluation shard hash mismatch: {source['path']}")
    for relative, digest in config["contracts"].items():
        if relative == "docs/reports/gate3_protocol.md":
            continue
        if sha256(output / relative) != digest:
            raise ValueError(f"Frozen contract hash mismatch: {relative}")

    files = [
        inventory_record(output, path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "kaggle_bundle_manifest.json"
    ]
    manifest = {
        "bundle_format": 1,
        "evaluation_chunks": len(prepared["evaluation_chunks"]),
        "evaluation_rows": sum(source["rows"] for source in prepared["evaluation_chunks"]),
        "files": files,
        "purpose": "BASE-002 approved LR/MLP convergence rerun only",
        "sample_manifest_sha256": prepared["sample_manifest_sha256"],
        "training_rows": prepared["training_rows"],
    }
    write_json(output / "kaggle_bundle_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiments/baselines/kaggle_rerun_bundle"),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="refresh changed source/config files in an existing generated bundle",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.root / args.output
    manifest = build(args.root.resolve(), output.resolve(), refresh=args.refresh)
    print(
        json.dumps(
            {
                "bundle": str(output.resolve()),
                "evaluation_chunks": manifest["evaluation_chunks"],
                "evaluation_rows": manifest["evaluation_rows"],
                "files": len(manifest["files"]),
                "manifest_sha256": sha256(output / "kaggle_bundle_manifest.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
