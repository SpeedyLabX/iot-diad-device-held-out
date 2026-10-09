from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

FeatureView = Literal["flow", "packet"]

FLOW_DIRECTORY = "Anomaly Detection - Flow Based features"
PACKET_DIRECTORY = "Device Identification_Anomaly Detection - Packet Based Features"

VIEW_BY_DIRECTORY: dict[str, FeatureView] = {
    FLOW_DIRECTORY: "flow",
    PACKET_DIRECTORY: "packet",
}

LABEL_ALIASES = {
    "Benign": "Benign",
    "BenignTraffic": "Benign",
    "BruteForce": "Brute Force",
    "DDoS": "DDoS",
    "DoS": "DoS",
    "Mirai": "Mirai",
    "Recon": "Recon",
    "Spoofing": "Spoofing",
    "Web-Based": "Web-based",
}


@dataclass(frozen=True, slots=True)
class SourceFile:
    path: Path
    relative_path: str
    feature_view: FeatureView
    coarse_label: str
    fine_label: str
    capture_id: str
    size_bytes: int


def discover_sources(raw_dir: Path) -> list[SourceFile]:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {raw_dir}")
    sources = [parse_source(raw_dir, path) for path in raw_dir.rglob("*.csv")]
    if not sources:
        raise ValueError(f"No CSV files found under {raw_dir}")
    return sorted(sources, key=lambda source: source.relative_path)


def parse_source(raw_dir: Path, path: Path) -> SourceFile:
    relative = path.relative_to(raw_dir)
    parts = relative.parts
    if len(parts) < 3:
        raise ValueError(f"Unexpected dataset path: {relative}")

    directory, raw_label = parts[0], parts[1]
    try:
        feature_view = VIEW_BY_DIRECTORY[directory]
    except KeyError as exc:
        raise ValueError(f"Unknown feature view in path: {relative}") from exc
    try:
        coarse_label = LABEL_ALIASES[raw_label]
    except KeyError as exc:
        raise ValueError(f"Unknown coarse label in path: {relative}") from exc

    stem = strip_capture_suffix(path.name)
    fine_label = parts[-2] if len(parts) > 3 else normalize_capture_name(stem)
    if coarse_label == "Benign":
        fine_label = "Benign"

    return SourceFile(
        path=path,
        relative_path=relative.as_posix(),
        feature_view=feature_view,
        coarse_label=coarse_label,
        fine_label=fine_label,
        capture_id=relative.with_suffix("").as_posix(),
        size_bytes=path.stat().st_size,
    )


def strip_capture_suffix(filename: str) -> str:
    stem = filename.removesuffix(".csv")
    return stem.removesuffix(".pcap_Flow")


def normalize_capture_name(name: str) -> str:
    return re.sub(r"(?<=\D)\d+$", "", name)
