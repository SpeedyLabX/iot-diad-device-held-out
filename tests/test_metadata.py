from pathlib import Path

import pytest

from iot_diad.metadata import parse_source


def test_parse_nested_flow_capture(tmp_path: Path) -> None:
    path = (
        tmp_path
        / "Anomaly Detection - Flow Based features"
        / "DDoS"
        / "DDoS ACK Fragmentation"
        / "DDoS-ACK_Fragmentation7.pcap_Flow.csv"
    )
    path.parent.mkdir(parents=True)
    path.write_text("x\n", encoding="utf-8")

    source = parse_source(tmp_path, path)

    assert source.feature_view == "flow"
    assert source.coarse_label == "DDoS"
    assert source.fine_label == "DDoS ACK Fragmentation"
    assert source.capture_id.endswith("DDoS-ACK_Fragmentation7.pcap_Flow")


def test_normalize_benign_packet_label(tmp_path: Path) -> None:
    path = (
        tmp_path
        / "Device Identification_Anomaly Detection - Packet Based Features"
        / "BenignTraffic"
        / "BenignTraffic.csv"
    )
    path.parent.mkdir(parents=True)
    path.write_text("x\n", encoding="utf-8")

    source = parse_source(tmp_path, path)

    assert source.feature_view == "packet"
    assert source.coarse_label == "Benign"
    assert source.fine_label == "Benign"


def test_reject_unknown_label(tmp_path: Path) -> None:
    path = tmp_path / "Anomaly Detection - Flow Based features" / "Unknown" / "x.csv"
    path.parent.mkdir(parents=True)
    path.write_text("x\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown coarse label"):
        parse_source(tmp_path, path)
