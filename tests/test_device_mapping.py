from pathlib import Path

from iot_diad.device_mapping import (
    count_tuples,
    flow_candidates,
    ip_scope,
    normalize_identifier,
    packet_candidates,
)
from iot_diad.metadata import FeatureView, SourceFile
from iot_diad.schema import inspect_schema


def source(path: Path, view: FeatureView = "packet", fine: str = "Benign") -> SourceFile:
    return SourceFile(path, str(path), view, "Benign", fine, str(path), 0)


def test_headerless_single_flow_row_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "one.csv"
    path.write_text("id,10.0.0.1,10.0.0.2\n")
    item = source(path, "flow")
    schema = inspect_schema(item, ("Flow ID", "Src IP", "Dst IP"))
    assert schema.status == "headerless"
    assert count_tuples(item, schema) == [
        {"Src IP": "10.0.0.1", "Dst IP": "10.0.0.2", "row_count": 1}
    ]


def test_normalization_preserves_label_case_and_rejects_partial_macs() -> None:
    assert normalize_identifier(" AA-BB-CC-DD-EE-FF ") == "aa:bb:cc:dd:ee:ff"
    assert normalize_identifier("Camera A") != normalize_identifier("camera a")
    assert normalize_identifier("AA:BB:CC") == "AA:BB:CC"
    assert normalize_identifier(None) == ""
    assert ip_scope("172.31.0.1") == "rfc1918"
    assert ip_scope("172.32.0.1") == "other_ip"
    assert ip_scope("not-an-ip") == "invalid"
    assert ip_scope("") == "missing"


def test_packet_endpoint_orientation_does_not_link_peer_or_unmatched() -> None:
    tuples = [
        {
            "device_mac": "Camera",
            "src_mac": "Camera",
            "dst_mac": "Router",
            "src_ip": "10.0.0.2",
            "dst_ip": "10.0.0.1",
            "row_count": 3,
        },
        {
            "device_mac": "Camera",
            "src_mac": "Router",
            "dst_mac": "Camera",
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.2",
            "row_count": 5,
        },
        {
            "device_mac": "Unknown",
            "src_mac": "Camera",
            "dst_mac": "Router",
            "src_ip": "10.0.0.2",
            "dst_ip": "10.0.0.1",
            "row_count": 7,
        },
        {
            "device_mac": "Self",
            "src_mac": "Self",
            "dst_mac": "Self",
            "src_ip": "10.0.0.3",
            "dst_ip": "10.0.0.4",
            "row_count": 2,
        },
    ]
    candidates, coverage = packet_candidates(source(Path("capture.csv")), tuples)
    assert {r["ip"] for r in candidates if r["identifier"] == "Camera"} == {"10.0.0.2"}
    unknown = next(r for r in candidates if r["identifier"] == "Unknown")
    assert unknown["ip"] == "" and unknown["evidence"] == "no_endpoint_match"
    assert sum(r["row_count"] for r in coverage) == 17
    assert sum(r["both_rows"] for r in coverage) == 2
    assert all(r["status"] == "unresolved" and r["canonical_device_id"] == "" for r in candidates)


def test_flow_ambiguous_ips_remain_multiple_tentative_candidates() -> None:
    packet = source(Path("packet/capture.csv"))
    flow = source(Path("flow/capture.pcap_Flow.csv"), "flow")
    packet_rows = [
        {
            "capture_id": packet.capture_id,
            "endpoint_side": "src",
            "ip": "10.0.0.1",
            "ip_scope": "rfc1918",
            "identifier": name,
        }
        for name in ("Camera", "aa:bb:cc:dd:ee:ff")
    ]
    tuples = [{"Src IP": "10.0.0.1", "Dst IP": "10.0.0.2", "row_count": 10}]
    candidates = flow_candidates(flow, tuples, [packet], packet_rows)
    matched = [r for r in candidates if r["field"] == "Src IP"]
    assert {r["packet_candidate_identifier"] for r in matched} == {"Camera", "aa:bb:cc:dd:ee:ff"}
    assert all(r["confidence"] == "tentative" and r["status"] == "unresolved" for r in matched)
    peer = next(r for r in candidates if r["field"] == "Dst IP")
    assert peer["packet_candidate_identifier"] == ""


def test_flow_does_not_join_global_ip_or_different_fine_label() -> None:
    packet = source(Path("packet/other.csv"))
    flow = source(Path("flow/capture.pcap_Flow.csv"), "flow")
    row = {
        "capture_id": packet.capture_id,
        "endpoint_side": "src",
        "ip": "10.0.0.1",
        "ip_scope": "rfc1918",
        "identifier": "Camera",
    }
    tuples = [{"Src IP": "10.0.0.1", "Dst IP": "10.0.0.2", "row_count": 1}]
    result = flow_candidates(flow, tuples, [packet], [row])
    assert all(r["packet_candidate_identifier"] == "" for r in result)
    packet = source(Path("packet/capture.csv"), fine="benign")
    row["capture_id"] = packet.capture_id
    assert all(
        r["packet_candidate_identifier"] == ""
        for r in flow_candidates(flow, tuples, [packet], [row])
    )
