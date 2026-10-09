from __future__ import annotations

from typing import Literal

from iot_diad.metadata import FeatureView

ColumnRole = Literal[
    "candidate_feature",
    "group_identifier",
    "high_risk_metadata",
    "session_identifier",
    "source_label",
]

FLOW_STRING_COLUMNS = frozenset({"Flow ID", "Src IP", "Dst IP", "Timestamp", "Label"})
PACKET_STRING_COLUMNS = frozenset(
    {
        "src_mac",
        "dst_mac",
        "src_ip",
        "dst_ip",
        "handshake_version",
        "tls_server",
        "http_request_method",
        "http_host",
        "user_agent",
        "dns_server",
        "device_mac",
        "eth_src_oui",
        "eth_dst_oui",
        "highest_layer",
        "http_uri",
        "http_content_type",
    }
)

GROUP_IDENTIFIERS = frozenset(
    {"Src IP", "Dst IP", "src_ip", "dst_ip", "src_mac", "dst_mac", "device_mac"}
)
SESSION_IDENTIFIERS = frozenset(
    {"Flow ID", "Timestamp", "Src Port", "Dst Port", "stream", "src_port", "dst_port"}
)
HIGH_RISK_METADATA = frozenset(
    {
        "eth_src_oui",
        "eth_dst_oui",
        "tls_server",
        "http_host",
        "user_agent",
        "dns_server",
        "http_uri",
    }
)
SOURCE_LABELS = frozenset({"Label"})

REPRESENTATIVE_FEATURES: dict[FeatureView, tuple[str, ...]] = {
    "flow": (
        "Flow Duration",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Packet Length Mean",
        "SYN Flag Count",
        "Down/Up Ratio",
    ),
    "packet": (
        "inter_arrival_time",
        "eth_size",
        "payload_entropy",
        "jitter",
        "stream_10_count",
        "l3_ip_dst_count",
    ),
}


def is_numeric_column(feature_view: FeatureView, column: str) -> bool:
    string_columns = FLOW_STRING_COLUMNS if feature_view == "flow" else PACKET_STRING_COLUMNS
    return column not in string_columns


def column_role(column: str) -> ColumnRole:
    if column in SOURCE_LABELS:
        return "source_label"
    if column in GROUP_IDENTIFIERS:
        return "group_identifier"
    if column in SESSION_IDENTIFIERS:
        return "session_identifier"
    if column in HIGH_RISK_METADATA:
        return "high_risk_metadata"
    return "candidate_feature"


def candidate_feature_columns(feature_view: FeatureView, columns: tuple[str, ...]) -> list[str]:
    return [
        column
        for column in columns
        if column_role(column) == "candidate_feature" and is_numeric_column(feature_view, column)
    ]
