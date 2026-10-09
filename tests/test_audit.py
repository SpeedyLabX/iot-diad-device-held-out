from pathlib import Path

from iot_diad.audit import (
    aggregate_classes,
    audit_file,
    count_physical_lines,
    render_report,
    select_sources,
)
from iot_diad.metadata import FeatureView, SourceFile
from iot_diad.schema import SchemaInfo, hash_columns


def test_count_physical_lines_handles_missing_final_newline(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_bytes(b"a,b\n1,2\n3,4")

    assert count_physical_lines(path, chunk_size=3) == 3


def test_headerless_row_is_counted_as_data(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text("1,2\n3,4\n", encoding="utf-8")
    source = SourceFile(
        path=path,
        relative_path="flow/DoS/rows.csv",
        feature_view="flow",
        coarse_label="DoS",
        fine_label="DoS",
        capture_id="rows",
        size_bytes=path.stat().st_size,
    )
    columns = ("Flow ID", "Label")
    schema = SchemaInfo(
        has_header=False,
        status="headerless",
        column_count=2,
        schema_hash=hash_columns(columns),
        columns=columns,
    )

    result = audit_file(source, schema)

    assert result.row_count == 2


def test_aggregate_classes() -> None:
    base = {
        "relative_path": "a.csv",
        "feature_view": "flow",
        "coarse_label": "DoS",
        "fine_label": "DoS",
        "capture_id": "a",
        "column_count": 84,
        "has_header": True,
        "schema_status": "canonical",
        "schema_hash": "hash",
    }
    first = audit_from(base, size_bytes=10, row_count=3)
    second = audit_from({**base, "relative_path": "b.csv"}, size_bytes=20, row_count=7)

    assert aggregate_classes([first, second]) == [
        {
            "feature_view": "flow",
            "coarse_label": "DoS",
            "file_count": 2,
            "row_count": 10,
            "size_bytes": 30,
        }
    ]


def test_subset_selection_includes_both_feature_views(tmp_path: Path) -> None:
    sources = [
        source_from(tmp_path, "flow", "large.csv", 100),
        source_from(tmp_path, "flow", "small.csv", 10),
        source_from(tmp_path, "packet", "large.csv", 90),
        source_from(tmp_path, "packet", "small.csv", 9),
    ]

    selected = select_sources(sources, 2)

    assert {(source.feature_view, source.path.name) for source in selected} == {
        ("flow", "small.csv"),
        ("packet", "small.csv"),
    }


def test_report_distinguishes_mac_addresses_from_device_names() -> None:
    summary = {
        "file_count": 2,
        "discovered_file_count": 2,
        "complete_inventory": True,
        "row_count": 20,
        "size_bytes": 100,
        "headerless_file_count": 0,
    }
    identifiers = [
        identifier("packet", "device_mac", "00:11:22:33:44:55"),
        identifier("packet", "device_mac", "Example Camera"),
        identifier("flow", "Src IP", "192.168.1.2"),
    ]

    report = render_report(summary, [], [], identifiers)

    assert "MAC-form values: 1" in report
    assert "Device-name or other textual values: 1" in report
    assert "Distinct private IP values in flow data: 1" in report


def audit_from(values: dict[str, object], **overrides: object):
    from iot_diad.audit import FileAudit

    return FileAudit(**values, **overrides)


def source_from(tmp_path: Path, view: FeatureView, name: str, size_bytes: int) -> SourceFile:
    return SourceFile(
        path=tmp_path / view / name,
        relative_path=f"{view}/{name}",
        feature_view=view,
        coarse_label="Benign",
        fine_label="Benign",
        capture_id=f"{view}/{name}",
        size_bytes=size_bytes,
    )


def identifier(view: str, field: str, value: str) -> dict[str, object]:
    return {
        "feature_view": view,
        "coarse_label": "Benign",
        "field": field,
        "value": value,
        "row_count": 1,
        "capture_count": 1,
    }
