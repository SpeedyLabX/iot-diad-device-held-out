from pathlib import Path

from iot_diad.metadata import SourceFile
from iot_diad.schema import inspect_schema


def make_source(path: Path) -> SourceFile:
    return SourceFile(
        path=path,
        relative_path="Anomaly Detection - Flow Based features/DoS/capture.csv",
        feature_view="flow",
        coarse_label="DoS",
        fine_label="DoS",
        capture_id="capture",
        size_bytes=path.stat().st_size,
    )


def test_detect_canonical_header(tmp_path: Path) -> None:
    path = tmp_path / "capture.csv"
    path.write_text("Flow ID,Src IP,Label\na,10.0.0.1,x\n", encoding="utf-8")

    result = inspect_schema(make_source(path), ("Flow ID", "Src IP", "Label"))

    assert result.has_header is True
    assert result.status == "canonical"


def test_detect_headerless_csv_without_losing_effective_schema(tmp_path: Path) -> None:
    path = tmp_path / "capture.csv"
    path.write_text("a,10.0.0.1,x\nb,10.0.0.2,x\n", encoding="utf-8")
    canonical = ("Flow ID", "Src IP", "Label")

    result = inspect_schema(make_source(path), canonical)

    assert result.has_header is False
    assert result.status == "headerless"
    assert result.columns == canonical
