import polars as pl

from iot_diad.eda_analysis import summarize_identifiers, summarize_sample_duplicates


def test_identifier_summary_distinguishes_representations() -> None:
    frame = pl.DataFrame(
        {
            "feature_view": ["packet", "packet", "packet"],
            "coarse_label": ["Benign", "DDoS", "Recon"],
            "field": ["device_mac", "device_mac", "device_mac"],
            "value": ["00:11:22:33:44:55", "Example Camera", "not sure"],
            "row_count": [10, 20, 5],
            "capture_count": [1, 2, 1],
        }
    )

    profile, coverage = summarize_identifiers(frame)

    assert set(profile["representation"]) == {"MAC address", "device name", "ambiguous text"}
    assert coverage.height == 3


def test_sample_duplicate_summary_finds_cross_capture_and_cross_class_rows() -> None:
    frame = pl.DataFrame(
        {
            "feature": [1.0, 1.0, 2.0],
            "_capture_id": ["a", "b", "c"],
            "_coarse_label": ["Benign", "DDoS", "DDoS"],
        }
    )

    summary = summarize_sample_duplicates(frame, "flow", ("feature",), seed=2027)

    assert summary["duplicate_rows"] == 1
    assert summary["rows_in_cross_capture_groups"] == 2
    assert summary["rows_in_cross_class_groups"] == 2
