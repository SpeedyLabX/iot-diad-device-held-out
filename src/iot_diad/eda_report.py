from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from iot_diad.audit import format_bytes


def render_eda_report(audit_dir: Path, eda_dir: Path) -> str:
    audit_summary = pl.read_json(audit_dir / "audit_summary.json")
    classes = pl.read_csv(audit_dir / "class_summary.csv")
    columns = pl.read_csv(eda_dir / "column_profile.csv")
    duplicates = pl.read_csv(eda_dir / "duplicate_summary.csv")
    sample_duplicates = pl.read_csv(eda_dir / "sample_duplicate_summary.csv")
    packet_devices = pl.read_csv(eda_dir / "packet_identifier_profile.csv")
    roles = pl.read_csv(eda_dir / "column_roles.csv")
    correlations = pl.read_csv(eda_dir / "correlation_pairs.csv")
    run = json.loads((eda_dir / "eda_run.json").read_text(encoding="utf-8"))

    summary = audit_summary.row(0, named=True)
    if run["complete_run"]:
        evidence_scope = (
            f"The EDA covers all {int(summary['file_count']):,} CSV files, "
            f"{int(summary['row_count']):,} rows, and {format_bytes(int(summary['size_bytes']))}."
        )
    else:
        evidence_scope = (
            f"The dataset inventory covers {int(summary['file_count']):,} CSV files, but this "
            f"diagnostic EDA run profiles {int(run['selected_file_count']):,} of "
            f"{int(run['discovered_file_count']):,} files. Its profile values are not release "
            "evidence."
        )
    lines = [
        "# CIC IoT-DIAD 2024 In-Depth EDA",
        "",
        "## Scope and Evidence",
        "",
        evidence_scope + " Missingness, parse failures, non-finite values, numeric ranges, and "
        "within-capture "
        "duplicate counts are exact. Distribution, correlation, and cross-capture duplicate "
        "results are estimates from the fixed-seed, capture-aware sample.",
        "",
        "## Class and Capture Structure",
        "",
        "| View | Largest class | Rows | Share | Captures |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for view in ("flow", "packet"):
        group = classes.filter(pl.col("feature_view") == view)
        largest = group.sort("row_count", descending=True).row(0, named=True)
        total = int(group["row_count"].sum())
        lines.append(
            f"| {view} | {largest['coarse_label']} | {int(largest['row_count']):,} | "
            f"{int(largest['row_count']) / total:.1%} | {int(group['file_count'].sum()):,} |"
        )

    lines.extend(["", "## Exact Data-Quality Profile", ""])
    overall_columns = columns.filter(pl.col("scope") == "overall")
    for view in ("flow", "packet"):
        group = overall_columns.filter(pl.col("feature_view") == view)
        missing = group.filter(pl.col("missing_count") > 0).sort("missing_rate", descending=True)
        parse_failures = int(group["parse_failure_count"].sum())
        nonfinite = int(group["nonfinite_count"].sum())
        constant = group.filter(
            (pl.col("inferred_kind") == "numeric")
            & (pl.col("valid_numeric_count") > 0)
            & (pl.col("min") == pl.col("max"))
        )
        sparse = group.filter(
            (pl.col("valid_numeric_count") > 0)
            & (pl.col("zero_count") / pl.col("valid_numeric_count") >= 0.99)
        )
        lines.append(f"### {view.title()} view")
        lines.append("")
        lines.append(
            f"- Columns with at least one missing value: {missing.height} of {group.height}."
        )
        lines.append(f"- Numeric parse failures: {parse_failures:,}.")
        lines.append(f"- Non-finite numeric values: {nonfinite:,}.")
        lines.append(f"- Constant numeric columns: {constant.height}.")
        lines.append(f"- Numeric columns with at least 99% zeros: {sparse.height}.")
        if missing.height:
            lines.append("- Highest missingness: " + format_missing_columns(missing.head(5)) + ".")
        nonfinite_columns = group.filter(pl.col("nonfinite_count") > 0).sort(
            "nonfinite_count", descending=True
        )
        if nonfinite_columns.height:
            lines.append(
                "- Non-finite values occur in: "
                + format_count_columns(nonfinite_columns, "nonfinite_count")
                + "."
            )
        if constant.height:
            lines.append(
                "- Constant columns: "
                + ", ".join(f"`{value}`" for value in constant["column"])
                + "."
            )
        lines.append("")

    lines.extend(
        [
            "## Exact Within-Capture Duplicates",
            "",
            "| View | Raw duplicate rows | Raw rate | Candidate-feature duplicates | "
            "Feature rate |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in duplicates.filter(pl.col("coarse_label") == "All").iter_rows(named=True):
        lines.append(
            f"| {row['feature_view']} | {int(row['raw_duplicate_rows']):,} | "
            f"{float(row['raw_duplicate_rate']):.2%} | "
            f"{int(row['feature_duplicate_rows']):,} | "
            f"{float(row['feature_duplicate_rate']):.2%} |"
        )

    flow_duplicate_classes = (
        duplicates.filter((pl.col("feature_view") == "flow") & (pl.col("coarse_label") != "All"))
        .sort("feature_duplicate_rate", descending=True)
        .head(3)
    )

    lines.extend(
        [
            "",
            "Candidate-feature duplicates exclude group identifiers, session identifiers, source "
            "labels, and high-risk metadata. They are therefore more relevant to model leakage "
            "than duplicates of complete source rows.",
            "",
            "The highest Flow candidate-feature duplicate rates are "
            + ", ".join(
                f"{row['coarse_label']} ({float(row['feature_duplicate_rate']):.2%})"
                for row in flow_duplicate_classes.iter_rows(named=True)
            )
            + ".",
            "",
            "## Sample-Based Cross-Capture Duplicate Risk",
            "",
            "| View | Sample rows | Duplicate rate | Rows in cross-capture groups | "
            "Rows in cross-class groups |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sample_duplicates.iter_rows(named=True):
        lines.append(
            f"| {row['feature_view']} | {int(row['sample_rows']):,} | "
            f"{float(row['duplicate_rate']):.2%} | "
            f"{int(row['rows_in_cross_capture_groups']):,} | "
            f"{int(row['rows_in_cross_class_groups']):,} |"
        )

    representation = packet_devices.group_by("representation").agg(
        pl.len().alias("values"), pl.col("row_count").sum().alias("rows")
    )
    lines.extend(["", "## Device-Identity Feasibility", ""])
    for row in representation.sort("representation").iter_rows(named=True):
        lines.append(
            f"- Packet `device_mac` {row['representation']}: {int(row['values']):,} distinct "
            f"values covering {int(row['rows']):,} rows."
        )
    all_classes = packet_devices.filter(pl.col("class_count") == 8).height
    lines.append(
        f"- {all_classes} of {packet_devices.height} packet identifier values occur in all eight "
        "coarse classes."
    )
    lines.extend(
        [
            "- Flow data exposes private IP endpoints rather than a validated device key.",
            "- The current evidence cannot establish a one-to-one mapping among textual device "
            "names, MAC addresses, and flow IPs.",
            "",
            "## Feature Roles",
            "",
            "| View | Role | Columns |",
            "| --- | --- | ---: |",
        ]
    )
    role_counts = roles.group_by(["feature_view", "role"]).agg(pl.len().alias("columns"))
    for row in role_counts.sort(["feature_view", "role"]).iter_rows(named=True):
        lines.append(f"| {row['feature_view']} | {row['role']} | {int(row['columns'])} |")

    lines.extend(["", "## Strongest Sample Correlations", ""])
    for view in ("flow", "packet"):
        view_correlations = correlations.filter(pl.col("feature_view") == view)
        highly_correlated = view_correlations.filter(pl.col("absolute_correlation") >= 0.95).height
        lines.append(f"### {view.title()} view")
        lines.append("")
        lines.append(f"{highly_correlated} feature pairs have |ρ| ≥ 0.95 in the sample.")
        lines.append("")
        top = view_correlations.head(8)
        for row in top.iter_rows(named=True):
            lines.append(
                f"- `{row['left_feature']}` and `{row['right_feature']}`: "
                f"ρ = {float(row['spearman_correlation']):.3f}."
            )
        lines.append("")

    lines.extend(
        [
            "## Readiness Gates",
            "",
            "| Gate | Status | Evidence |",
            "| --- | --- | --- |",
            "| Schema normalization | Ready | Both canonical schemas are identified; five "
            "headerless Flow rows are retained. |",
            "| Label provenance | Ready | Coarse and fine labels are derived from paths, not the "
            "unreliable source label field. |",
            "| Feature exclusion policy | Ready for baseline | Identifier and high-risk metadata "
            "roles are explicit in `column_roles.csv`. |",
            "| Physical-device mapping | Blocked | Packet identifiers mix representations and Flow "
            "uses IP endpoints. A mapping table is still required. |",
            "| Device-held-out split | Not started | It must wait for a validated physical-device "
            "mapping. |",
            "| Train/test leakage rate | Not measurable yet | It requires the final grouped split; "
            "the present duplicate analysis identifies the risk beforehand. |",
            "",
            "## Implications for the Evaluation Protocol",
            "",
            "1. Build and manually validate a canonical physical-device table before constructing "
            "S1/S2/S3 or train/calibration/test partitions.",
            "2. Exclude all identifier, session, source-label, and high-risk metadata columns from "
            "the primary baseline; introduce high-risk metadata only through a named ablation.",
            "3. Deduplicate within the training partition and audit candidate-feature hashes "
            "across partitions after grouped splitting.",
            "4. Convert non-finite Flow rates to missing values, remove training-constant columns, "
            "and learn every imputation rule from training devices only.",
            "5. Fit scaling, sampling, encoding, and feature selection on training devices only.",
            "6. Include a redundancy-reduced feature ablation because many windowed features are "
            "nearly deterministic transformations of one another.",
            "7. Preserve class and device counts in calibration because rare-class conformal "
            "coverage depends on the available calibration examples.",
            "",
            "## Limitations",
            "",
            "Sample-derived distributions and correlations are descriptive rather than "
            "inferential. Within-capture duplicate counts are exact, while cross-capture "
            "duplicate findings are a risk diagnostic until the full split-aware leakage audit "
            "is run. Source identifiers are treated conservatively and are not assumed to be "
            "physical devices.",
            "",
        ]
    )
    return "\n".join(lines)


def format_missing_columns(frame: pl.DataFrame) -> str:
    return ", ".join(
        f"`{row['column']}` ({float(row['missing_rate']):.1%})"
        for row in frame.iter_rows(named=True)
    )


def format_count_columns(frame: pl.DataFrame, field: str) -> str:
    return ", ".join(
        f"`{row['column']}` ({int(row[field]):,})" for row in frame.iter_rows(named=True)
    )
