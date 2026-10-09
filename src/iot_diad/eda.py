from __future__ import annotations

import argparse
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import polars as pl

from iot_diad.audit import select_sources, write_csv, write_json, write_text
from iot_diad.eda_analysis import (
    balance_classes,
    load_samples,
    summarize_correlations,
    summarize_distributions,
    summarize_exact_duplicates,
    summarize_histograms,
    summarize_identifiers,
    summarize_sample_duplicates,
)
from iot_diad.eda_plots import generate_figures
from iot_diad.eda_report import render_eda_report
from iot_diad.metadata import FeatureView, SourceFile, discover_sources
from iot_diad.profiling import profile_view_and_duplicates, sample_sources
from iot_diad.roles import column_role, is_numeric_column
from iot_diad.schema import SchemaInfo, find_canonical_schemas, inspect_schema


@dataclass(frozen=True, slots=True)
class EdaConfig:
    raw_dir: Path
    audit_dir: Path
    output_dir: Path
    sample_dir: Path
    report_path: Path
    sample_rows_per_capture: int
    analysis_rows_per_class: int
    correlation_rows_per_view: int
    correlation_feature_count: int
    histogram_bins: int
    seed: int


def load_eda_config(path: Path) -> EdaConfig:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    paths = raw["paths"]
    eda = raw["eda"]
    config = EdaConfig(
        raw_dir=Path(paths["raw_dir"]),
        audit_dir=Path(paths["output_dir"]),
        output_dir=Path(eda["output_dir"]),
        sample_dir=Path(eda["sample_dir"]),
        report_path=Path(eda["report_path"]),
        sample_rows_per_capture=int(eda["sample_rows_per_capture"]),
        analysis_rows_per_class=int(eda["analysis_rows_per_class"]),
        correlation_rows_per_view=int(eda["correlation_rows_per_view"]),
        correlation_feature_count=int(eda["correlation_feature_count"]),
        histogram_bins=int(eda["histogram_bins"]),
        seed=int(eda["seed"]),
    )
    validate_config(config)
    return config


def validate_config(config: EdaConfig) -> None:
    positive = {
        "sample_rows_per_capture": config.sample_rows_per_capture,
        "analysis_rows_per_class": config.analysis_rows_per_class,
        "correlation_rows_per_view": config.correlation_rows_per_view,
        "correlation_feature_count": config.correlation_feature_count,
        "histogram_bins": config.histogram_bins,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError(f"EDA settings must be positive: {', '.join(invalid)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run in-depth EDA for CIC IoT-DIAD 2024.")
    parser.add_argument("--config", type=Path, default=Path("config/audit.toml"))
    parser.add_argument(
        "--stage",
        choices=("profile", "sample", "analyze", "all"),
        default="all",
        help="run exact profiling, sampling, artifact analysis, or the complete workflow",
    )
    parser.add_argument("--limit-files", type=int, help="process a deterministic subset")
    parser.add_argument("--output-dir", type=Path, help="override the EDA artifact directory")
    parser.add_argument("--sample-dir", type=Path, help="override the sampled-row directory")
    parser.add_argument("--report-path", type=Path, help="override the generated report path")
    return parser


def run(args: argparse.Namespace) -> None:
    config = apply_overrides(load_eda_config(args.config), args)
    if args.stage == "analyze":
        analyze_artifacts(config)
        return

    all_sources = discover_sources(config.raw_dir)
    sources = select_sources(all_sources, args.limit_files)
    canonical = find_canonical_schemas(all_sources)
    schemas = inspect_schemas(sources, canonical)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(config, sources, len(all_sources))
    write_column_roles(config.output_dir, canonical)

    if args.stage in {"profile", "all"}:
        run_exact_profile(config, sources, schemas)
    if args.stage in {"sample", "all"}:
        run_sampling(config, sources, schemas)
    if args.stage == "all":
        analyze_artifacts(config)


def apply_overrides(config: EdaConfig, args: argparse.Namespace) -> EdaConfig:
    output_dir = args.output_dir or config.output_dir
    sample_dir = args.sample_dir or (
        output_dir / "samples" if args.output_dir else config.sample_dir
    )
    report_path = args.report_path or config.report_path
    return replace(config, output_dir=output_dir, sample_dir=sample_dir, report_path=report_path)


def inspect_schemas(
    sources: Sequence[SourceFile], canonical: dict[FeatureView, tuple[str, ...]]
) -> dict[str, SchemaInfo]:
    return {
        source.relative_path: inspect_schema(source, canonical[source.feature_view])
        for source in sources
    }


def write_run_metadata(config: EdaConfig, sources: Sequence[SourceFile], discovered: int) -> None:
    write_json(
        config.output_dir / "eda_run.json",
        {
            "selected_file_count": len(sources),
            "discovered_file_count": discovered,
            "complete_run": len(sources) == discovered,
            "seed": config.seed,
            "sample_rows_per_capture": config.sample_rows_per_capture,
            "analysis_rows_per_class": config.analysis_rows_per_class,
            "correlation_rows_per_view": config.correlation_rows_per_view,
            "correlation_feature_count": config.correlation_feature_count,
            "histogram_bins": config.histogram_bins,
        },
    )


def write_column_roles(output_dir: Path, canonical: dict[FeatureView, tuple[str, ...]]) -> None:
    rows: list[dict[str, Any]] = []
    for view, columns in canonical.items():
        for index, column in enumerate(columns):
            role = column_role(column)
            rows.append(
                {
                    "feature_view": view,
                    "column_index": index,
                    "column": column,
                    "inferred_kind": "numeric"
                    if is_numeric_column(view, column)
                    else "categorical",
                    "role": role,
                    "primary_baseline_action": baseline_action(view, column, role),
                    "rationale": role_rationale(role),
                }
            )
    write_csv(output_dir / "column_roles.csv", rows)


def role_rationale(role: str) -> str:
    return {
        "candidate_feature": "Eligible for the primary baseline after train-only preprocessing.",
        "group_identifier": "Retain for grouping and leakage checks; exclude from model inputs.",
        "session_identifier": "May identify sessions or captures; exclude from model inputs.",
        "source_label": "Not a reliable task label and must not be used as a feature.",
        "high_risk_metadata": (
            "Potential shortcut or device fingerprint; reserve for named ablations."
        ),
    }[role]


def baseline_action(feature_view: FeatureView, column: str, role: str) -> str:
    if role != "candidate_feature":
        return "exclude"
    if is_numeric_column(feature_view, column):
        return "include_numeric"
    return "encode_after_validation"


def run_exact_profile(
    config: EdaConfig, sources: Sequence[SourceFile], schemas: dict[str, SchemaInfo]
) -> None:
    rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    for view in ("flow", "packet"):
        print(f"Profiling {view} columns over selected source files...")
        view_rows, view_duplicates = profile_view_and_duplicates(sources, schemas, view)
        rows.extend(view_rows)
        duplicate_rows.extend(view_duplicates)
    write_csv(config.output_dir / "column_profile.csv", rows)
    write_csv(config.output_dir / "duplicate_by_capture.csv", duplicate_rows)
    duplicate_summary = summarize_exact_duplicates(pl.DataFrame(duplicate_rows))
    duplicate_summary.write_csv(config.output_dir / "duplicate_summary.csv")


def run_sampling(
    config: EdaConfig, sources: Sequence[SourceFile], schemas: dict[str, SchemaInfo]
) -> None:
    inventory = pl.read_csv(config.audit_dir / "file_inventory.csv")
    row_counts = dict(
        zip(inventory["relative_path"].to_list(), inventory["row_count"].to_list(), strict=True)
    )
    missing = [source.relative_path for source in sources if source.relative_path not in row_counts]
    if missing:
        raise ValueError(f"Audit inventory is missing {len(missing)} selected source files")
    manifest = sample_sources(
        sources,
        schemas,
        {path: int(count) for path, count in row_counts.items()},
        config.sample_dir,
        config.sample_rows_per_capture,
        config.seed,
    )
    write_csv(config.output_dir / "sample_manifest.csv", manifest)


def analyze_artifacts(config: EdaConfig) -> None:
    roles = pl.read_csv(config.output_dir / "column_roles.csv")
    manifest = pl.read_csv(config.output_dir / "sample_manifest.csv")
    distribution_rows: list[dict[str, Any]] = []
    histogram_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    sample_duplicate_rows: list[dict[str, Any]] = []
    for view in ("flow", "packet"):
        print(f"Analyzing the {view} sample...")
        columns = columns_for_view(roles, view)
        sample = load_samples(manifest, view)
        balanced = balance_classes(sample, config.analysis_rows_per_class, config.seed)
        distribution_rows.extend(summarize_distributions(balanced, view, columns))
        histogram_rows.extend(summarize_histograms(balanced, view, config.histogram_bins))
        pairs, matrix = summarize_correlations(
            balanced,
            view,
            columns,
            config.correlation_rows_per_view,
            config.correlation_feature_count,
            config.seed,
        )
        correlation_rows.extend(pairs)
        matrix.write_csv(config.output_dir / f"correlation_matrix_{view}.csv")
        sample_duplicate_rows.append(
            summarize_sample_duplicates(sample, view, columns, config.seed)
        )
    write_csv(config.output_dir / "feature_distribution_summary.csv", distribution_rows)
    write_csv(config.output_dir / "feature_histograms.csv", histogram_rows)
    write_csv(config.output_dir / "correlation_pairs.csv", correlation_rows)
    write_csv(config.output_dir / "sample_duplicate_summary.csv", sample_duplicate_rows)
    analyze_identifiers(config)
    generate_figures(config.audit_dir, config.output_dir)
    report = render_eda_report(config.audit_dir, config.output_dir)
    write_text(config.output_dir / "eda_report.md", report)
    write_text(config.report_path, report)
    print(f"EDA artifacts written to {config.output_dir}")


def columns_for_view(roles: pl.DataFrame, view: FeatureView) -> tuple[str, ...]:
    return tuple(
        roles.filter(pl.col("feature_view") == view).sort("column_index")["column"].to_list()
    )


def analyze_identifiers(config: EdaConfig) -> None:
    identifiers = pl.read_csv(config.audit_dir / "identifier_counts.csv")
    packet_profile, coverage = summarize_identifiers(identifiers)
    packet_profile.write_csv(config.output_dir / "packet_identifier_profile.csv")
    coverage.write_csv(config.output_dir / "identifier_coverage.csv")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        run(parser.parse_args(argv))
    except (FileNotFoundError, ValueError, RuntimeError, pl.exceptions.PolarsError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
