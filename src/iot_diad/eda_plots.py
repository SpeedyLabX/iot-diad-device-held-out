from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from iot_diad.metadata import FeatureView
from iot_diad.roles import REPRESENTATIVE_FEATURES

COLORS = {"flow": "#2F6690", "packet": "#D1495B"}
FEATURE_VIEWS: tuple[FeatureView, FeatureView] = ("flow", "packet")


def generate_figures(audit_dir: Path, eda_dir: Path) -> None:
    figure_dir = eda_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    plot_class_distribution(pl.read_csv(audit_dir / "class_summary.csv"), figure_dir)
    plot_capture_sizes(pl.read_csv(audit_dir / "file_inventory.csv"), figure_dir)
    plot_missingness(pl.read_csv(eda_dir / "column_profile.csv"), figure_dir)
    plot_duplicates(pl.read_csv(eda_dir / "duplicate_summary.csv"), figure_dir)
    plot_identity_coverage(pl.read_csv(eda_dir / "identifier_coverage.csv"), figure_dir)
    distributions = pl.read_csv(eda_dir / "feature_distribution_summary.csv")
    plot_feature_quantiles(distributions, figure_dir)
    histograms = pl.read_csv(eda_dir / "feature_histograms.csv")
    for view in FEATURE_VIEWS:
        plot_feature_histograms(histograms, view, figure_dir)
    for view in ("flow", "packet"):
        plot_correlation_heatmap(view, eda_dir, figure_dir)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_class_distribution(frame: pl.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, constrained_layout=True)
    for axis, view in zip(axes, FEATURE_VIEWS, strict=True):
        group = frame.filter(pl.col("feature_view") == view).sort("row_count")
        axis.barh(group["coarse_label"], group["row_count"], color=COLORS[view])
        axis.set_xscale("log")
        axis.set_title(f"{view.title()} features")
        axis.set_xlabel("Rows (log scale)")
        add_row_labels(axis, group["row_count"].to_list())
    fig.suptitle("Class distribution across the complete dataset")
    save_figure(fig, figure_dir / "class_distribution")


def plot_capture_sizes(frame: pl.DataFrame, figure_dir: Path) -> None:
    labels = frame["coarse_label"].unique().sort().to_list()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, constrained_layout=True)
    for axis, view in zip(axes, FEATURE_VIEWS, strict=True):
        view_frame = frame.filter(pl.col("feature_view") == view)
        values = [
            view_frame.filter(pl.col("coarse_label") == label)["row_count"].to_numpy()
            for label in labels
        ]
        axis.boxplot(values, tick_labels=labels, showfliers=False, patch_artist=True)
        for patch in axis.patches:
            patch.set_facecolor(COLORS[view])
            patch.set_alpha(0.7)
        axis.set_yscale("log")
        axis.tick_params(axis="x", rotation=45)
        axis.set_title(f"{view.title()} captures")
        axis.set_ylabel("Rows per capture (log scale)")
    fig.suptitle("Capture-size heterogeneity")
    save_figure(fig, figure_dir / "capture_size_distribution")


def plot_missingness(frame: pl.DataFrame, figure_dir: Path) -> None:
    overall = frame.filter(pl.col("scope") == "overall")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)
    for axis, view in zip(axes, FEATURE_VIEWS, strict=True):
        group = (
            overall.filter(pl.col("feature_view") == view)
            .sort("missing_rate", descending=True)
            .head(15)
            .sort("missing_rate")
        )
        max_missing = group.select(pl.max("missing_rate")).item() if group.height else 0.0
        if not max_missing:
            axis.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=11)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(f"{view.title()} features")
            continue
        axis.barh(group["column"], group["missing_rate"] * 100, color=COLORS[view])
        axis.set_title(f"{view.title()} features")
        axis.set_xlabel("Missing rows (%)")
        axis.set_xlim(left=0)
    fig.suptitle("Most incomplete columns (exact full-data counts)")
    save_figure(fig, figure_dir / "missingness")


def plot_duplicates(frame: pl.DataFrame, figure_dir: Path) -> None:
    overall = frame.filter(pl.col("coarse_label") == "All").sort("feature_view")
    positions = np.arange(overall.height)
    width = 0.34
    fig, axis = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    axis.bar(
        positions - width / 2,
        overall["raw_duplicate_rate"] * 100,
        width,
        label="Complete source rows",
        color="#7A8B99",
    )
    axis.bar(
        positions + width / 2,
        overall["feature_duplicate_rate"] * 100,
        width,
        label="Candidate-feature rows",
        color="#E09F3E",
    )
    axis.set_xticks(positions, [value.title() for value in overall["feature_view"]])
    axis.set_ylabel("Within-capture duplicate rows (%)")
    axis.set_title("Exact within-capture duplicate burden")
    axis.legend(frameon=False)
    save_figure(fig, figure_dir / "duplicate_rates")


def plot_identity_coverage(frame: pl.DataFrame, figure_dir: Path) -> None:
    packet = frame.filter(
        (pl.col("feature_view") == "packet") & (pl.col("field") == "device_mac")
    ).sort("distinct_values")
    fig, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    axis.barh(packet["coarse_label"], packet["distinct_values"], color=COLORS["packet"])
    axis.set_xlabel("Distinct source values")
    axis.set_title("Packet device_mac coverage by class")
    save_figure(fig, figure_dir / "packet_device_coverage")


def plot_feature_quantiles(frame: pl.DataFrame, figure_dir: Path) -> None:
    overall = frame.filter(pl.col("coarse_label") == "All")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for axis, view in zip(axes, FEATURE_VIEWS, strict=True):
        features = REPRESENTATIVE_FEATURES[view]
        group = overall.filter((pl.col("feature_view") == view) & pl.col("feature").is_in(features))
        ordered = [feature for feature in features if feature in set(group["feature"])]
        group = group.join(
            pl.DataFrame({"feature": ordered, "order": range(len(ordered))}), on="feature"
        ).sort("order")
        medians = group["median"].to_numpy()
        lower = medians - group["q25"].to_numpy()
        upper = group["q75"].to_numpy() - medians
        axis.errorbar(
            medians,
            np.arange(len(ordered)),
            xerr=np.vstack([lower, upper]),
            fmt="o",
            color=COLORS[view],
            capsize=3,
        )
        axis.set_yticks(np.arange(len(ordered)), ordered)
        axis.set_xscale("symlog", linthresh=1e-3)
        axis.set_title(f"{view.title()} features")
        axis.set_xlabel("Median and interquartile range (symmetric log scale)")
    fig.suptitle("Representative feature scales from the balanced sample")
    save_figure(fig, figure_dir / "representative_feature_quantiles")


def plot_feature_histograms(frame: pl.DataFrame, view: FeatureView, figure_dir: Path) -> None:
    view_frame = frame.filter(pl.col("feature_view") == view)
    features = [
        feature
        for feature in REPRESENTATIVE_FEATURES[view]
        if feature in set(view_frame["feature"])
    ]
    labels = view_frame["coarse_label"].unique().sort().to_list()
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    for axis in axes.flat:
        axis.set_visible(False)
    for axis, feature in zip(axes.flat, features, strict=False):
        axis.set_visible(True)
        feature_frame = view_frame.filter(pl.col("feature") == feature)
        for label, color in zip(labels, colors, strict=True):
            group = feature_frame.filter(pl.col("coarse_label") == label).sort("bin_left")
            midpoint = (group["bin_left"] + group["bin_right"]) / 2
            axis.plot(midpoint, group["density"], label=label, color=color, linewidth=1.1)
        axis.set_title(feature)
        axis.set_xscale("symlog", linthresh=1e-3)
        axis.set_ylabel("Sample share")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(f"{view.title()} class-conditional feature distributions (1st–99th percentile)")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.18, hspace=0.42, wspace=0.24)
    save_figure(fig, figure_dir / f"feature_distributions_{view}")


def plot_correlation_heatmap(view: FeatureView, eda_dir: Path, figure_dir: Path) -> None:
    frame = pl.read_csv(eda_dir / f"correlation_matrix_{view}.csv")
    labels = frame["feature"].to_list()
    matrix = frame.drop("feature").to_numpy()
    fig, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    axis.set_yticks(range(len(labels)), labels, fontsize=6)
    axis.set_title(f"{view.title()} feature Spearman correlations")
    fig.colorbar(image, ax=axis, label="Spearman correlation", shrink=0.8)
    save_figure(fig, figure_dir / f"correlation_{view}")


def add_row_labels(axis: Axes, values: list[int]) -> None:
    for patch, value in zip(axis.patches, values, strict=True):
        if not isinstance(patch, Rectangle):
            continue
        axis.text(
            patch.get_width() * 1.05,
            patch.get_y() + patch.get_height() / 2,
            f"{value / 1_000_000:.2f}M",
            va="center",
            fontsize=7,
        )


def save_figure(figure: Figure, path: Path) -> None:
    figure.savefig(path.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
