"""
Matplotlib plot builders for EOY, METRICS and QUINTILE evaluation modes.
"""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from Aequitas.eval_config import (
    MAX_PLOT_COLUMNS,
    FIXED_SUBPLOT_WIDTH,
    FIXED_SUBPLOT_HEIGHT_EOY,
    FIXED_SUBPLOT_HEIGHT_QUINTILE,
)
from Aequitas.eval_scan import find_column

# -------------------------------------------------
# Low-level helpers
# -------------------------------------------------

def format_length_value(value) -> str:
    try:
        value = float(value)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except Exception:
        return str(value)


def make_length_label(sl, pl) -> str:
    return f"SL={format_length_value(sl)} | PL={format_length_value(pl)}"


def sort_mixed(values) -> list:
    def sort_key(v):
        try:
            return (0, float(v))
        except Exception:
            return (1, str(v))

    return sorted(values, key=sort_key)


def safe_numeric_series(series):
    return pd.to_numeric(series, errors="coerce")


def get_first_column(df, column_name):
    """Always return the first matching Series even if column names are duplicated."""
    values = df.loc[:, column_name]
    if isinstance(values, pd.DataFrame):
        return values.iloc[:, 0]
    return values


def find_plot_column(df, aliases):
    return find_column(df, aliases)


# -------------------------------------------------
# Axis styling
# -------------------------------------------------

def style_evaluation_axis(ax):
    ax.set_facecolor("#fbfdff")
    ax.grid(True, axis="y", color="#e0e7ff", linewidth=0.8, alpha=0.75)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(axis="both", labelsize=8, colors="#334155")
    ax.axhline(0, color="#94a3b8", linewidth=0.9, alpha=0.8, zorder=1)


# -------------------------------------------------
# Shared Y-axis helpers
# -------------------------------------------------

def compute_shared_y_limits(values, include_zero=True, padding_ratio=0.08):
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    y_min = float(np.min(arr))
    y_max = float(np.max(arr))
    if include_zero:
        y_min = min(y_min, 0.0)
        y_max = max(y_max, 0.0)
    if y_min == y_max:
        padding = abs(y_min) * 0.10 if y_min != 0 else 1.0
    else:
        padding = (y_max - y_min) * padding_ratio
    return y_min - padding, y_max + padding


def apply_shared_y_axis(ax, shared_ylim):
    if shared_ylim is None:
        return
    ax.set_ylim(shared_ylim)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


# -------------------------------------------------
# Color map
# -------------------------------------------------

def build_label_color_map(labels: list) -> dict:
    labels = [str(lbl) for lbl in labels]
    palette = [
        "#5b359c", "#ec4899", "#14b8a6", "#0ea5e9", "#10b981",
        "#f59e0b", "#4f46e5", "#ef4444", "#6366f1", "#8b5cf6",
    ]
    return {lbl: palette[idx % len(palette)] for idx, lbl in enumerate(labels)}


# -------------------------------------------------
# Legend helper
# -------------------------------------------------

def collect_unique_legend_items(axes):
    handles, labels = [], []
    for ax in axes.flatten():
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    return handles, labels


# -------------------------------------------------
# Base dataframe builder
# -------------------------------------------------
def build_plot_base_dataframe(df):
    universe_col = find_plot_column(df, ["selected_universe", "universe"])

    model_col = find_plot_column(df, ["model", "selected_model"])

    config_col = find_plot_column(df, ["config", "configuration", "selected_configuration"])
    task_type_col = find_plot_column(df, ["task_type", "type", "selected_task_type"])
    sl_col = find_plot_column(df, ["sl", "sequence_length", "selected_sequence_length"])
    pl_col = find_plot_column(df, ["pl", "prediction_length", "selected_prediction_length"])

    selected_model_col = find_plot_column(df, ["selected_model"])

    if universe_col is None or model_col is None:
        return None

    plot_df = pd.DataFrame({
        "universe": get_first_column(df, universe_col).astype(str),
        "model": get_first_column(df, model_col).astype(str),
    })

    plot_df["config"] = (
        get_first_column(df, config_col).astype(str)
        if config_col is not None else ""
    )

    plot_df["task_type"] = (
        get_first_column(df, task_type_col).astype(str)
        if task_type_col is not None else ""
    )

    plot_df["sl"] = (
        get_first_column(df, sl_col)
        if sl_col is not None else ""
    )

    plot_df["pl"] = (
        get_first_column(df, pl_col)
        if pl_col is not None else ""
    )

    plot_df["selected_model"] = (
        get_first_column(df, selected_model_col).astype(str)
        if selected_model_col is not None else plot_df["model"]
    )

    if "row_type" in df.columns:
        plot_df["row_type"] = get_first_column(df, "row_type").astype(str)
    else:
        benchmark_mask = pd.Series(False, index=plot_df.index)

        for col in ["model", "config", "task_type"]:
            benchmark_mask |= (
                plot_df[col]
                .astype(str)
                .str.lower()
                .str.contains("benchmark|bench|market", regex=True, na=False)
            )

        plot_df["row_type"] = np.where(
            benchmark_mask,
            "benchmark",
            "strategy",
        )

    plot_df["length_label"] = [
        make_length_label(sl, pl)
        for sl, pl in zip(plot_df["sl"], plot_df["pl"])
    ]

    strategy_only = plot_df[
        plot_df["row_type"].astype(str).str.lower().ne("benchmark")
    ].copy()

    if strategy_only.empty:
        strategy_only = plot_df.copy()

    model_config_counts = (
        strategy_only[["model", "config"]]
        .drop_duplicates()
        .groupby("model")
        .size()
    )

    show_config = (
            "config" in strategy_only.columns
            and not model_config_counts.empty
            and model_config_counts.max() > 1
    )

    if show_config:
        plot_df["facet_model"] = plot_df["model"] + "\n" + plot_df["config"]
    else:
        plot_df["facet_model"] = plot_df["model"]

    benchmark_mask = plot_df["row_type"].astype(str).str.lower().eq("benchmark")

    plot_df.loc[benchmark_mask, "facet_model"] = plot_df.loc[
        benchmark_mask,
        "selected_model"
    ]

    return plot_df


# -------------------------------------------------
# Facet grid builder
# -------------------------------------------------

def build_plot_facets(plot_df) -> list:
    universes = sort_mixed(plot_df["universe"].dropna().unique())
    models = sort_mixed(plot_df["facet_model"].dropna().unique())
    facets = []
    for universe in universes:
        for model_name in models:
            subset = plot_df[
                (plot_df["universe"] == universe)
                & (plot_df["facet_model"] == model_name)
                ]
            if not subset.empty:
                facets.append((universe, model_name))
    return facets


def create_fixed_facet_grid(facets, subplot_width, subplot_height,
                            max_columns=MAX_PLOT_COLUMNS, dpi=130):
    n_facets = max(len(facets), 1)
    n_cols = min(max_columns, n_facets)
    n_rows = int(np.ceil(n_facets / n_cols))
    fig_width = subplot_width * n_cols
    fig_height = subplot_height * n_rows + 1.25
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(fig_width, fig_height),
                             squeeze=False, dpi=dpi)
    return fig, axes, n_rows, n_cols


# -------------------------------------------------
# Quintile plot
# -------------------------------------------------

def build_quintile_plot(df):
    if df is None or df.empty:
        return None

    q_columns = []
    for q in ["q1", "q2", "q3", "q4", "q5"]:
        col = find_plot_column(df, [q])
        if col is None:
            return None
        q_columns.append(col)

    base = build_plot_base_dataframe(df)
    if base is None:
        return None

    plot_df = base.copy()
    for q_name, q_col in zip(["q1", "q2", "q3", "q4", "q5"], q_columns):
        plot_df[q_name] = safe_numeric_series(get_first_column(df, q_col))

    plot_df = plot_df.dropna(subset=["q1", "q2", "q3", "q4", "q5"], how="all")
    if plot_df.empty:
        return None

    shared_ylim = compute_shared_y_limits(
        plot_df[["q1", "q2", "q3", "q4", "q5"]].to_numpy(), include_zero=True
    )

    facets = build_plot_facets(plot_df)
    if not facets:
        return None

    length_labels = sort_mixed(plot_df["length_label"].dropna().unique())
    length_color_map = build_label_color_map(length_labels)

    fig, axes, n_rows, n_cols = create_fixed_facet_grid(
        facets, FIXED_SUBPLOT_WIDTH, FIXED_SUBPLOT_HEIGHT_QUINTILE,
        max_columns=MAX_PLOT_COLUMNS, dpi=130,
    )
    fig.patch.set_facecolor("#ffffff")

    x_labels = ["q1", "q2", "q3", "q4", "q5"]
    x = np.arange(len(x_labels))

    for facet_idx, (universe, model_name) in enumerate(facets):
        row_idx, col_idx = divmod(facet_idx, n_cols)
        ax = axes[row_idx][col_idx]

        subset = plot_df[
            (plot_df["universe"] == universe)
            & (plot_df["facet_model"] == model_name)
            ]

        for length_label in length_labels:
            length_subset = subset[subset["length_label"].astype(str) == str(length_label)]
            if length_subset.empty:
                continue
            values = (
                length_subset[["q1", "q2", "q3", "q4", "q5"]]
                .mean(axis=0, numeric_only=True)
                .values
            )
            ax.plot(x, values,
                    marker="o", linewidth=1.8, markersize=4.2,
                    label=str(length_label),
                    color=length_color_map[str(length_label)],
                    zorder=3)

        style_evaluation_axis(ax)
        apply_shared_y_axis(ax, shared_ylim)
        ax.set_title(f"{model_name}", fontsize=10.5, fontweight="bold",
                     color="#1e1b4b", pad=8)
        if col_idx == 0:
            ax.set_ylabel(str(universe), fontsize=9.5, fontweight="bold",
                          color="#1e1b4b", labelpad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)

    for ax in axes.flatten()[len(facets):]:
        ax.axis("off")

    handles, labels = collect_unique_legend_items(axes)
    if handles:
        legend = fig.legend(handles, labels, loc="lower center",
                            ncol=min(len(labels), 4), frameon=True,
                            fontsize=8.5, bbox_to_anchor=(0.5, 0.02))
        legend.get_frame().set_facecolor("#ffffff")
        legend.get_frame().set_edgecolor("#c7d2fe")
        legend.get_frame().set_linewidth(0.9)

    fig.suptitle("Quintile analysis", fontsize=14, fontweight="bold",
                 color="#111827", y=0.985)
    fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.93])
    return fig


# -------------------------------------------------
# EOY plot
# -------------------------------------------------

def build_eoy_plot(df):
    if df is None or df.empty:
        return None

    year_columns = sorted(
        [col for col in df.columns if re.match(r"^\d{4}$", str(col))],
        key=lambda x: int(str(x)),
    )
    if not year_columns:
        return None

    base = build_plot_base_dataframe(df)
    if base is None:
        return None

    plot_df = base.copy()
    for year_col in year_columns:
        plot_df[str(year_col)] = safe_numeric_series(get_first_column(df, year_col))

    plot_df = plot_df.dropna(
        subset=[str(c) for c in year_columns], how="all"
    )
    if plot_df.empty:
        return None

    # Split benchmark vs strategy rows
    if "row_type" in plot_df.columns:
        benchmark_mask = (
            plot_df["row_type"]
            .astype(str)
            .str.lower()
            .eq("benchmark")
        )
    else:
        benchmark_mask = pd.Series(False, index=plot_df.index)

        for col in ["model", "config", "task_type"]:
            if col in plot_df.columns:
                benchmark_mask |= (
                    plot_df[col]
                    .astype(str)
                    .str.lower()
                    .str.contains("benchmark|bench|market", regex=True, na=False)
                )

    benchmark_df = plot_df[benchmark_mask].copy()
    strategy_df = plot_df[~benchmark_mask].copy()

    if strategy_df.empty:
        strategy_df = plot_df.copy()
        benchmark_df = pd.DataFrame()

    id_vars = ["universe", "model", "config", "task_type",
               "sl", "pl", "length_label", "facet_model"]

    long_strategy = strategy_df.melt(
        id_vars=id_vars,
        value_vars=[str(c) for c in year_columns],
        var_name="year", value_name="return",
    )
    long_strategy["return"] = safe_numeric_series(long_strategy["return"])
    long_strategy = long_strategy.dropna(subset=["return"])
    if long_strategy.empty:
        return None

    if not benchmark_df.empty:
        long_benchmark = benchmark_df.melt(
            id_vars=id_vars,
            value_vars=[str(c) for c in year_columns],
            var_name="year", value_name="return",
        )
        long_benchmark["return"] = safe_numeric_series(long_benchmark["return"])
        long_benchmark = long_benchmark.dropna(subset=["return"])
    else:
        long_benchmark = pd.DataFrame()

    shared_y_values = [long_strategy["return"]]
    if not long_benchmark.empty:
        shared_y_values.append(long_benchmark["return"])
    shared_ylim = compute_shared_y_limits(
        pd.concat(shared_y_values, ignore_index=True), include_zero=True
    )

    facets = build_plot_facets(long_strategy)
    if not facets:
        return None

    horizons = sort_mixed(long_strategy["length_label"].dropna().unique())
    years = [str(c) for c in year_columns]
    horizon_color_map = build_label_color_map(horizons)

    fig, axes, n_rows, n_cols = create_fixed_facet_grid(
        facets, FIXED_SUBPLOT_WIDTH, FIXED_SUBPLOT_HEIGHT_EOY,
        max_columns=MAX_PLOT_COLUMNS, dpi=130,
    )
    fig.patch.set_facecolor("#ffffff")

    x = np.arange(len(years))
    n_horizons = max(len(horizons), 1)
    bar_width = min(0.75 / n_horizons, 0.22)

    for facet_idx, (universe, model_name) in enumerate(facets):
        row_idx, col_idx = divmod(facet_idx, n_cols)
        ax = axes[row_idx][col_idx]

        subset = long_strategy[
            (long_strategy["universe"] == universe)
            & (long_strategy["facet_model"] == model_name)
            ]

        for horizon_idx, horizon in enumerate(horizons):
            horizon_subset = subset[subset["length_label"].astype(str) == str(horizon)]
            if horizon_subset.empty:
                continue
            values_by_year = (
                horizon_subset.groupby("year")["return"].mean().reindex(years)
            )
            offset = (horizon_idx - (n_horizons - 1) / 2) * bar_width
            ax.bar(x + offset, values_by_year.values,
                   width=bar_width, label=str(horizon),
                   color=horizon_color_map[str(horizon)])

        if not long_benchmark.empty:
            bench_subset = long_benchmark[long_benchmark["universe"] == universe]
            if not bench_subset.empty:
                benchmark_by_year = (
                    bench_subset.groupby("year")["return"].mean().reindex(years)
                )
                ax.plot(x, benchmark_by_year.values,
                        marker="o", linewidth=1.35, markersize=3.4,
                        label="Benchmark", color="#addb21",
                        linestyle="--", zorder=4)

        style_evaluation_axis(ax)
        apply_shared_y_axis(ax, shared_ylim)
        ax.set_title(f"{model_name}", fontsize=10.5, fontweight="bold",
                     color="#1e1b4b", pad=8)
        if col_idx == 0:
            ax.set_ylabel(str(universe), fontsize=9.5, fontweight="bold",
                          color="#1e1b4b", labelpad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(years, rotation=0)
        ax.tick_params(axis="both", labelsize=8)

    for ax in axes.flatten()[len(facets):]:
        ax.axis("off")

    handles, labels = collect_unique_legend_items(axes)
    if handles:
        legend = fig.legend(handles, labels, loc="lower center",
                            ncol=min(len(labels), 4), frameon=True,
                            fontsize=8.5, bbox_to_anchor=(0.5, 0.02))
        legend.get_frame().set_facecolor("#ffffff")
        legend.get_frame().set_edgecolor("#c7d2fe")
        legend.get_frame().set_linewidth(0.9)

    fig.suptitle("End-of-Year returns", fontsize=13, fontweight="bold",
                 color="#111827", y=0.985)
    fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.93])
    return fig


# -------------------------------------------------
# Dispatcher
# -------------------------------------------------

def build_evaluation_plot(df, eval_mode):
    if eval_mode == "QUINTILE":
        return build_quintile_plot(df)
    if eval_mode == "EOY":
        return build_eoy_plot(df)
    return None
