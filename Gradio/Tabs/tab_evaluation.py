import re
import sys
import shlex
import subprocess
from pathlib import Path
from functools import partial
import pandas as pd
import gradio as gr
import numpy as np
from html import escape
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

MAX_PLOT_COLUMNS = 3
FIXED_SUBPLOT_WIDTH = 4.2
FIXED_SUBPLOT_HEIGHT_EOY = 2.55
FIXED_SUBPLOT_HEIGHT_QUINTILE = 2.65

# -------------------------------------------------
# OpenAI config
# -------------------------------------------------
OPENAI_EXPLANATION_MODEL = os.getenv("OPENAI_EXPLANATION_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(f"Using OpenAI model: {OPENAI_EXPLANATION_MODEL}")

# -------------------------------------------------
# Evaluation config
# -------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()

# tab_evaluation.py is in:
# FINBENCH_DEMO/Gradio/Tabs/tab_evaluation.py
# so    parents[0] = Tabs
#       parents[1] = Gradio
#       parents[2] = FINBENCH_DEMO
PROJECT_ROOT = CURRENT_FILE.parents[2]

FINBENCH_RESULTS_ROOT = PROJECT_ROOT

EVALUATION_DIR = PROJECT_ROOT / "Evaluation"
METRICS_EXTRACTION_SCRIPT = EVALUATION_DIR / "metrics_extraction.py"

SEED = 42

EVAL_MODES = ["EOY", "METRICS", "QUINTILE"]

EVAL_TO_CMD = {
    "EOY": "eoy",
    "METRICS": "metrics",
    "QUINTILE": "quintile",
}

YEAR_DIR_RE = re.compile(
    r"^y(?P<year_start>\d{4})_y(?P<year_end>\d{4})$"
)

PKL_FILE_RE = re.compile(
    r"^results_sl(?P<sl>\d+)_pl(?P<pl>\d+)\.pkl$"
)

TOP_SHORT_PAIRS_BY_UNIVERSE = {
    "dji": [(5, 0), (5, 5)],
    "nasdaq100": [(5, 0), (5, 5)],
    "sx5e": [(5, 0), (5, 5)],
    "sxxp": [(10, 0), (10, 10)],
    "sp500": [(10, 0), (10, 10)],
}

# Helper grafici
def format_length_value(value):
    try:
        value = float(value)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except Exception:
        return str(value)


def make_length_label(sl, pl):
    return f"SL={format_length_value(sl)} | PL={format_length_value(pl)}"


def collect_unique_legend_items(axes):
    handles = []
    labels = []

    for ax in axes.flatten():
        ax_handles, ax_labels = ax.get_legend_handles_labels()

        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    return handles, labels


def style_evaluation_axis(ax):
    ax.set_facecolor("#fbfdff")

    ax.grid(
        True,
        axis="y",
        color="#e0e7ff",
        linewidth=0.8,
        alpha=0.75,
    )

    ax.grid(
        False,
        axis="x",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")

    ax.tick_params(
        axis="both",
        labelsize=8,
        colors="#334155",
    )

    ax.axhline(
        0,
        color="#94a3b8",
        linewidth=0.9,
        alpha=0.8,
        zorder=1,
    )


# -------------------------------------------------
# A list of the only PKLs that are actually available
# -------------------------------------------------
def scan_available_pkls():
    """
    Read only exsisted i .pkl with structure:

    /FINBENCH_DEMO/
        <type>/<model>/results/<universe>/<configuration>/SEED/
        y<start_year>_y<end_year>/
        results_sl<sl>_pl<pl>.pkl
    """

    records = []

    if not FINBENCH_RESULTS_ROOT.exists():
        return records

    pattern = FINBENCH_RESULTS_ROOT.glob(f"*/*/results/*/*/{SEED}/y*_y*/results_sl*_pl*.pkl")

    for pkl_path in pattern:
        try:
            rel = pkl_path.relative_to(FINBENCH_RESULTS_ROOT)
            parts = rel.parts

            # <type>/<model>/results/<universe>/<configuration>/SEED/<year_dir>/<file.pkl>
            if len(parts) != 8:
                continue

            type = parts[0]
            model = parts[1]
            results_literal = parts[2]
            universe = parts[3]
            configuration = parts[4]
            seed = parts[5]
            year_dir = parts[6]
            filename = parts[7]

            if results_literal != "results":
                continue

            if seed != str(SEED):
                continue

            year_match = YEAR_DIR_RE.match(year_dir)
            file_match = PKL_FILE_RE.match(filename)

            if not year_match or not file_match:
                continue

            records.append(
                {
                    "task_type": type,
                    "model": model,
                    "universe": universe,
                    "configuration": configuration,
                    "year_start": int(year_match.group("year_start")),
                    "year_end": int(year_match.group("year_end")),
                    "sequence_length": int(file_match.group("sl")),
                    "prediction_length": int(file_match.group("pl")),
                    "pkl_path": str(pkl_path),
                }
            )

        except Exception:
            continue

    records.sort(
        key=lambda r: (
            r["task_type"],
            r["model"],
            r["universe"],
            r["configuration"],
            r["year_start"],
            r["year_end"],
            r["sequence_length"],
            r["prediction_length"],
        )
    )

    return records

MULTISELECT_FIELDS = {
    "model",
    "configuration",
    "length_pair",
}


def as_list(value):
    if value is None or value == "":
        return []

    if isinstance(value, (list, tuple, set)):
        return [
            v for v in value
            if v is not None and v != ""
        ]

    return [value]


def filter_records(records, **filters):
    filtered = records

    for key, value in filters.items():
        values = as_list(value)

        if not values:
            continue

        allowed = {str(v) for v in values}

        filtered = [
            r for r in filtered
            if str(r[key]) in allowed
        ]

    return filtered


def unique_values(records, key):
    return sorted({r[key] for r in records})


def value_if_valid(value, choices):
    if value in choices:
        return value

    if value is not None and str(value) in [str(c) for c in choices]:
        for c in choices:
            if str(c) == str(value):
                return c

    return None


def values_if_valid(values, choices):
    selected = as_list(values)

    by_string = {
        str(choice): choice
        for choice in choices
    }

    return [
        by_string[str(value)]
        for value in selected
        if str(value) in by_string
    ]


def is_not_empty(value):
    return len(as_list(value)) > 0

def format_length_pair(sequence_length, prediction_length):
    return f"SL={int(sequence_length)} | PL={int(prediction_length)}"


def parse_length_pair(value):
    """
    Parse strings like:
    SL=8 | PL=10
    """
    if value is None or value == "":
        return None

    match = re.search(
        r"SL\s*=\s*(?P<sl>\d+)\s*\|\s*PL\s*=\s*(?P<pl>\d+)",
        str(value),
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return int(match.group("sl")), int(match.group("pl"))


def unique_length_pair_choices(records):
    pairs = sorted(
        {
            (
                int(r["sequence_length"]),
                int(r["prediction_length"]),
            )
            for r in records
        }
    )

    return [
        format_length_pair(sl, pl)
        for sl, pl in pairs
    ]


def filter_records_by_length_pairs(records, length_pairs):
    selected_pairs = set()

    for value in as_list(length_pairs):
        parsed = parse_length_pair(value)

        if parsed is not None:
            selected_pairs.add(parsed)

    if not selected_pairs:
        return records

    return [
        r for r in records
        if (
            int(r["sequence_length"]),
            int(r["prediction_length"]),
        ) in selected_pairs
    ]

# -------------------------------------------------
# Dropdown a cascata
# -------------------------------------------------
FIELD_ORDER = [
    "task_type",
    "model",
    "universe",
    "configuration",
    "year_start",
    "year_end",
    "length_pair",
    "top_k",
    "short_k",
]

def empty_value_for_field(field):
    if field in MULTISELECT_FIELDS:
        return []
    return None


def reset_downstream_values(changed_field, values):
    """
    When a field changes, all subsequent fields are reset.
    This prevents old or invalid combinations.
    """

    if changed_field == "eval_mode":
        for field in FIELD_ORDER:
            values[field] = empty_value_for_field(field)
        return values

    if changed_field not in FIELD_ORDER:
        return values

    changed_idx = FIELD_ORDER.index(changed_field)

    for field in FIELD_ORDER[changed_idx + 1:]:
        values[field] = empty_value_for_field(field)

    return values

def get_top_short_choices(universe, selected_top_k=None):
    pairs = TOP_SHORT_PAIRS_BY_UNIVERSE.get(str(universe).lower(), [])

    top_k_choices = sorted({top for top, _ in pairs})

    if selected_top_k is None:
        short_k_choices = []
    else:
        short_k_choices = sorted(
            {
                short
                for top, short in pairs
                if int(top) == int(selected_top_k)
            }
        )

    return top_k_choices, short_k_choices

def selected_top_short_pairs_for_universe(universe, top_k, short_k):
    selected_pairs = {
        (int(top), int(short))
        for top in as_list(top_k)
        for short in as_list(short_k)
    }

    available_pairs = set(
        TOP_SHORT_PAIRS_BY_UNIVERSE.get(str(universe).lower(), [])
    )

    return sorted(selected_pairs & available_pairs)

def update_eval_form(
    eval_mode,
    task_type,
    model,
    universe,
    configuration,
    year_start,
    year_end,
    length_pair,
    top_k,
    short_k,
    changed_field,
):
    records = scan_available_pkls()

    values = {
        "task_type": task_type,
        "model": model,
        "universe": universe,
        "configuration": configuration,
        "year_start": year_start,
        "year_end": year_end,
        "length_pair": length_pair,
        "top_k": top_k,
        "short_k": short_k,
    }

    values = reset_downstream_values(changed_field, values)

    form_visible = eval_mode in EVAL_MODES
    top_short_visible = eval_mode in ["EOY", "METRICS"]

    if not form_visible:
        return [
            gr.update(visible=False),

            gr.update(choices=[], value=None, interactive=False),  # task_type
            gr.update(choices=[], value=[], interactive=False),    # model
            gr.update(choices=[], value=None, interactive=False),  # universe
            gr.update(choices=[], value=[], interactive=False),    # configuration
            gr.update(choices=[], value=None, interactive=False),  # year_start
            gr.update(choices=[], value=None, interactive=False),  # year_end
            gr.update(choices=[], value=[], interactive=False),    # length_pair

            gr.update(visible=False),
            gr.update(choices=[], value=None, interactive=False),  # top_k
            gr.update(choices=[], value=None, interactive=False),  # short_k

            gr.update(interactive=False),
        ]

    # 1. Task type
    task_type_choices = unique_values(records, "task_type")
    values["task_type"] = value_if_valid(
        values["task_type"],
        task_type_choices,
    )

    # 2. Models
    model_records = filter_records(
        records,
        task_type=values["task_type"],
    )

    model_choices = (
        unique_values(model_records, "model")
        if is_not_empty(values["task_type"])
        else []
    )

    values["model"] = values_if_valid(
        values["model"],
        model_choices,
    )

    # 3. Universe
    universe_records = filter_records(
        records,
        task_type=values["task_type"],
        model=values["model"],
    )

    universe_choices = (
        unique_values(universe_records, "universe")
        if is_not_empty(values["model"])
        else []
    )

    values["universe"] = value_if_valid(
        values["universe"],
        universe_choices,
    )

    # 4. Configuration
    configuration_records = filter_records(
        records,
        task_type=values["task_type"],
        model=values["model"],
        universe=values["universe"],
    )

    configuration_choices = (
        unique_values(configuration_records, "configuration")
        if is_not_empty(values["universe"])
        else []
    )

    values["configuration"] = values_if_valid(
        values["configuration"],
        configuration_choices,
    )

    # 5. Year start
    year_start_records = filter_records(
        records,
        task_type=values["task_type"],
        model=values["model"],
        universe=values["universe"],
        configuration=values["configuration"],
    )

    year_start_choices = (
        unique_values(year_start_records, "year_start")
        if is_not_empty(values["configuration"])
        else []
    )

    values["year_start"] = value_if_valid(
        values["year_start"],
        year_start_choices,
    )

    # 6. Year end
    year_end_records = filter_records(
        records,
        task_type=values["task_type"],
        model=values["model"],
        universe=values["universe"],
        configuration=values["configuration"],
        year_start=values["year_start"],
    )

    year_end_choices = (
        unique_values(year_end_records, "year_end")
        if is_not_empty(values["year_start"])
        else []
    )

    values["year_end"] = value_if_valid(
        values["year_end"],
        year_end_choices,
    )

    # 7. Sequence / prediction length combinations
    length_pair_records = filter_records(
        records,
        task_type=values["task_type"],
        model=values["model"],
        universe=values["universe"],
        configuration=values["configuration"],
        year_start=values["year_start"],
        year_end=values["year_end"],
    )

    length_pair_choices = (
        unique_length_pair_choices(length_pair_records)
        if is_not_empty(values["year_end"])
        else []
    )

    values["length_pair"] = values_if_valid(
        values["length_pair"],
        length_pair_choices,
    )

    # 8. Top K / Short K
    top_k_choices = []
    short_k_choices = []

    if top_short_visible and is_not_empty(values["length_pair"]):
        top_k_choices, short_k_choices = get_top_short_choices(
            values["universe"],
            values["top_k"],
        )

        values["top_k"] = value_if_valid(
            values["top_k"],
            top_k_choices,
        )

        if is_not_empty(values["top_k"]):
            _, short_k_choices = get_top_short_choices(
                values["universe"],
                values["top_k"],
            )

            values["short_k"] = value_if_valid(
                values["short_k"],
                short_k_choices,
            )
        else:
            values["short_k"] = None
            short_k_choices = []
    else:
        values["top_k"] = None
        values["short_k"] = None

    base_complete = all(
        is_not_empty(values[field])
        for field in [
            "task_type",
            "model",
            "universe",
            "configuration",
            "year_start",
            "year_end",
            "length_pair",
        ]
    )

    if eval_mode in ["EOY", "METRICS"]:
        complete = (
            base_complete
            and is_not_empty(values["top_k"])
            and is_not_empty(values["short_k"])
        )
    else:
        complete = base_complete

    return [
        gr.update(visible=True),

        gr.update(
            choices=task_type_choices,
            value=values["task_type"],
            interactive=bool(task_type_choices),
        ),
        gr.update(
            choices=model_choices,
            value=values["model"],
            interactive=is_not_empty(values["task_type"]) and bool(model_choices),
        ),
        gr.update(
            choices=universe_choices,
            value=values["universe"],
            interactive=is_not_empty(values["model"]) and bool(universe_choices),
        ),
        gr.update(
            choices=configuration_choices,
            value=values["configuration"],
            interactive=is_not_empty(values["universe"]) and bool(configuration_choices),
        ),
        gr.update(
            choices=year_start_choices,
            value=values["year_start"],
            interactive=is_not_empty(values["configuration"]) and bool(year_start_choices),
        ),
        gr.update(
            choices=year_end_choices,
            value=values["year_end"],
            interactive=is_not_empty(values["year_start"]) and bool(year_end_choices),
        ),
        gr.update(
            choices=length_pair_choices,
            value=values["length_pair"],
            interactive=is_not_empty(values["year_end"]) and bool(length_pair_choices),
        ),

        gr.update(visible=top_short_visible),

        gr.update(
            choices=top_k_choices,
            value=values["top_k"],
            interactive=(
                top_short_visible
                and is_not_empty(values["length_pair"])
                and bool(top_k_choices)
            ),
        ),
        gr.update(
            choices=short_k_choices,
            value=values["short_k"],
            interactive=(
                top_short_visible
                and is_not_empty(values["top_k"])
                and bool(short_k_choices)
            ),
        ),

        gr.update(interactive=complete),
    ]

# -------------------------------------------------
# Run script + read output CSV
# -------------------------------------------------
def output_csv_path(eval_mode, top_k=None, short_k=None):
    if eval_mode == "EOY":
        return EVALUATION_DIR / f"aggregated_eoy_returns_{int(top_k)}_{int(short_k)}.csv"

    if eval_mode == "METRICS":
        return EVALUATION_DIR / "evaluation_result_summary_metrics_mean_over_seeds.csv"

    if eval_mode == "QUINTILE":
        return EVALUATION_DIR / "evaluation_result_summary_metrics_quintile_mean_over_seeds.csv"

    raise ValueError(f"Evaluation mode not valid: {eval_mode}")


def normalize_column_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_column(df, aliases):
    normalized = {
        normalize_column_name(col): col
        for col in df.columns
    }

    for alias in aliases:
        key = normalize_column_name(alias)

        if key in normalized:
            return normalized[key]

    return None


def filter_output_dataframe(
    df,
    eval_mode,
    task_type,
    model,
    universe,
    configuration,
    year_start,
    year_end,
    sequence_length,
    prediction_length,
    top_k,
    short_k,
):
    def apply_period_filter(filtered_df, start_year, end_year):
        start_year = str(start_year)
        end_year = str(end_year)

        start_col = find_column(
            filtered_df,
            ["initial_year", "start_year", "year_start", "initialyear"],
        )

        end_col = find_column(
            filtered_df,
            ["end_year", "final_year", "year_end", "endyear"],
        )

        if start_col is not None and end_col is not None:
            start_values = pd.to_numeric(filtered_df[start_col], errors="coerce")
            end_values = pd.to_numeric(filtered_df[end_col], errors="coerce")

            mask = (
                (start_values == int(start_year))
                & (end_values == int(end_year))
            )

            return (
                filtered_df[mask],
                [f"{start_col}={start_year}", f"{end_col}={end_year}"],
                [],
            )

        period_col = find_column(
            filtered_df,
            [
                "period",
                "periodo",
                "year_range",
                "years",
                "window",
                "date_range",
                "evaluation_period",
            ],
        )

        if period_col is None:
            return (
                filtered_df,
                [],
                ["period / year_start-year_end"],
            )

        period_values = (
            filtered_df[period_col]
            .astype(str)
            .str.lower()
            .str.replace(" ", "", regex=False)
        )

        extracted = period_values.str.extract(r"(?P<start>\d{4}).*?(?P<end>\d{4})")

        mask = (
                pd.to_numeric(extracted["start"], errors="coerce").eq(int(start_year))
                & pd.to_numeric(extracted["end"], errors="coerce").eq(int(end_year))
        )

        return (
            filtered_df[mask],
            [f"{period_col}={start_year}-{end_year}"],
            [],
        )

    filters = [
        (task_type, ["type", "task_type"]),
        (model, ["model"]),
        (universe, ["universe"]),
        (configuration, ["configuration", "config"]),
        (sequence_length, ["sl", "sequence_length"]),
        (prediction_length, ["pl", "prediction_length"]),
    ]

    if eval_mode in ["EOY", "METRICS"]:
        filters.extend(
            [
                (top_k, ["top_k", "topk"]),
                (short_k, ["short_k", "shortk"]),
            ]
        )

    filtered = df.copy()
    applied = []
    missing = []

    for value, aliases in filters:
        if value is None or value == "":
            continue

        col = find_column(filtered, aliases)

        if col is None:
            missing.append("/".join(aliases))
            continue

        numeric_value = None

        try:
            numeric_value = int(value)
        except Exception:
            pass

        if numeric_value is not None:
            col_as_num = pd.to_numeric(filtered[col], errors="coerce")
            mask = col_as_num == numeric_value
        else:
            mask = filtered[col].astype(str).str.strip() == str(value).strip()

        filtered = filtered[mask]
        applied.append(f"{col}={value}")

    if year_start is not None and year_end is not None:
        filtered, period_applied, period_missing = apply_period_filter(
            filtered,
            year_start,
            year_end,
        )

        applied.extend(period_applied)
        missing.extend(period_missing)

    return filtered, applied, missing

# -------------------------------------------------
# Plot output dataframe
# -------------------------------------------------
def get_first_column(df, column_name):
    """
    Pandas returns a DataFrame instead of a Series if duplicate column names exist.
    This helper always returns the first matching Series.
    """
    values = df.loc[:, column_name]

    if isinstance(values, pd.DataFrame):
        return values.iloc[:, 0]

    return values


def find_plot_column(df, aliases):
    return find_column(df, aliases)


def sort_mixed(values):
    def sort_key(value):
        try:
            return (0, float(value))
        except Exception:
            return (1, str(value))

    return sorted(values, key=sort_key)


def safe_numeric_series(series):
    return pd.to_numeric(series, errors="coerce")


def build_plot_base_dataframe(df):
    universe_col = find_plot_column(
        df,
        ["selected_universe", "universe"],
    )

    model_col = find_plot_column(
        df,
        ["selected_model", "model"],
    )

    config_col = find_plot_column(
        df,
        ["selected_configuration", "configuration", "config"],
    )

    task_type_col = find_plot_column(
        df,
        ["selected_task_type", "task_type", "type"],
    )

    sl_col = find_plot_column(
        df,
        ["selected_sequence_length", "sequence_length", "sl"],
    )

    pl_col = find_plot_column(
        df,
        ["selected_prediction_length", "prediction_length", "pl"],
    )

    if universe_col is None or model_col is None:
        return None

    plot_df = pd.DataFrame(
        {
            "universe": get_first_column(df, universe_col).astype(str),
            "model": get_first_column(df, model_col).astype(str),
        }
    )

    if config_col is not None:
        plot_df["config"] = get_first_column(df, config_col).astype(str)
    else:
        plot_df["config"] = ""

    if task_type_col is not None:
        plot_df["task_type"] = get_first_column(df, task_type_col).astype(str)
    else:
        plot_df["task_type"] = ""

    if sl_col is not None:
        plot_df["sl"] = get_first_column(df, sl_col)
    else:
        plot_df["sl"] = ""

    if pl_col is not None:
        plot_df["pl"] = get_first_column(df, pl_col)
    else:
        plot_df["pl"] = ""

    plot_df["length_label"] = [
        make_length_label(sl, pl)
        for sl, pl in zip(plot_df["sl"], plot_df["pl"])
    ]

    # If the same model has multiple configs, show config in the column title.
    model_config_counts = (
        plot_df[["model", "config"]]
        .drop_duplicates()
        .groupby("model")
        .size()
    )

    show_config = (
        "config" in plot_df.columns
        and not model_config_counts.empty
        and model_config_counts.max() > 1
    )

    if show_config:
        plot_df["facet_model"] = plot_df["model"] + "\n" + plot_df["config"]
    else:
        plot_df["facet_model"] = plot_df["model"]

    return plot_df

def compute_shared_y_limits(values, include_zero=True, padding_ratio=0.08):
    """
    Compute shared y-axis limits for all subplots.
    """
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

def build_label_color_map(labels):
    """
    Assign one stable color to each label across all subplots.
    """
    labels = [str(label) for label in labels]
    palette = [
        "#5b359c",  # violet
        "#ec4899",  # pink
        "#14b8a6",  # teal
        "#0ea5e9",  # sky
        "#10b981",  # emerald
        "#f59e0b",  # amber
        "#4f46e5",  # indigo
        "#ef4444",  # red
        "#6366f1",  # soft indigo
        "#8b5cf6",  # soft violet
    ]
    return {
        label: palette[idx % len(palette)]
        for idx, label in enumerate(labels)
    }

def build_plot_facets(plot_df):
    """
    Build only the non-empty subplot combinations.
    Each facet is one (universe, facet_model) plot.
    """
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


def create_fixed_facet_grid(
    facets,
    subplot_width,
    subplot_height,
    max_columns=MAX_PLOT_COLUMNS,
    dpi=130,
):
    """
    Create a fixed subplot grid with at most max_columns plots per row.
    """
    n_facets = max(len(facets), 1)
    n_cols = min(max_columns, n_facets)
    n_rows = int(np.ceil(n_facets / n_cols))

    fig_width = subplot_width * n_cols
    fig_height = subplot_height * n_rows + 1.25

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        dpi=dpi,
    )

    return fig, axes, n_rows, n_cols

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

    plot_df = plot_df.dropna(
        subset=["q1", "q2", "q3", "q4", "q5"],
        how="all",
    )

    if plot_df.empty:
        return None

    shared_ylim = compute_shared_y_limits(
        plot_df[["q1", "q2", "q3", "q4", "q5"]].to_numpy(),
        include_zero=True,
    )

    facets = build_plot_facets(plot_df)

    if not facets:
        return None

    length_labels = sort_mixed(plot_df["length_label"].dropna().unique())
    length_color_map = build_label_color_map(length_labels)

    fig, axes, n_rows, n_cols = create_fixed_facet_grid(
        facets=facets,
        subplot_width=FIXED_SUBPLOT_WIDTH,
        subplot_height=FIXED_SUBPLOT_HEIGHT_QUINTILE,
        max_columns=MAX_PLOT_COLUMNS,
        dpi=130,
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
            length_subset = subset[
                subset["length_label"].astype(str) == str(length_label)
            ]

            if length_subset.empty:
                continue

            values = (
                length_subset[["q1", "q2", "q3", "q4", "q5"]]
                .mean(axis=0, numeric_only=True)
                .values
            )

            ax.plot(
                x,
                values,
                marker="o",
                linewidth=1.8,
                markersize=4.2,
                label=str(length_label),
                color=length_color_map[str(length_label)],
                zorder=3,
            )

        style_evaluation_axis(ax)
        apply_shared_y_axis(ax, shared_ylim)

        ax.set_title(
            f"{model_name}",
            fontsize=10.5,
            fontweight="bold",
            color="#1e1b4b",
            pad=8,
        )

        if col_idx == 0:
            ax.set_ylabel(
                str(universe),
                fontsize=9.5,
                fontweight="bold",
                color="#1e1b4b",
                labelpad=8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)

    for ax in axes.flatten()[len(facets):]:
        ax.axis("off")

    handles, labels = collect_unique_legend_items(axes)

    if handles:
        legend = fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 4),
            frameon=True,
            fontsize=8.5,
            bbox_to_anchor=(0.5, 0.02),
        )

        legend.get_frame().set_facecolor("#ffffff")
        legend.get_frame().set_edgecolor("#c7d2fe")
        legend.get_frame().set_linewidth(0.9)

    fig.suptitle(
        "Quintile analysis",
        fontsize=14,
        fontweight="bold",
        color="#111827",
        y=0.985,
    )

    fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.93])

    return fig


def build_eoy_plot(df):
    if df is None or df.empty:
        return None

    year_columns = [
        col for col in df.columns
        if re.match(r"^\d{4}$", str(col))
    ]

    if not year_columns:
        return None

    year_columns = sorted(year_columns, key=lambda x: int(str(x)))

    base = build_plot_base_dataframe(df)

    if base is None:
        return None

    plot_df = base.copy()

    for year_col in year_columns:
        plot_df[str(year_col)] = safe_numeric_series(
            get_first_column(df, year_col)
        )

    plot_df = plot_df.dropna(
        subset=[str(col) for col in year_columns],
        how="all",
    )

    if plot_df.empty:
        return None

    benchmark_mask = pd.Series(False, index=plot_df.index)

    for col in ["model", "config", "task_type"]:
        benchmark_mask = benchmark_mask | (
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

    long_strategy = strategy_df.melt(
        id_vars=[
            "universe",
            "model",
            "config",
            "task_type",
            "sl",
            "pl",
            "length_label",
            "facet_model",
        ],
        value_vars=[str(col) for col in year_columns],
        var_name="year",
        value_name="return",
    )

    long_strategy["return"] = safe_numeric_series(long_strategy["return"])
    long_strategy = long_strategy.dropna(subset=["return"])

    if long_strategy.empty:
        return None

    if not benchmark_df.empty:
        long_benchmark = benchmark_df.melt(
            id_vars=[
                "universe",
                "model",
                "config",
                "task_type",
                "sl",
                "pl",
                "length_label",
                "facet_model",
            ],
            value_vars=[str(col) for col in year_columns],
            var_name="year",
            value_name="return",
        )

        long_benchmark["return"] = safe_numeric_series(long_benchmark["return"])
        long_benchmark = long_benchmark.dropna(subset=["return"])
    else:
        long_benchmark = pd.DataFrame()

    shared_y_values = [long_strategy["return"]]

    if not long_benchmark.empty:
        shared_y_values.append(long_benchmark["return"])

    shared_ylim = compute_shared_y_limits(
        pd.concat(shared_y_values, ignore_index=True),
        include_zero=True,
    )

    facets = build_plot_facets(long_strategy)

    if not facets:
        return None

    horizons = sort_mixed(long_strategy["length_label"].dropna().unique())
    years = [str(col) for col in year_columns]
    horizon_color_map = build_label_color_map(horizons)

    fig, axes, n_rows, n_cols = create_fixed_facet_grid(
        facets=facets,
        subplot_width=FIXED_SUBPLOT_WIDTH,
        subplot_height=FIXED_SUBPLOT_HEIGHT_EOY,
        max_columns=MAX_PLOT_COLUMNS,
        dpi=130,
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
            horizon_subset = subset[
                subset["length_label"].astype(str) == str(horizon)
            ]

            if horizon_subset.empty:
                continue

            values_by_year = (
                horizon_subset
                .groupby("year")["return"]
                .mean()
                .reindex(years)
            )

            offset = (
                horizon_idx - (n_horizons - 1) / 2
            ) * bar_width

            ax.bar(
                x + offset,
                values_by_year.values,
                width=bar_width,
                label=str(horizon),
                color=horizon_color_map[str(horizon)],
            )

        if not long_benchmark.empty:
            bench_subset = long_benchmark[
                long_benchmark["universe"] == universe
            ]

            if not bench_subset.empty:
                benchmark_by_year = (
                    bench_subset
                    .groupby("year")["return"]
                    .mean()
                    .reindex(years)
                )

                ax.plot(
                    x,
                    benchmark_by_year.values,
                    marker="o",
                    linewidth=1.35,
                    markersize=3.4,
                    label="Benchmark",
                    color="#111827",
                    linestyle="--",
                    zorder=4,
                )

        style_evaluation_axis(ax)
        apply_shared_y_axis(ax, shared_ylim)

        ax.set_title(
            f"{model_name}",
            fontsize=10.5,
            fontweight="bold",
            color="#1e1b4b",
            pad=8,
        )

        if col_idx == 0:
            ax.set_ylabel(
                str(universe),
                fontsize=9.5,
                fontweight="bold",
                color="#1e1b4b",
                labelpad=8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(years, rotation=0)

        ax.tick_params(axis="both", labelsize=8)

    for ax in axes.flatten()[len(facets):]:
        ax.axis("off")

    handles, labels = collect_unique_legend_items(axes)

    if handles:
        legend = fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 4),
            frameon=True,
            fontsize=8.5,
            bbox_to_anchor=(0.5, 0.02),
        )

        legend.get_frame().set_facecolor("#ffffff")
        legend.get_frame().set_edgecolor("#c7d2fe")
        legend.get_frame().set_linewidth(0.9)

    fig.suptitle(
        "End-of-Year returns",
        fontsize=13,
        fontweight="bold",
        color="#111827",
        y=0.985,
    )

    fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.93])

    return fig


def build_evaluation_plot(df, eval_mode):
    if eval_mode == "QUINTILE":
        return build_quintile_plot(df)

    if eval_mode == "EOY":
        return build_eoy_plot(df)

    return None

def empty_plot():
    return gr.update(value=None, visible=False)


def matplotlib_plot_update(fig):
    if fig is None:
        return gr.update(value=None, visible=False)

    return gr.update(value=fig, visible=True)


def show_evaluation_loading():
    return (
        gr.update(visible=True),              # table_loading
        gr.update(visible=True),              # plot_loading
        gr.update(interactive=False),         # run_btn
        gr.update(value=None, visible=False), # download_evaluation_button
        gr.update(visible=False, interactive=False), #explain_evaluation_button
        gr.update(visible=False),             # chatgpt_explanation_group
        "",                                   # chatgpt_explanation
    )


def hide_evaluation_loading():
    return (
        gr.update(visible=False),       # table_loading
        gr.update(visible=False),       # plot_loading
        gr.update(interactive=True),    # run_btn
        gr.update(visible=True, interactive=True),  # explain_evaluation_button
    )

def show_chatgpt_explanation_loading():
    return (
        gr.update(visible=True),                     # chatgpt_loading
        gr.update(visible=True, interactive=False),  # explain_evaluation_button
        gr.update(visible=False),                    # chatgpt_explanation_group
        "",                                          # chatgpt_explanation
    )


def hide_chatgpt_explanation_loading():
    return (
        gr.update(visible=False),                    # chatgpt_loading
        gr.update(visible=True, interactive=True),   # explain_evaluation_button
        gr.update(visible=True),                     # chatgpt_explanation_group
    )

def selected_records_for_form(
    task_type,
    model,
    universe,
    configuration,
    year_start,
    year_end,
    length_pair,
):
    records = scan_available_pkls()

    filtered = filter_records(
        records,
        task_type=task_type,
        model=model,
        universe=universe,
        configuration=configuration,
        year_start=year_start,
        year_end=year_end,
    )

    return filter_records_by_length_pairs(
        filtered,
        length_pair,
    )

def build_eval_command(
    eval_mode,
    record,
    top_k=None,
    short_k=None,
):
    cmd_name = EVAL_TO_CMD[eval_mode]

    cmd = [
        sys.executable,
        str(METRICS_EXTRACTION_SCRIPT),
        cmd_name,
        "--universe",
        str(record["universe"]),
        "--model",
        str(record["model"]),
        "--type",
        str(record["task_type"]),
        "--initial_year",
        str(int(record["year_start"])),
        "--end_year",
        str(int(record["year_end"])),
        "--seed",
        str(SEED),
        "--sl",
        str(int(record["sequence_length"])),
        "--pl",
        str(int(record["prediction_length"])),
        "--results_root",
        str(FINBENCH_RESULTS_ROOT),
    ]

    if eval_mode == "EOY":
        cmd.extend(
            [
                "--top_k",
                str(int(top_k)),
                "--short_k",
                str(int(short_k)),
                "--base_dir",
                str(EVALUATION_DIR / "evaluation_result"),
                "--output_prefix",
                str(EVALUATION_DIR / "aggregated_eoy_returns"),
            ]
        )

    elif eval_mode == "METRICS":
        cmd.extend(
            [
                "--top_k",
                str(int(top_k)),
                "--short_k",
                str(int(short_k)),
                "--root_dir",
                str(EVALUATION_DIR / "evaluation_result"),
                "--mean_output",
                str(EVALUATION_DIR / "evaluation_result_summary_metrics_mean_over_seeds.csv"),
            ]
        )

    elif eval_mode == "QUINTILE":
        cmd.extend(
            [
                "--root_dir",
                str(EVALUATION_DIR / "evaluation_result_quintile"),
                "--mean_output",
                str(EVALUATION_DIR / "evaluation_result_summary_metrics_quintile_mean_over_seeds.csv"),
            ]
        )

    return cmd


def add_selection_context(
    df,
    record,
    eval_mode,
    top_k=None,
    short_k=None,
):
    enriched = df.copy()

    context = {
        "selected_task_type": record["task_type"],
        "selected_model": record["model"],
        "selected_universe": record["universe"],
        "selected_configuration": record["configuration"],
        "selected_year_start": record["year_start"],
        "selected_year_end": record["year_end"],
        "selected_sequence_length": record["sequence_length"],
        "selected_prediction_length": record["prediction_length"],
    }

    if eval_mode in ["EOY", "METRICS"]:
        context["selected_top_k"] = top_k
        context["selected_short_k"] = short_k

    for col, value in reversed(list(context.items())):
        enriched.insert(0, col, value)

    return enriched

def humanize_column_name(name):
    raw = str(name).strip()

    if raw.startswith("selected_"):
        raw = raw[len("selected_"):]

    aliases = {
        "task_type": "Task type",
        "type": "Task type",

        "model": "Model",
        "selected_model": "Model",

        "universe": "Universe",
        "selected_universe": "Universe",

        "configuration": "Configuration",
        "config": "Configuration",

        "year_start": "Year start",
        "initial_year": "Year start",
        "initialyear": "Year start",

        "year_end": "Year end",
        "end_year": "Year end",
        "endyear": "Year end",
        "final_year": "Year end",

        "sequence_length": "Sequence length",
        "sl": "Sequence length",

        "prediction_length": "Prediction length",
        "pl": "Prediction length",

        "top_k": "Top K",
        "topk": "Top K",

        "short_k": "Short K",
        "shortk": "Short K",

        "period": "Period",

        "q1": "Q1",
        "q2": "Q2",
        "q3": "Q3",
        "q4": "Q4",
        "q5": "Q5",
    }

    normalized = normalize_column_name(raw)

    if normalized in aliases:
        return aliases[normalized]

    if re.match(r"^\d{4}$", raw):
        return raw

    return raw.replace("_", " ").strip().capitalize()


def prettify_output_dataframe(df):
    if df is None or df.empty:
        return pd.DataFrame()

    pretty_df = df.copy()

    pretty_df.columns = [
        humanize_column_name(col)
        for col in pretty_df.columns
    ]

    # Keep the first occurrence of duplicated human-readable columns.
    # Since selected_* columns are inserted first, this preserves the UI context
    # and removes duplicated metadata coming from the generated CSV.
    pretty_df = pretty_df.loc[
        :,
        ~pd.Index(pretty_df.columns).duplicated(keep="first")
    ]

    return pretty_df


def empty_evaluation_table_html(message="No evaluation output yet."):
    return f"""
    <div class="metrics-table-wrap evaluation-metrics-table-wrap">
        <div class="metrics-empty">
            <div>{escape(message)}</div>
        </div>
    </div>
    """


def dataframe_to_metrics_html(df):
    pretty_df = prettify_output_dataframe(df)

    if pretty_df.empty:
        return empty_evaluation_table_html("No rows to display for this selection.")

    table_html = pretty_df.to_html(
        index=False,
        border=0,
        classes="mpt",
        escape=True,
        na_rep="",
    )

    return f"""
    <div class="metrics-table-wrap evaluation-metrics-table-wrap">
        <div class="metrics-outer metrics-scroll">
            {table_html}
        </div>
    </div>
    """


def write_evaluation_download_file(df, eval_mode):
    pretty_df = prettify_output_dataframe(df)

    if pretty_df.empty:
        return None

    output_path = EVALUATION_DIR / f"evaluation_output_{str(eval_mode).lower()}.csv"
    pretty_df.to_csv(output_path, index=False)

    return str(output_path)

def dataframe_to_llm_text(df, max_rows=80):
    pretty_df = prettify_output_dataframe(df)

    if pretty_df.empty:
        return ""

    truncated = pretty_df.head(max_rows).copy()

    csv_text = truncated.to_csv(index=False)

    if len(pretty_df) > max_rows:
        csv_text += (
            f"\n\n[Nota: la tabella originale contiene {len(pretty_df)} righe; "
            f"qui sono mostrate solo le prime {max_rows}.]"
        )

    return csv_text


def explain_evaluation_with_chatgpt(output_df, eval_mode):
    if output_df is None or output_df.empty:
        return "Run an evaluation first, then ask ChatGPT to explain the results."

    table_text = dataframe_to_llm_text(output_df)

    prompt = f"""
You are a financial evaluation analyst.

The user has run a evaluation with mode: {eval_mode}.

Explain the results in English.

Use Markdown formatting. 

Focus on:
1. What the table is showing.
2. Which models/configurations look better.
3. Any relevant differences between sequence length and prediction length.
4. Any relevant universe/configuration effects.

Use a clear, concise explanation suitable for a demo UI.

Here is the evaluation output table in CSV format:

{table_text}
"""

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)

        response = client.responses.create(
            model=OPENAI_EXPLANATION_MODEL,
            input=prompt,
        )

        explanation = response.output_text

        return explanation

    except Exception as e:
        return f"Error while calling ChatGPT: {escape(str(e))}"

def run_evaluation(
    eval_mode,
    task_type,
    model,
    universe,
    configuration,
    year_start,
    year_end,
    length_pair,
    top_k,
    short_k,
):
    selected_records = selected_records_for_form(
        task_type=task_type,
        model=model,
        universe=universe,
        configuration=configuration,
        year_start=year_start,
        year_end=year_end,
        length_pair=length_pair,
    )

    if not selected_records:
        return (
            "The selected values do not match any available .pkl file.",
            empty_evaluation_table_html("The selected values do not match any available .pkl file."),
            gr.update(value=None, visible=False),
            gr.update(visible=False, interactive=False),
            empty_plot(),
            pd.DataFrame()
        )

    jobs = []

    for record in selected_records:
        if eval_mode in ["EOY", "METRICS"]:
            jobs.append(
                {
                    "record": record,
                    "top_k": top_k,
                    "short_k": short_k,
                }
            )
        else:
            jobs.append(
                {
                    "record": record,
                    "top_k": None,
                    "short_k": None,
                }
            )

    if not jobs:
        return (
            "No valid Top K / Short K pair is available for the selected universes.",
            empty_evaluation_table_html("No valid Top K / Short K pair is available for the selected universes."),
            gr.update(value=None, visible=False),
            gr.update(visible=False, interactive=False),
            empty_plot(),
            pd.DataFrame()
        )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    logs = [
        f"Selected .pkl combinations: {len(selected_records)}",
        f"Evaluation jobs to process: {len(jobs)}",
    ]

    frames = []
    command_cache = {}

    for job_idx, job in enumerate(jobs, start=1):
        record = job["record"]
        job_top_k = job["top_k"]
        job_short_k = job["short_k"]

        cmd = build_eval_command(
            eval_mode=eval_mode,
            record=record,
            top_k=job_top_k,
            short_k=job_short_k,
        )

        command_key = tuple(cmd)

        logs.append(
            "\n"
            + "=" * 80
            + f"\nJob {job_idx}/{len(jobs)}"
            + f"\nPKL:\n{record['pkl_path']}"
        )

        if command_key not in command_cache:
            completed = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                capture_output=True,
                text=True,
            )

            command_str = " ".join(shlex.quote(x) for x in cmd)

            logs.append(f"\nCommand executed:\n{command_str}")

            if completed.stdout:
                logs.append("\nSTDOUT:\n" + completed.stdout)

            if completed.stderr:
                logs.append("\nSTDERR:\n" + completed.stderr)

            if completed.returncode != 0:
                logs.append(
                    f"\nError: process terminated with code {completed.returncode}."
                )
                command_cache[command_key] = {
                    "ok": False,
                    "df": pd.DataFrame(),
                    "csv_path": None,
                }
                continue

            csv_path = output_csv_path(
                eval_mode=eval_mode,
                top_k=job_top_k,
                short_k=job_short_k,
            )

            if not csv_path.exists():
                logs.append(f"\nThe expected CSV file was not found:\n{csv_path}")
                command_cache[command_key] = {
                    "ok": False,
                    "df": pd.DataFrame(),
                    "csv_path": csv_path,
                }
                continue

            df = pd.read_csv(csv_path)

            command_cache[command_key] = {
                "ok": True,
                "df": df,
                "csv_path": csv_path,
            }

        else:
            logs.append("\nCommand already executed before; reusing generated CSV.")

        cached = command_cache[command_key]

        if not cached["ok"]:
            continue

        df = cached["df"]
        csv_path = cached["csv_path"]

        filtered_df, applied, missing = filter_output_dataframe(
            df=df,
            eval_mode=eval_mode,
            task_type=record["task_type"],
            model=record["model"],
            universe=record["universe"],
            configuration=record["configuration"],
            year_start=record["year_start"],
            year_end=record["year_end"],
            sequence_length=record["sequence_length"],
            prediction_length=record["prediction_length"],
            top_k=job_top_k,
            short_k=job_short_k,
        )

        logs.append(f"\nCSV file read:\n{csv_path}")
        logs.append(f"Total number of rows in the CSV file: {len(df)}")
        logs.append(f"Rows shown for this job: {len(filtered_df)}")

        if applied:
            logs.append(
                "\nFilters applied:\n"
                + "\n".join(f"- {x}" for x in applied)
            )

        if missing:
            logs.append(
                "\nColumns not found in the CSV file, so not used for filtering:\n"
                + "\n".join(f"- {x}" for x in missing)
            )

        if not filtered_df.empty:
            frames.append(
                add_selection_context(
                    df=filtered_df,
                    record=record,
                    eval_mode=eval_mode,
                    top_k=job_top_k,
                    short_k=job_short_k,
                )
            )

    if frames:
        output_df = pd.concat(frames, ignore_index=True)
        output_df = output_df.drop_duplicates().reset_index(drop=True)
    else:
        output_df = pd.DataFrame()

    logs.append("\n" + "=" * 80)
    logs.append(f"Unique commands executed: {len(command_cache)}")
    logs.append(f"Final rows shown: {len(output_df)}")

    plt.close("all")

    fig = build_evaluation_plot(output_df, eval_mode)
    plot_update = matplotlib_plot_update(fig)

    table_html = dataframe_to_metrics_html(output_df)
    download_file = write_evaluation_download_file(output_df, eval_mode)

    if download_file is None:
        download_update = gr.update(value=None, visible=False)
        gpt_button_update = gr.update(visible=False,interactive=False,)
    else:
        download_update = gr.update(value=download_file, visible=True)
        gpt_button_update = gr.update(
            value="Explain with ChatGPT",
            visible=True,
            interactive=True,
        )

    return "\n".join(logs), table_html, download_update, gpt_button_update, plot_update, output_df

# -------------------------------------------------
# Build tab
# -------------------------------------------------
def build_tab():
    with gr.Tab("Evaluation", elem_id="main-tabs"):
        with gr.Column():

            # ── main two-column row ──────────────────────────────────────────
            with gr.Row(elem_classes=["evaluation-mode-row"]):

                # LEFT – assessment type
                with gr.Column(scale=1, elem_classes=["evaluation-mode-col"]):
                    with gr.Group(elem_classes=["evaluation-mode-card"]):
                        gr.Markdown(
                            """
                            <div class="evaluation-mode-title-wrap">
                                <div class="evaluation-section-title evaluation-mode-title">
                                    <span class="evaluation-section-dot"></span>
                                    Assessment type
                                </div>
                            </div>
                            """
                        )
                        eval_mode = gr.Radio(
                            choices=EVAL_MODES,
                            value=None,
                            label=None,
                            interactive=True,
                            show_label=False,
                            container=False,
                            elem_classes=["evaluation-radio"],
                        )
                    run_btn = gr.Button(
                        "Run evaluation",
                        variant="primary",
                        interactive=False,
                        elem_id="evaluation-run-button",
                    )

                # RIGHT – params group
                with gr.Column(scale=4, elem_classes=["evaluation-params-col"]):
                    with gr.Group(
                        visible=False,
                        elem_classes=["evaluation-panel"],
                    ) as params_group:

                        # ── Result selection ─────────────────────────────────
                        gr.Markdown(
                            """
                            <div class="evaluation-section-title">
                                <span class="evaluation-section-dot"></span>
                                Result selection
                            </div>
                            """
                        )

                        with gr.Row(elem_classes=["evaluation-form-row"]):
                            task_type = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="Task type",
                                interactive=False,
                            )
                            model = gr.Dropdown(
                                choices=[],
                                value=[],
                                label="Model",
                                interactive=False,
                                multiselect=True,
                                elem_classes=["dropdown-multiselect-compact"],
                            )
                            universe = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="Universe",
                                interactive=False,
                            )
                            configuration = gr.Dropdown(
                                choices=[],
                                value=[],
                                label="Configuration",
                                interactive=False,
                                multiselect=True,
                                elem_classes=["dropdown-multiselect-compact"],
                            )

                        # ── Evaluation window ────────────────────────────────
                        gr.Markdown(
                            """
                            <div class="evaluation-section-title evaluation-section-spaced">
                                <span class="evaluation-section-dot"></span>
                                Evaluation window
                            </div>
                            """
                        )

                        with gr.Row(elem_classes=["evaluation-form-row"]):
                            year_start = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="Year start",
                                interactive=False,
                            )
                            year_end = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="Year end",
                                interactive=False,
                            )
                            length_pair = gr.CheckboxGroup(
                                choices=[],
                                value=[],
                                label="Sequence / prediction length",
                                interactive=False,
                                elem_classes=["evaluation-length-checkbox"],
                            )

                        # ── Portfolio construction (conditional) ─────────────
                        with gr.Group(
                            visible=False,
                            elem_classes=["evaluation-portfolio-card"],
                        ) as top_short_row:
                            gr.Markdown(
                                """
                                <div class="evaluation-section-title">
                                    <span class="evaluation-section-dot"></span>
                                    Portfolio construction
                                </div>
                                """
                            )
                            with gr.Row(elem_classes=["evaluation-form-row", "evaluation-portfolio-compact-row"]):
                                with gr.Column(scale=0, min_width=180, elem_classes=["evaluation-portfolio-small-col"]):
                                    top_k = gr.Dropdown(
                                        choices=[],
                                        value=None,
                                        label="Top K",
                                        interactive=False,
                                    )
                                with gr.Column(scale=0, min_width=180, elem_classes=["evaluation-portfolio-small-col"]):
                                    short_k = gr.Dropdown(
                                        choices=[],
                                        value=None,
                                        label="Short K",
                                        interactive=False,
                                    )

            # ── Output area ──────────────────────────────────────────────────
            with gr.Row(elem_classes=["evaluation-output-row"]):
                with gr.Column(
                    scale=1,
                    elem_classes=["evaluation-output-card", "evaluation-log-card"],
                ):
                    gr.Markdown(
                        """
                        <div class="evaluation-output-header">
                            <span>Execution log</span>
                        </div>
                        """
                    )
                    with gr.Accordion(
                        "Show command details",
                        open=False,
                        elem_classes=["evaluation-log-accordion"],
                    ):
                        log_box = gr.Textbox(
                            label=None,
                            interactive=False,
                            lines=12,
                            max_lines=12,
                            show_label=False,
                            elem_id="execution-log",
                        )

                with gr.Column(
                    scale=4,
                    elem_classes=[
                        "evaluation-output-card",
                        "evaluation-table-card",
                        "evaluation-loading-host",
                    ],
                ):
                    gr.Markdown(
                        """
                        <div class="evaluation-output-header">
                            <span>Evaluation output</span>
                        </div>
                        """
                    )
                    table_loading = gr.HTML(
                        """
                        <div class="evaluation-loading-overlay">
                            <div class="evaluation-loading-box">
                                <div class="evaluation-loading-spinner"></div>
                                <div class="evaluation-loading-text">Running evaluation...</div>
                            </div>
                        </div>
                        """,
                        visible=False,
                    )

                    output_table = gr.HTML(
                        value=empty_evaluation_table_html(),
                        elem_id="evaluation-output-table",
                    )
                    with gr.Row(elem_classes=["metrics-download-row", "evaluation-download-row"]):
                        download_evaluation_button = gr.DownloadButton(
                            "Download CSV ⭣",
                            value=None,
                            visible=False,
                            elem_id="download-evaluation-button",
                        )

                        explain_evaluation_button = gr.Button(
                            "Explain with ChatGPT",
                            visible=False,
                            interactive=False,
                            elem_id="explain-evaluation-button",
                        )

                    chatgpt_loading = gr.HTML(
                        """
                        <div class="evaluation-chatgpt-inline-loading">
                            <div class="evaluation-loading-spinner"></div>
                            <div class="evaluation-loading-text">Generating explanation...</div>
                        </div>
                        """,
                        visible=False,
                    )
                    with gr.Group(
                            visible=False,
                            elem_classes=[
                                "evaluation-chatgpt-explanation-card",
                            ],
                    )as chatgpt_explanation_group:
                        chatgpt_explanation = gr.Markdown(
                            value="",
                            elem_id="evaluation-chatgpt-explanation",
                        )
            # ── Plot ─────────────────────────────────────────────────────────
            with gr.Group(
                visible=True,
                elem_classes=[
                    "evaluation-output-card",
                    "evaluation-plot-card",
                    "evaluation-loading-host",
                ],
            ):
                gr.Markdown(
                    """
                    <div class="evaluation-output-header">
                        <span>Evaluation plot</span>
                    </div>
                    """
                )
                plot_loading = gr.HTML(
                    """
                    <div class="evaluation-loading-overlay">
                        <div class="evaluation-loading-box">
                            <div class="evaluation-loading-spinner"></div>
                            <div class="evaluation-loading-text">Building plot...</div>
                        </div>
                    </div>
                    """,
                    visible=False,
                )
                plot_output = gr.Plot(
                    label="",
                    show_label=False,
                    visible=False,
                    container=False,
                    elem_id="evaluation-plot",
                )

        evaluation_output_state = gr.State(value=pd.DataFrame())

        form_inputs = [
            eval_mode,
            task_type,
            model,
            universe,
            configuration,
            year_start,
            year_end,
            length_pair,
            top_k,
            short_k,
        ]

        form_outputs = [
            params_group,
            task_type,
            model,
            universe,
            configuration,
            year_start,
            year_end,
            length_pair,
            top_short_row,
            top_k,
            short_k,
            run_btn,
        ]

        def bind_form_input(component, changed_field):
            component.input(
                fn=partial(update_eval_form, changed_field=changed_field),
                inputs=form_inputs,
                outputs=form_outputs,
                show_progress="hidden",
                queue=False,
            )

        bind_form_input(eval_mode, "eval_mode")
        bind_form_input(task_type, "task_type")
        bind_form_input(model, "model")
        bind_form_input(universe, "universe")
        bind_form_input(configuration, "configuration")
        bind_form_input(year_start, "year_start")
        bind_form_input(year_end, "year_end")
        bind_form_input(length_pair, "length_pair")
        bind_form_input(top_k, "top_k")
        bind_form_input(short_k, "short_k")

        run_event = run_btn.click(
            fn=show_evaluation_loading,
            inputs=[],
            outputs=[
                table_loading,
                plot_loading,
                run_btn,
                download_evaluation_button,
                explain_evaluation_button,
                chatgpt_explanation_group,
                chatgpt_explanation,
            ],
            show_progress="hidden",
            queue=False,
        )

        run_event = run_event.then(
            fn=run_evaluation,
            inputs=form_inputs,
            outputs=[
                log_box,
                output_table,
                download_evaluation_button,
                explain_evaluation_button,
                plot_output,
                evaluation_output_state,

            ],
            show_progress="hidden",
        )

        run_event.then(
            fn=hide_evaluation_loading,
            inputs=[],
            outputs=[
                table_loading,
                plot_loading,
                run_btn,
                explain_evaluation_button
            ],
            show_progress="hidden",
            queue=False,
        )

        explain_event = explain_evaluation_button.click(
            fn=show_chatgpt_explanation_loading,
            inputs=[],
            outputs=[
                chatgpt_loading,
                explain_evaluation_button,
                chatgpt_explanation_group,
                chatgpt_explanation,
            ],
            show_progress="hidden",
            queue=False,
        )

        explain_event = explain_event.then(
            fn=explain_evaluation_with_chatgpt,
            inputs=[
                evaluation_output_state,
                eval_mode,
            ],
            outputs=[
                chatgpt_explanation,
            ],
            show_progress="hidden",
        )

        explain_event.then(
            fn=hide_chatgpt_explanation_loading,
            inputs=[],
            outputs=[
                chatgpt_loading,
                explain_evaluation_button,
                chatgpt_explanation_group,
            ],
            show_progress="hidden",
            queue=False,
        )