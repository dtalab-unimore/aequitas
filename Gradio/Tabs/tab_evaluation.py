import os
import re
import sys
import shlex
import subprocess
from pathlib import Path
from functools import partial

import pandas as pd
import gradio as gr


# -------------------------------------------------
# Evaluation config
# -------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()

# tab_evaluation.py sta in:
# FINBENCH_DEMO/Gradio/Tabs/tab_evaluation.py
# quindi parents[0] = Tabs
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

# -------------------------------------------------
# A list of the only PKLs that are actually available
# -------------------------------------------------
def scan_available_pkls():
    """
    Read only exsisted i .pkl with structure:

    /FINBENCH_DEMO/
        <type>/<model>/results/<universe>/<configuration>/42/
        y<start_year>_y<end_year>/
        results_sl<sl>_pl<pl>.pkl
    """

    records = []

    if not FINBENCH_RESULTS_ROOT.exists():
        return records

    pattern = FINBENCH_RESULTS_ROOT.glob("*/*/results/*/*/42/y*_y*/results_sl*_pl*.pkl")

    for pkl_path in pattern:
        try:
            rel = pkl_path.relative_to(FINBENCH_RESULTS_ROOT)
            parts = rel.parts

            # <type>/<model>/results/<universe>/<configuration>/42/<year_dir>/<file.pkl>
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


def filter_records(records, **filters):
    filtered = records

    for key, value in filters.items():
        if value is None or value == "":
            continue

        filtered = [
            r for r in filtered
            if str(r[key]) == str(value)
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


def selected_combination_exists(
    task_type,
    model,
    universe,
    configuration,
    year_start,
    year_end,
    sequence_length,
    prediction_length,
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
        sequence_length=sequence_length,
        prediction_length=prediction_length,
    )

    if not filtered:
        return False, None

    return True, filtered[0]["pkl_path"]


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
    "sequence_length",
    "prediction_length",
    "top_k",
    "short_k",
]


def reset_downstream_values(changed_field, values):
    """
    When a field changes, all subsequent fields are reset to zero.
    This prevents old or invalid combinations.
    """

    if changed_field == "eval_mode":
        for field in FIELD_ORDER:
            values[field] = None
        return values

    if changed_field not in FIELD_ORDER:
        return values

    changed_idx = FIELD_ORDER.index(changed_field)

    for field in FIELD_ORDER[changed_idx + 1:]:
        values[field] = None

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


def is_not_empty(value):
    return value is not None and value != ""


def update_eval_form(
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
        "sequence_length": sequence_length,
        "prediction_length": prediction_length,
        "top_k": top_k,
        "short_k": short_k,
    }

    values = reset_downstream_values(changed_field, values)

    form_visible = eval_mode in EVAL_MODES
    top_short_visible = eval_mode in ["EOY", "METRICS"]

    if not form_visible:
        return [
            gr.update(visible=False),

            gr.update(choices=[], value=None, interactive=False),
            gr.update(choices=[], value=None, interactive=False),
            gr.update(choices=[], value=None, interactive=False),
            gr.update(choices=[], value=None, interactive=False),
            gr.update(choices=[], value=None, interactive=False),
            gr.update(choices=[], value=None, interactive=False),
            gr.update(choices=[], value=None, interactive=False),
            gr.update(choices=[], value=None, interactive=False),

            gr.update(visible=False),
            gr.update(choices=[], value=None, interactive=False),
            gr.update(choices=[], value=None, interactive=False),

            gr.update(interactive=False),
        ]

    # 1. Task type
    task_type_choices = unique_values(records, "task_type")
    values["task_type"] = value_if_valid(values["task_type"], task_type_choices)

    # 2. Model
    model_records = filter_records(
        records,
        task_type=values["task_type"],
    )

    model_choices = (
        unique_values(model_records, "model")
        if is_not_empty(values["task_type"])
        else []
    )

    values["model"] = value_if_valid(values["model"], model_choices)

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

    values["universe"] = value_if_valid(values["universe"], universe_choices)

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

    values["configuration"] = value_if_valid(
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

    # 7. Sequence length
    sequence_length_records = filter_records(
        records,
        task_type=values["task_type"],
        model=values["model"],
        universe=values["universe"],
        configuration=values["configuration"],
        year_start=values["year_start"],
        year_end=values["year_end"],
    )

    sequence_length_choices = (
        unique_values(sequence_length_records, "sequence_length")
        if is_not_empty(values["year_end"])
        else []
    )

    values["sequence_length"] = value_if_valid(
        values["sequence_length"],
        sequence_length_choices,
    )

    # 8. Prediction length
    prediction_length_records = filter_records(
        records,
        task_type=values["task_type"],
        model=values["model"],
        universe=values["universe"],
        configuration=values["configuration"],
        year_start=values["year_start"],
        year_end=values["year_end"],
        sequence_length=values["sequence_length"],
    )

    prediction_length_choices = (
        unique_values(prediction_length_records, "prediction_length")
        if is_not_empty(values["sequence_length"])
        else []
    )

    values["prediction_length"] = value_if_valid(
        values["prediction_length"],
        prediction_length_choices,
    )

    # 9. Top K e Short K
    top_k_choices = []
    short_k_choices = []

    if top_short_visible and is_not_empty(values["prediction_length"]):
        top_k_choices, short_k_choices = get_top_short_choices(
            values["universe"],
            values["top_k"],
        )

        values["top_k"] = value_if_valid(values["top_k"], top_k_choices)

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
            "sequence_length",
            "prediction_length",
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
            choices=sequence_length_choices,
            value=values["sequence_length"],
            interactive=is_not_empty(values["year_end"]) and bool(sequence_length_choices),
        ),
        gr.update(
            choices=prediction_length_choices,
            value=values["prediction_length"],
            interactive=is_not_empty(values["sequence_length"]) and bool(prediction_length_choices),
        ),

        # Questa è la riga intera Top K / Short K
        gr.update(visible=top_short_visible),

        gr.update(
            choices=top_k_choices,
            value=values["top_k"],
            interactive=(
                    top_short_visible
                    and is_not_empty(values["prediction_length"])
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


def run_evaluation(
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
    exists, pkl_path = selected_combination_exists(
        task_type=task_type,
        model=model,
        universe=universe,
        configuration=configuration,
        year_start=year_start,
        year_end=year_end,
        sequence_length=sequence_length,
        prediction_length=prediction_length,
    )

    if not exists:
        return (
            "The combination you have selected does not match any available .pkl file.",
            pd.DataFrame(),
        )

    cmd_name = EVAL_TO_CMD[eval_mode]

    cmd = [
        sys.executable,
        str(METRICS_EXTRACTION_SCRIPT),
        cmd_name,
        "--universe",
        str(universe),
        "--model",
        str(model),
        "--type",
        str(task_type),
        "--initial_year",
        str(int(year_start)),
        "--end_year",
        str(int(year_end)),
        "--seed",
        str(SEED),
        "--sl",
        str(int(sequence_length)),
        "--pl",
        str(int(prediction_length)),
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

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    completed = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )

    command_str = " ".join(shlex.quote(x) for x in cmd)

    logs = [
        f"Command executed:\n{command_str}",
        f"\nPKL:\n{pkl_path}",
    ]

    if completed.stdout:
        logs.append("\nSTDOUT:\n" + completed.stdout)

    if completed.stderr:
        logs.append("\nSTDERR:\n" + completed.stderr)

    if completed.returncode != 0:
        logs.append(f"\nError: process terminated with code {completed.returncode}.")
        return "\n".join(logs), pd.DataFrame()

    csv_path = output_csv_path(
        eval_mode=eval_mode,
        top_k=top_k,
        short_k=short_k,
    )

    if not csv_path.exists():
        logs.append(f"\nThe expected CSV file was not found:\n{csv_path}")
        return "\n".join(logs), pd.DataFrame()

    df = pd.read_csv(csv_path)

    filtered_df, applied, missing = filter_output_dataframe(
        df=df,
        eval_mode=eval_mode,
        task_type=task_type,
        model=model,
        universe=universe,
        configuration=configuration,
        year_start=year_start,
        year_end=year_end,
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        top_k=top_k,
        short_k=short_k,
    )

    logs.append(f"\nCSV file read:\n{csv_path}")
    logs.append(f"\nTotal number of rows in the CSV file: {len(df)}")
    logs.append(f"Lines shown: {len(filtered_df)}")

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

    return "\n".join(logs), filtered_df


# -------------------------------------------------
# Build tab
# -------------------------------------------------
def build_tab():
    with gr.Tab("Evaluation"):
        gr.Markdown("### Evaluation")

        eval_mode = gr.Radio(
            choices=EVAL_MODES,
            value=None,
            label="Type of assessment",
            interactive=True,
        )

        with gr.Group(visible=False) as params_group:
            with gr.Row():
                task_type = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Task type",
                    interactive=False,
                )

                model = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Model",
                    interactive=False,
                )

            with gr.Row():
                universe = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Universe",
                    interactive=False,
                )

                configuration = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Configuration",
                    interactive=False,
                )

            with gr.Row():
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

            with gr.Row():
                sequence_length = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Sequence length",
                    interactive=False,
                )

                prediction_length = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Prediction length",
                    interactive=False,
                )

            with gr.Row(visible=False) as top_short_row:
                top_k = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Top K",
                    interactive=False,
                )

                short_k = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Short K",
                    interactive=False,
                )
        with gr.Row():
            with gr.Column(scale=2):
                ""
            with gr.Column(scale=1):
                run_btn = gr.Button(
                    "Run evaluation",
                    variant="primary",
                    interactive=False,
                )

            with gr.Column(scale=2):
                ""
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("Execution log", open=False):
                    log_box = gr.Textbox(
                        label=None,
                        interactive=False,
                        lines=10,
                        max_lines=10,
                        elem_id="execution-log",
                    )
            with gr.Column(scale=4):
                output_table = gr.Dataframe(
                    label="Output",
                    interactive=False,
                    wrap=False,
                )

        form_inputs = [
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
        ]

        form_outputs = [
            params_group,
            task_type,
            model,
            universe,
            configuration,
            year_start,
            year_end,
            sequence_length,
            prediction_length,
            top_short_row,
            top_k,
            short_k,
            run_btn,
        ]

        eval_mode.change(
            fn=partial(update_eval_form, changed_field="eval_mode"),
            inputs=form_inputs,
            outputs=form_outputs,
            show_progress="hidden",
            queue=False,

        )

        task_type.change(
            fn=partial(update_eval_form, changed_field="task_type"),
            inputs=form_inputs,
            outputs=form_outputs,
            show_progress="hidden",
            queue=False,
        )

        model.change(
            fn=partial(update_eval_form, changed_field="model"),
            inputs=form_inputs,
            outputs=form_outputs,
            show_progress="hidden",
            queue=False,
        )

        universe.change(
            fn=partial(update_eval_form, changed_field="universe"),
            inputs=form_inputs,
            outputs=form_outputs,
            show_progress="hidden",
            queue=False,
        )

        configuration.change(
            fn=partial(update_eval_form, changed_field="configuration"),
            inputs=form_inputs,
            outputs=form_outputs,
            show_progress="hidden",
            queue=False,
        )

        year_start.change(
            fn=partial(update_eval_form, changed_field="year_start"),
            inputs=form_inputs,
            outputs=form_outputs,
            show_progress="hidden",
            queue=False,
        )

        year_end.change(
            fn=partial(update_eval_form, changed_field="year_end"),
            inputs=form_inputs,
            outputs=form_outputs,
            show_progress="hidden",
            queue=False,
        )

        sequence_length.change(
            fn=partial(update_eval_form, changed_field="sequence_length"),
            inputs=form_inputs,
            outputs=form_outputs,
            show_progress="hidden",
            queue=False,
        )

        prediction_length.change(
            fn=partial(update_eval_form, changed_field="prediction_length"),
            inputs=form_inputs,
            outputs=form_outputs,
            show_progress="hidden",
            queue=False,
        )

        top_k.change(
            fn=partial(update_eval_form, changed_field="top_k"),
            inputs=form_inputs,
            outputs=form_outputs,
            show_progress="hidden",
            queue=False,
        )

        short_k.change(
            fn=partial(update_eval_form, changed_field="short_k"),
            inputs=form_inputs,
            outputs=form_outputs,
            show_progress="hidden",
            queue=False,
        )

        run_btn.click(
            fn=run_evaluation,
            inputs=form_inputs,
            outputs=[
                log_box,
                output_table,
            ],
        )