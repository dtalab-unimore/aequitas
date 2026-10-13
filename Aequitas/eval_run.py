"""
Evaluation runner: build subprocess commands, execute them, read and filter
the output CSVs, and expose the Aequitas loading-state helpers.
"""
import os
import re
import shlex
import subprocess
import sys
import time
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Aequitas.eval_config import (
    SEED,
    EVAL_TO_CMD,
    EVALUATION_DIR,
    FINBENCH_RESULTS_ROOT,
    METRICS_EXTRACTION_SCRIPT,
    PROJECT_ROOT,
)
from Aequitas.eval_scan import (
    scan_available_pkls,
    filter_records,
    filter_records_by_length_pairs,
    find_column,
)
from Aequitas.eval_form import update_eval_form
from Aequitas.eval_plot import build_evaluation_plot
from Aequitas.script_runner import start_job, follow_job, get_job
from Aequitas.eval_output import (
    dataframe_to_metrics_html,
    empty_evaluation_table_html,
    write_evaluation_download_file,
)

TARGET_METRICS = [
    "Cumulative Return",
    "CAGR﹪",
    "Sharpe",
    "Max Drawdown",
    "Volatility (ann.)",
    "Sortino/√2",
    "Avg. Drawdown"
]

# -------------------------------------------------
# Aequitas loading-state helpers
# -------------------------------------------------

def empty_plot():
    return gr.update(value=None, visible=False)


def matplotlib_plot_update(fig):
    if fig is None:
        return gr.update(value=None, visible=False)
    return gr.update(value=fig, visible=True)


def show_evaluation_loading():
    return (
        gr.update(visible=True),               # table_loading
        gr.update(visible=True),               # plot_loading
        gr.update(interactive=False),          # run_btn
        gr.update(value=None, visible=False),  # download_evaluation_button
        gr.update(visible=False, interactive=False),  # explain_evaluation_button
        gr.update(visible=False),              # chatgpt_explanation_group
        "",                                    # chatgpt_explanation
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
        gr.update(visible=False),                   # chatgpt_loading
        gr.update(visible=True, interactive=True),  # explain_evaluation_button
        gr.update(visible=True),                    # chatgpt_explanation_group
    )


# -------------------------------------------------
# Output CSV path
# -------------------------------------------------

def output_csv_path(eval_mode: str, top_k=None, short_k=None):
    if eval_mode == "EOY":
        return EVALUATION_DIR / f"aggregated_eoy_returns_{int(top_k)}_{int(short_k)}.csv"
    if eval_mode == "METRICS":
        return EVALUATION_DIR / "evaluation_result_summary_metrics_mean_over_seeds.csv"
    if eval_mode == "QUINTILE":
        return EVALUATION_DIR / "evaluation_result_summary_metrics_quintile_mean_over_seeds.csv"
    raise ValueError(f"Evaluation mode not valid: {eval_mode}")


# -------------------------------------------------
# DataFrame filter
# -------------------------------------------------

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
        end_year   = str(end_year)

        start_col = find_column(filtered_df,
                                ["initial_year", "start_year", "year_start", "initialyear"])
        end_col   = find_column(filtered_df,
                                ["end_year", "final_year", "year_end", "endyear"])

        if start_col is not None and end_col is not None:
            start_values = pd.to_numeric(filtered_df[start_col], errors="coerce")
            end_values   = pd.to_numeric(filtered_df[end_col],   errors="coerce")
            mask = (start_values == int(start_year)) & (end_values == int(end_year))
            return (
                filtered_df[mask],
                [f"{start_col}={start_year}", f"{end_col}={end_year}"],
                [],
            )

        period_col = find_column(
            filtered_df,
            ["period", "periodo", "year_range", "years", "window",
             "date_range", "evaluation_period"],
        )
        if period_col is None:
            return filtered_df, [], ["period / year_start-year_end"]

        period_values = (
            filtered_df[period_col].astype(str)
            .str.lower().str.replace(" ", "", regex=False)
        )
        extracted = period_values.str.extract(r"(?P<start>\d{4}).*?(?P<end>\d{4})")
        mask = (
            pd.to_numeric(extracted["start"], errors="coerce").eq(int(start_year))
            & pd.to_numeric(extracted["end"],   errors="coerce").eq(int(end_year))
        )
        before = len(filtered_df)

        filtered_df = filtered_df[mask]

        after = len(filtered_df)

        return (
            filtered_df,
            [f"{period_col}={start_year}-{end_year} ({before}->{after})"],
            [],
        )

    filters = [
        (task_type,        ["type", "task_type"]),
        (model,            ["model"]),
        (universe,         ["universe"]),
        (configuration,    ["configuration", "config"]),
        (sequence_length,  ["sl", "sequence_length"]),
        (prediction_length,["pl", "prediction_length"]),
    ]
    if eval_mode in ["EOY", "METRICS"]:
        filters.extend([
            (top_k,   ["top_k", "topk"]),
            (short_k, ["short_k", "shortk"]),
        ])

    filtered = df.copy()
    applied  = []
    missing  = []


    def is_benchmark_row(dataframe):
        mask = pd.Series(False, index=dataframe.index)

        for aliases in [["model"], ["configuration", "config"], ["task_type", "type"]]:
            col = find_column(dataframe, aliases)
            if col is not None:
                mask |= (
                    dataframe[col]
                    .astype(str)
                    .str.lower()
                    .str.contains("benchmark|bench|market", regex=True, na=False)
                )

        return mask


    def build_filter_mask(series, value):
        values = value if isinstance(value, list) else [value]
        values = [v for v in values if v is not None and v != ""]

        if not values:
            return pd.Series(True, index=series.index)

        numeric_values = []

        all_numeric = True
        for v in values:
            try:
                numeric_values.append(int(v))
            except Exception:
                all_numeric = False
                break

        if all_numeric:
            return pd.to_numeric(series, errors="coerce").isin(numeric_values)

        values_str = {str(v).strip() for v in values}

        return series.astype(str).str.strip().isin(values_str)

    for value, aliases in filters:
        if value is None or value == "" or value == []:
            continue

        col = find_column(filtered, aliases)

        if col is None:
            missing.append("/".join(aliases))
            continue

        before = len(filtered)

        normal_mask = build_filter_mask(filtered[col], value)

        if aliases == ["model"]:
            mask = normal_mask | is_benchmark_row(filtered)
        else:
            mask = normal_mask

        filtered = filtered[mask]

        after = len(filtered)
        applied.append(f"{col}={value} ({before}->{after})")

    if year_start is not None and year_end is not None:
        filtered, period_applied, period_missing = apply_period_filter(
            filtered, year_start, year_end
        )
        applied.extend(period_applied)
        missing.extend(period_missing)

    return filtered, applied, missing


# -------------------------------------------------
# Selection context enrichment
# -------------------------------------------------

def add_selection_context(df, record, eval_mode, top_k=None, short_k=None):
    enriched = df.copy()

    def detect_benchmark_rows(dataframe):
        mask = pd.Series(False, index=dataframe.index)

        for aliases in [["model"], ["configuration", "config"], ["task_type", "type"]]:
            col = find_column(dataframe, aliases)
            if col is not None:
                mask |= (
                    dataframe[col]
                    .astype(str)
                    .str.lower()
                    .str.contains("benchmark|bench|market", regex=True, na=False)
                )

        return mask

    benchmark_mask = detect_benchmark_rows(enriched)

    if "row_type" not in enriched.columns:
        enriched.insert(
            0,
            "row_type",
            np.where(benchmark_mask, "benchmark", "strategy")
        )
    else:
        enriched["row_type"] = np.where(benchmark_mask, "benchmark", "strategy")

    context = {
        "selected_task_type":         record["task_type"],
        "selected_model":             record["model"],
        "selected_universe":          record["universe"],
        "selected_configuration":     record["configuration"],
        "selected_year_start":        record["year_start"],
        "selected_year_end":          record["year_end"],
        "selected_sequence_length":   record["sequence_length"],
        "selected_prediction_length": record["prediction_length"],
    }

    if eval_mode in ["EOY", "METRICS"]:
        context["selected_top_k"] = top_k
        context["selected_short_k"] = short_k

    for col, value in reversed(list(context.items())):
        enriched.insert(0, col, value)

    return enriched


# -------------------------------------------------
# Record selection helper
# -------------------------------------------------

def selected_records_for_form(
    task_type, model, universe, configuration,
    year_start, year_end, length_pair,
) -> list:
    records = scan_available_pkls()
    filtered = filter_records(
        records,
        task_type=task_type, model=model,
        universe=universe, configuration=configuration,
        year_start=year_start, year_end=year_end,
    )
    return filter_records_by_length_pairs(filtered, length_pair)


# -------------------------------------------------
# Command builder
# -------------------------------------------------

def build_eval_command(eval_mode: str, record: dict,
                       top_k=None, short_k=None) -> list:
    cmd_name = EVAL_TO_CMD[eval_mode]
    cmd = [
        sys.executable,
        str(METRICS_EXTRACTION_SCRIPT),
        cmd_name,
        "--universe",     str(record["universe"]),
        "--model",        str(record["model"]),
        "--type",         str(record["task_type"]),
        "--initial_year", str(int(record["year_start"])),
        "--end_year",     str(int(record["year_end"])),
        "--seed",         str(SEED),
        "--sl",           str(int(record["sequence_length"])),
        "--pl",           str(int(record["prediction_length"])),
        "--results_root", str(FINBENCH_RESULTS_ROOT),
    ]

    if eval_mode == "EOY":
        cmd.extend([
            "--top_k",         str(int(top_k)),
            "--short_k",       str(int(short_k)),
            "--base_dir",      str(EVALUATION_DIR / "evaluation_result"),
            "--output_prefix", str(EVALUATION_DIR / "aggregated_eoy_returns"),
        ])
    elif eval_mode == "METRICS":
        cmd.extend([
            "--top_k",       str(int(top_k)),
            "--short_k",     str(int(short_k)),
            "--root_dir",    str(EVALUATION_DIR / "evaluation_result"),
            "--mean_output", str(EVALUATION_DIR / "evaluation_result_summary_metrics_mean_over_seeds.csv"),
        ])
    elif eval_mode == "QUINTILE":
        cmd.extend([
            "--root_dir",    str(EVALUATION_DIR / "evaluation_result_quintile"),
            "--mean_output", str(EVALUATION_DIR / "evaluation_result_summary_metrics_quintile_mean_over_seeds.csv"),
        ])

    return cmd


def collapse_benchmark_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "row_type" not in df.columns:
        return df

    benchmark_mask = (
        df["row_type"]
        .astype(str)
        .str.lower()
        .eq("benchmark")
    )

    strategy_df = df[~benchmark_mask].copy()
    benchmark_df = df[benchmark_mask].copy()

    if benchmark_df.empty:
        return df

    value_cols = [
        col for col in benchmark_df.columns
        if re.match(r"^\d{4}$", str(col)) or col in TARGET_METRICS
    ]

    group_cols = [
        col for col in [
            "selected_task_type",
            "selected_model",
            "selected_universe",
            "selected_configuration",
            "selected_year_start",
            "selected_year_end",
            "selected_sequence_length",
            "selected_prediction_length",
            "selected_top_k",
            "selected_short_k",
            "universe",
            "task_type",
            "model",
            "config",
            "row_type",
            "topk",
            "shortk",
            "sl",
            "pl",
            "period",
        ]
        if col in benchmark_df.columns
    ]

    agg_dict = {}

    for col in benchmark_df.columns:
        if col in group_cols:
            continue

        if col in value_cols:
            agg_dict[col] = "mean"
        else:
            agg_dict[col] = "first"

    benchmark_df = (
        benchmark_df
        .groupby(group_cols, as_index=False, dropna=False)
        .agg(agg_dict)
    )

    return (
        pd.concat([strategy_df, benchmark_df], ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )

# -------------------------------------------------
# Main runner
# -------------------------------------------------
# The evaluation runs as a background job (script_runner): it survives page
# reloads, and the Evaluation tab can re-attach to it (follow_evaluation).

def evaluate(
    eval_mode,
    task_type, model, universe, configuration,
    year_start, year_end, length_pair,
    top_k, short_k,
    log_fn,
):
    """
    Run the evaluation commands and collect their output rows.

    Returns {"eval_mode", "output_df"} or, when nothing can be evaluated,
    {"eval_mode", "error"}. Progress goes to `log_fn`.
    """
    def error(msg):
        log_fn(msg)
        return {"eval_mode": eval_mode, "error": msg}

    selected_records = selected_records_for_form(
        task_type=task_type, model=model, universe=universe,
        configuration=configuration, year_start=year_start,
        year_end=year_end, length_pair=length_pair,
    )

    if not selected_records:
        return error("The selected values do not match any available .pkl file.")

    jobs = [
        {
            "record": rec,
            "top_k":   top_k if eval_mode in ["EOY", "METRICS"] else None,
            "short_k": short_k if eval_mode in ["EOY", "METRICS"] else None,
        }
        for rec in selected_records
    ]

    log_fn(f"Selected .pkl combinations: {len(selected_records)}")
    log_fn(f"Evaluation jobs to process: {len(jobs)}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    frames = []
    command_cache = {}

    # ============================================================
    # 1) Execute all jobs first
    # ============================================================

    for job_idx, job in enumerate(jobs, start=1):
        record = job["record"]
        job_top_k = job["top_k"]
        job_short_k = job["short_k"]

        cmd = build_eval_command(eval_mode, record, job_top_k, job_short_k)
        command_key = tuple(cmd)

        log_fn("=" * 80)
        log_fn(f"Job {job_idx}/{len(jobs)}\nPKL:\n{record['pkl_path']}")

        if command_key in command_cache:
            log_fn("Command already executed before; skipping duplicate execution.")
            continue

        command_str = " ".join(shlex.quote(x) for x in cmd)
        log_fn(f"Command executed:\n{command_str}")

        completed = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
        )

        if completed.stdout:
            log_fn("STDOUT:\n" + completed.stdout.rstrip())

        if completed.stderr:
            log_fn("STDERR:\n" + completed.stderr.rstrip())

        command_cache[command_key] = {
            "returncode": completed.returncode,
            "cmd": cmd,
        }

        if completed.returncode != 0:
            log_fn(
                f"Warning: process terminated with code {completed.returncode}. "
                "The final CSV will still be read after all jobs."
            )

    # ============================================================
    # 2) Read output CSVs only after all jobs are finished
    # ============================================================

    log_fn("=" * 80)
    log_fn("Reading output CSV files after all jobs...")

    csv_cache = {}

    for job_idx, job in enumerate(jobs, start=1):
        record = job["record"]
        job_top_k = job["top_k"]
        job_short_k = job["short_k"]

        csv_path = output_csv_path(eval_mode, job_top_k, job_short_k)

        log_fn("-" * 80)
        log_fn(f"Filtering final CSV for job {job_idx}/{len(jobs)}\nExpected CSV:\n{csv_path}")

        if not csv_path.exists():
            log_fn(f"CSV file not found, skipping this job:\n{csv_path}")
            continue

        csv_key = str(csv_path)

        if csv_key not in csv_cache:
            try:
                csv_cache[csv_key] = pd.read_csv(csv_path)
            except Exception as e:
                log_fn(f"Could not read CSV file:\n{csv_path}\nError: {e}")
                continue

        df = csv_cache[csv_key]

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

        log_fn(f"CSV file read:\n{csv_path}")
        log_fn(f"Total number of rows in the CSV file: {len(df)}")
        log_fn(f"Rows shown for this job: {len(filtered_df)}")

        if applied:
            log_fn("Filters applied:\n" + "\n".join(f"- {x}" for x in applied))

        if missing:
            log_fn(
                "Columns not found in the CSV file, so not used for filtering:\n"
                + "\n".join(f"- {x}" for x in missing)
            )

        if not filtered_df.empty:
            frames.append(
                add_selection_context(
                    filtered_df,
                    record,
                    eval_mode,
                    job_top_k,
                    job_short_k,
                )
            )

    if frames:
        output_df = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    else:
        output_df = pd.DataFrame()

    output_df = collapse_benchmark_rows(output_df)

    log_fn("=" * 80)
    log_fn(f"Unique commands executed: {len(command_cache)}")
    log_fn(f"Final rows shown: {len(output_df)}")

    return {"eval_mode": eval_mode, "output_df": output_df}


def render_evaluation(result):
    """
    Build the tab outputs from an `evaluate` result.

    Returns: output_table, download_button, explain_button, plot, output_df.
    Runs in the request thread, not in the job thread: pyplot is not thread-safe.
    """
    if result is None or "error" in result:
        msg = (result or {}).get("error", "The evaluation failed: see the execution log.")
        return (
            empty_evaluation_table_html(msg),
            gr.update(value=None, visible=False),
            gr.update(visible=False, interactive=False),
            empty_plot(),
            pd.DataFrame(),
        )

    eval_mode = result["eval_mode"]
    output_df = result["output_df"]

    plt.close("all")

    fig = build_evaluation_plot(output_df, eval_mode)
    plot_update = matplotlib_plot_update(fig)
    table_html = dataframe_to_metrics_html(output_df)
    download_file = write_evaluation_download_file(output_df, eval_mode)

    if download_file is None:
        download_update = gr.update(value=None, visible=False)
        gpt_button_update = gr.update(visible=False, interactive=False)
    else:
        download_update = gr.update(value=download_file, visible=True)
        gpt_button_update = gr.update(value="Explain with ChatGPT",
                                      visible=True, interactive=True)

    return table_html, download_update, gpt_button_update, plot_update, output_df


def run_evaluation(
    eval_mode,
    task_type, model, universe, configuration,
    year_start, year_end, length_pair,
    top_k, short_k,
):
    """Start an evaluation job and stream it into the tab (see follow_evaluation)."""
    form_values = {
        "eval_mode": eval_mode,
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

    job_id = start_job(
        lambda log_fn: evaluate(**form_values, log_fn=log_fn),
        "Evaluation",
        meta=form_values,
        format_result=None,
    )
    yield from follow_evaluation(job_id)


def follow_evaluation(job_id):
    """
    Stream an evaluation job into the Evaluation tab.

    Yields: log_box, output_table, download_button, explain_button, plot,
    evaluation_output_state, eval_job_state, table_loading, plot_loading, run_btn.
    """
    job = get_job(job_id)
    first = True

    for logs_text, _, running in follow_job(job_id):
        job_state = job_id if first else gr.skip()
        first = False

        if running:
            yield (
                gr.update(value=logs_text),
                gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                job_state,
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(interactive=False),
            )
            continue

        if job.status == "error":
            logs_text += "\n\n" + job.result_text

        yield (
            gr.update(value=logs_text),
            *render_evaluation(job.result if job.status == "ok" else None),
            job_state,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(interactive=True),
        )


def restore_evaluation_view(job_id, run_shown):
    """
    When the Evaluation tab is opened after a page reload, put the form back
    as it was for the browser's last evaluation. Does nothing if this page
    already shows a run.

    Outputs: eval_mode, the form outputs of update_eval_form, resume_token, run_shown.
    The results are streamed by the resume_token change event (follow_evaluation).
    """
    job = get_job(job_id)
    if job is None or run_shown:
        return (*(gr.update(),) * 13, gr.update(), run_shown)

    m = job.meta
    form_updates = update_eval_form(**m, changed_field="short_k")

    return (
        gr.update(value=m["eval_mode"]),
        *form_updates,
        f"{job_id}:{time.time()}",
        True,
    )


def resume_evaluation(job_id):
    """After a page reload, re-attach the tab to the browser's last evaluation."""
    if get_job(job_id) is None:
        yield (gr.skip(),) * 10
        return

    yield from follow_evaluation(job_id)
