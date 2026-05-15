"""
Evaluation runner: build subprocess commands, execute them, read and filter
the output CSVs, and expose the Gradio loading-state helpers.
"""
import os
import shlex
import subprocess
import sys

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

from Gradio.eval_config import (
    SEED,
    EVAL_TO_CMD,
    EVALUATION_DIR,
    FINBENCH_RESULTS_ROOT,
    METRICS_EXTRACTION_SCRIPT,
    PROJECT_ROOT,
)
from Gradio.eval_scan import (
    scan_available_pkls,
    filter_records,
    filter_records_by_length_pairs,
    find_column,
)
from Gradio.eval_plot import build_evaluation_plot
from Gradio.eval_output import (
    dataframe_to_metrics_html,
    empty_evaluation_table_html,
    write_evaluation_download_file,
)

# -------------------------------------------------
# Gradio loading-state helpers
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


def hide_evaluation_loading():
    return (
        gr.update(visible=False),                  # table_loading
        gr.update(visible=False),                  # plot_loading
        gr.update(interactive=True),               # run_btn
        gr.update(visible=True, interactive=True), # explain_evaluation_button
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
        return (
            filtered_df[mask],
            [f"{period_col}={start_year}-{end_year}"],
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

    for value, aliases in filters:
        if value is None or value == "":
            continue
        col = find_column(filtered, aliases)
        if col is None:
            missing.append("/".join(aliases))
            continue
        try:
            numeric_value = int(value)
        except Exception:
            numeric_value = None

        if numeric_value is not None:
            mask = pd.to_numeric(filtered[col], errors="coerce") == numeric_value
        else:
            mask = filtered[col].astype(str).str.strip() == str(value).strip()

        filtered = filtered[mask]
        applied.append(f"{col}={value}")

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
        context["selected_top_k"]   = top_k
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
    records  = scan_available_pkls()
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
            "--mean_output", str(EVALUATION_DIR /
                                 "evaluation_result_summary_metrics_mean_over_seeds.csv"),
        ])
    elif eval_mode == "QUINTILE":
        cmd.extend([
            "--root_dir",    str(EVALUATION_DIR / "evaluation_result_quintile"),
            "--mean_output", str(EVALUATION_DIR /
                                 "evaluation_result_summary_metrics_quintile_mean_over_seeds.csv"),
        ])

    return cmd


# -------------------------------------------------
# Main runner
# -------------------------------------------------

def run_evaluation(
    eval_mode,
    task_type, model, universe, configuration,
    year_start, year_end, length_pair,
    top_k, short_k,
):
    _error_return = lambda msg: (
        msg,
        empty_evaluation_table_html(msg),
        gr.update(value=None, visible=False),
        gr.update(visible=False, interactive=False),
        empty_plot(),
        pd.DataFrame(),
    )

    selected_records = selected_records_for_form(
        task_type=task_type, model=model, universe=universe,
        configuration=configuration, year_start=year_start,
        year_end=year_end, length_pair=length_pair,
    )
    if not selected_records:
        return _error_return(
            "The selected values do not match any available .pkl file."
        )

    jobs = [
        {
            "record": rec,
            "top_k":   top_k   if eval_mode in ["EOY", "METRICS"] else None,
            "short_k": short_k if eval_mode in ["EOY", "METRICS"] else None,
        }
        for rec in selected_records
    ]

    if not jobs:
        return _error_return(
            "No valid Top K / Short K pair is available for the selected universes."
        )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    logs = [
        f"Selected .pkl combinations: {len(selected_records)}",
        f"Evaluation jobs to process: {len(jobs)}",
    ]

    frames        = []
    command_cache = {}

    for job_idx, job in enumerate(jobs, start=1):
        record     = job["record"]
        job_top_k  = job["top_k"]
        job_short_k = job["short_k"]

        cmd         = build_eval_command(eval_mode, record, job_top_k, job_short_k)
        command_key = tuple(cmd)

        logs.append(
            "\n" + "=" * 80
            + f"\nJob {job_idx}/{len(jobs)}"
            + f"\nPKL:\n{record['pkl_path']}"
        )

        if command_key not in command_cache:
            completed   = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env,
                                         capture_output=True, text=True)
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
                    "ok": False, "df": pd.DataFrame(), "csv_path": None
                }
                continue

            csv_path = output_csv_path(eval_mode, job_top_k, job_short_k)
            if not csv_path.exists():
                logs.append(f"\nThe expected CSV file was not found:\n{csv_path}")
                command_cache[command_key] = {
                    "ok": False, "df": pd.DataFrame(), "csv_path": csv_path
                }
                continue

            df = pd.read_csv(csv_path)
            command_cache[command_key] = {"ok": True, "df": df, "csv_path": csv_path}

        else:
            logs.append("\nCommand already executed before; reusing generated CSV.")

        cached = command_cache[command_key]
        if not cached["ok"]:
            continue

        df       = cached["df"]
        csv_path = cached["csv_path"]

        filtered_df, applied, missing = filter_output_dataframe(
            df=df, eval_mode=eval_mode,
            task_type=record["task_type"], model=record["model"],
            universe=record["universe"], configuration=record["configuration"],
            year_start=record["year_start"], year_end=record["year_end"],
            sequence_length=record["sequence_length"],
            prediction_length=record["prediction_length"],
            top_k=job_top_k, short_k=job_short_k,
        )

        logs.append(f"\nCSV file read:\n{csv_path}")
        logs.append(f"Total number of rows in the CSV file: {len(df)}")
        logs.append(f"Rows shown for this job: {len(filtered_df)}")

        if applied:
            logs.append("\nFilters applied:\n" + "\n".join(f"- {x}" for x in applied))
        if missing:
            logs.append(
                "\nColumns not found in the CSV file, so not used for filtering:\n"
                + "\n".join(f"- {x}" for x in missing)
            )

        if not filtered_df.empty:
            frames.append(
                add_selection_context(filtered_df, record, eval_mode,
                                      job_top_k, job_short_k)
            )

    if frames:
        output_df = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    else:
        output_df = pd.DataFrame()

    logs.append("\n" + "=" * 80)
    logs.append(f"Unique commands executed: {len(command_cache)}")
    logs.append(f"Final rows shown: {len(output_df)}")

    plt.close("all")

    fig          = build_evaluation_plot(output_df, eval_mode)
    plot_update  = matplotlib_plot_update(fig)
    table_html   = dataframe_to_metrics_html(output_df)
    download_file = write_evaluation_download_file(output_df, eval_mode)

    if download_file is None:
        download_update  = gr.update(value=None, visible=False)
        gpt_button_update = gr.update(visible=False, interactive=False)
    else:
        download_update  = gr.update(value=download_file, visible=True)
        gpt_button_update = gr.update(value="Explain with ChatGPT",
                                      visible=True, interactive=True)

    return "\n".join(logs), table_html, download_update, gpt_button_update, plot_update, output_df