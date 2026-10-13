"""
Output formatting: HTML table rendering, CSV download, prettification,
and LLM-based explanation via OpenAI.
"""
import re
from html import escape
import numpy as np
import pandas as pd
from openai import OpenAI

from Aequitas.eval_config import EVALUATION_DIR, OPENAI_API_KEY, OPENAI_EXPLANATION_MODEL
from Aequitas.eval_scan import normalize_column_name
from Aequitas.ui import empty_state

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
# Column-name humanization
# -------------------------------------------------

_COLUMN_ALIASES = {
    "task_type": "Task type",
    "type": "Task type",
    "model": "Model",
    "universe": "Universe",
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
    "q1": "Q1", "q2": "Q2", "q3": "Q3", "q4": "Q4", "q5": "Q5",
}


def humanize_column_name(name: str) -> str:
    raw = str(name).strip()
    if raw.startswith("selected_"):
        raw = raw[len("selected_"):]
    normalized = normalize_column_name(raw)
    if normalized in _COLUMN_ALIASES:
        return _COLUMN_ALIASES[normalized]
    if re.match(r"^\d{4}$", raw):
        return raw
    return raw.replace("_", " ").strip().capitalize()


# -------------------------------------------------
# DataFrame prettification
# -------------------------------------------------

def prettify_output_dataframe(df) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    pretty_df = df.copy()
    pretty_df.columns = [humanize_column_name(col) for col in pretty_df.columns]
    pretty_df = pretty_df.loc[
        :, ~pd.Index(pretty_df.columns).duplicated(keep="first")
    ]

    return pretty_df

# -------------------------------------------------
# HTML rendering
# -------------------------------------------------

def empty_evaluation_table_html(message: str = "No evaluation output yet.") -> str:
    return empty_state(message)


def format_display_value(value) -> str:
    if pd.isna(value):
        return "-"

    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4f}".rstrip("0").rstrip(".")

    return str(value)


def dataframe_to_metrics_html(df) -> str:
    if df is None or df.empty:
        return empty_evaluation_table_html("No rows to display for this selection.")

    df = df.copy()
    df = df.dropna(axis=1, how="all")

    # 1.  Collapse the duplicate benchmark rows.
    df = collapse_benchmark_rows_for_display(df)

    if df.empty:
        return empty_evaluation_table_html("No rows to display for this selection.")

    # 2. Find columns of data to compare: years or metrics.
    value_cols = _comparison_value_columns(df)

    # 3. Create a benchmark lookup.
    benchmark_mask = _benchmark_mask(df)
    benchmark_df = df[benchmark_mask].copy()
    benchmark_key_cols = []

    for alternatives in [
        ["universe", "selected_universe"],
        ["period"],
    ]:
        for col in alternatives:
            if col in df.columns:
                benchmark_key_cols.append(col)
                break

    benchmark_lookup = {}

    for _, bench_row in benchmark_df.iterrows():
        key = tuple(bench_row.get(col) for col in benchmark_key_cols)
        benchmark_lookup[key] = bench_row

    # 4. Format the columns, retaining the raw-to-formatted mapping.
    raw_cols = list(df.columns)
    pretty_cols = [humanize_column_name(col) for col in raw_cols]

    kept_raw_cols = []
    kept_pretty_cols = []
    seen = set()

    hidden_display_cols = {"row_type", "source_model"}

    for raw_col, pretty_col in zip(raw_cols, pretty_cols):
        if str(raw_col) in hidden_display_cols:
            continue

        if pretty_col in seen:
            continue

        seen.add(pretty_col)
        kept_raw_cols.append(raw_col)
        kept_pretty_cols.append(pretty_col)

    # 5.  Build the HTML manually.
    html = []
    html.append('<table class="aq-table">')

    html.append("<thead><tr>")
    for pretty_col in kept_pretty_cols:
        html.append(f"<th>{escape(str(pretty_col))}</th>")
    html.append("</tr></thead>")

    html.append("<tbody>")

    for _, row in df.iterrows():
        is_benchmark = str(row.get("row_type", "")).lower() == "benchmark"

        row_classes = []
        if is_benchmark:
            row_classes.append("benchmark-row")

        tr_class = " ".join(row_classes)
        html.append(f'<tr class="{tr_class}">')

        row_key = tuple(row.get(col) for col in benchmark_key_cols)
        benchmark_row = benchmark_lookup.get(row_key)

        for raw_col in kept_raw_cols:
            value = row.get(raw_col, "")

            td_classes = []

            if (
                not is_benchmark
                and benchmark_row is not None
                and raw_col in value_cols
            ):
                current_value = pd.to_numeric(value, errors="coerce")
                benchmark_value = pd.to_numeric(
                    benchmark_row.get(raw_col),
                    errors="coerce"
                )

                if (
                    pd.notna(current_value)
                    and pd.notna(benchmark_value)
                    and current_value > benchmark_value
                ):
                    td_classes.append("better-than-benchmark")

            td_class = " ".join(td_classes)

            display_value = format_display_value(value)

            html.append(
                f'<td class="{td_class}">{escape(display_value)}</td>'
            )

        html.append("</tr>")

    html.append("</tbody>")
    html.append("</table>")

    table_html = "\n".join(html)

    return f"""
    <div class="aq-table-wrap">
        <div class="aq-table-scroll">
            {table_html}
        </div>
    </div>
    """


def _benchmark_mask(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(False, index=df.index)

    if "row_type" in df.columns:
        return (
            df["row_type"]
            .astype(str)
            .str.lower()
            .eq("benchmark")
        )

    mask = pd.Series(False, index=df.index)

    for col in ["model", "config", "configuration", "task_type", "type"]:
        if col in df.columns:
            mask |= (
                df[col]
                .astype(str)
                .str.lower()
                .str.contains("benchmark|bench|market", regex=True, na=False)
            )

    return mask


def _comparison_value_columns(df: pd.DataFrame) -> list:
    cols = []

    for col in df.columns:
        col_str = str(col)

        # EOY columns: 2021, 2022, ...
        if re.match(r"^\d{4}$", col_str):
            cols.append(col)
            continue

        # Metrics columns
        if "TARGET_METRICS" in globals() and col in TARGET_METRICS:
            cols.append(col)

    return cols

def collapse_benchmark_rows_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()

    bench_mask = _benchmark_mask(df)

    if "row_type" not in df.columns:
        df.insert(
            0,
            "row_type",
            np.where(bench_mask, "benchmark", "strategy")
        )
    else:
        df["row_type"] = np.where(bench_mask, "benchmark", "strategy")

    strategy_df = df[~bench_mask].copy()
    benchmark_df = df[bench_mask].copy()

    if benchmark_df.empty:
        return df

    value_cols = _comparison_value_columns(df)

    def first_existing_col(columns):
        for col in columns:
            if col in benchmark_df.columns:
                return col
        return None

    universe_col = first_existing_col(["universe", "selected_universe"])
    period_col = first_existing_col(["period"])

    # Il benchmark non dipende dal task_type:
    # quindi NON raggruppiamo per task_type.
    group_cols = [
        col for col in [universe_col, period_col]
        if col is not None
    ]

    if not group_cols:
        return df

    agg_dict = {col: "mean" for col in value_cols}

    if agg_dict:
        benchmark_grouped = (
            benchmark_df
            .groupby(group_cols, as_index=False, dropna=False)
            .agg(agg_dict)
        )
    else:
        benchmark_grouped = benchmark_df[group_cols].drop_duplicates()

    benchmark_rows = []

    for _, bench in benchmark_grouped.iterrows():
        row = {col: np.nan for col in df.columns}

        universe_value = bench.get(universe_col, "") if universe_col else ""
        period_value = bench.get(period_col, "") if period_col else ""

        # Universe
        for col in ["universe", "selected_universe"]:
            if col in row:
                row[col] = universe_value

        # Task type unico per benchmark
        for col in ["task_type", "type", "selected_task_type"]:
            if col in row:
                row[col] = "-"

        # Model benchmark
        for col in ["model", "selected_model"]:
            if col in row:
                row[col] = "benchmark"

        # Period, se presente
        if "period" in row:
            row["period"] = period_value

        # Row type
        if "row_type" in row:
            row["row_type"] = "benchmark"

        # Valori benchmark: anni o metriche
        for col in value_cols:
            row[col] = bench.get(col, np.nan)

        benchmark_rows.append(row)

    benchmark_collapsed_df = pd.DataFrame(
        benchmark_rows,
        columns=df.columns,
    )

    # Benchmark in prima posizione nella tabella
    return (
        pd.concat(
            [benchmark_collapsed_df, strategy_df],
            ignore_index=True,
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )


# -------------------------------------------------
# CSV download
# -------------------------------------------------

def write_evaluation_download_file(df, eval_mode) -> str | None:
    pretty_df = prettify_output_dataframe(df)
    if pretty_df.empty:
        return None
    output_path = EVALUATION_DIR / f"evaluation_output_{str(eval_mode).lower()}.csv"
    pretty_df.to_csv(output_path, index=False)
    return str(output_path)


# -------------------------------------------------
# LLM text serialization
# -------------------------------------------------

def dataframe_to_llm_text(df, max_rows: int = 80) -> str:
    pretty_df = prettify_output_dataframe(df)
    if pretty_df.empty:
        return ""

    truncated = pretty_df.head(max_rows).copy()
    csv_text = truncated.to_csv(index=False)

    if len(pretty_df) > max_rows:
        csv_text += (
            f"\n\n[Note: the original table contains {len(pretty_df)} rows; "
            f"Only the first ones are shown here {max_rows}.]"
        )
    return csv_text


# -------------------------------------------------
# LLM explanation (OpenAI)
# -------------------------------------------------

def explain_evaluation_with_chatgpt(output_df, eval_mode: str) -> str:
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
        return response.output_text
    except Exception as e:
        return f"Error while calling ChatGPT: {escape(str(e))}"