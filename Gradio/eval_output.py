"""
Output formatting: HTML table rendering, CSV download, prettification,
and LLM-based explanation via OpenAI.
"""
import re
from html import escape

import pandas as pd
from openai import OpenAI

from Gradio.eval_config import EVALUATION_DIR, OPENAI_API_KEY, OPENAI_EXPLANATION_MODEL
from Gradio.eval_scan import normalize_column_name

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

    # Keep first occurrence of each human-readable column name.
    # selected_* columns are inserted first, so their version wins over
    # duplicated metadata columns coming from the generated CSV.
    pretty_df = pretty_df.loc[
        :, ~pd.Index(pretty_df.columns).duplicated(keep="first")
    ]
    return pretty_df


# -------------------------------------------------
# HTML rendering
# -------------------------------------------------

def empty_evaluation_table_html(message: str = "No evaluation output yet.") -> str:
    return f"""
    <div class="metrics-table-wrap evaluation-metrics-table-wrap">
        <div class="metrics-empty">
            <div>{escape(message)}</div>
        </div>
    </div>
    """


def dataframe_to_metrics_html(df) -> str:
    pretty_df = prettify_output_dataframe(df)

    if pretty_df.empty:
        return empty_evaluation_table_html("No rows to display for this selection.")

    table_html = pretty_df.to_html(
        index=False, border=0, classes="mpt", escape=True, na_rep=""
    )
    return f"""
    <div class="metrics-table-wrap evaluation-metrics-table-wrap">
        <div class="metrics-outer metrics-scroll">
            {table_html}
        </div>
    </div>
    """


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
    csv_text  = truncated.to_csv(index=False)

    if len(pretty_df) > max_rows:
        csv_text += (
            f"\n\n[Nota: la tabella originale contiene {len(pretty_df)} righe; "
            f"qui sono mostrate solo le prime {max_rows}.]"
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