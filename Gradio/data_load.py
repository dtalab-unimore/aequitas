from __future__ import annotations
import math
import json
import math
import os
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from html import escape
import gradio as gr
import numpy as np
import pandas as pd


# =============================================================================
# Project paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = PROJECT_ROOT / "Evaluation"
DATA_DIR = EVALUATION_DIR / "data"
CONFIG_PATH = PROJECT_ROOT / "Gradio" / "configs" / "model_to_data.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Evaluation.main_demo_dataload import (
    PipelineContext,
    MarketContext,
    TARGETS,
    MARKET_TARGETS,
    run_universe_pipeline,
    run_market_pipeline,
)


# =============================================================================
# Config
# =============================================================================

PREVIEW_PAGE_SIZE = 7

DATASET_TO_NATION = {
    "sp500": "us",
    "dji": "us",
    "nasdaq100": "us",
    "sxxp": "eu",
    "sx5e": "eu",
}


def read_configs() -> dict:
    if not CONFIG_PATH.exists():
        return {}

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


CONFIGS = read_configs()


# =============================================================================
# Model config -> pipeline target mapping
# =============================================================================

UNIVERSE_FILE_TO_TARGET = {
    "{universe}.csv": "prices",
    "{universe}_info.csv": "info",
    "{universe}_market_cap.csv": "market_cap",
    "{universe}_alpha158.csv": "alpha158",
    "{universe}_alpha360.csv": "alpha360",
    "{universe}_tech.csv": "technicals",
    "{universe}_sector_industry_matrix.npz": "sector_industry_matrix",
    "{universe}_corr_matrix.npz": "corr_matrix",
    "{universe}_wikidata_matrix.npz": "wikidata_matrix",
    "{universe}_inc_matrix.npz": "inc_matrix",
}

MARKET_FILE_TO_TARGET = {
    "{nation}_indexes.csv": ("auto", "indexes"),
    "{nation}_market.csv": ("auto", "market"),
    "cnnpred_market.csv": ("global", "cnnpred_market"),
}


def _template_basename(template: str) -> str:
    return str(template).replace("\\", "/").split("/")[-1]


def get_models() -> list[str]:
    return list(CONFIGS.keys())


def get_nation_from_universe(universe: str) -> str:
    if universe not in DATASET_TO_NATION:
        valid = ", ".join(sorted(DATASET_TO_NATION))
        raise ValueError(f"Unknown universe '{universe}'. Valid values: {valid}")

    return DATASET_TO_NATION[universe]


def _resolve_market_nation(dataset: str, nation: str) -> str:
    if nation == "auto":
        return get_nation_from_universe(dataset)

    return nation


def _dedupe_keep_order(items: list) -> list:
    out = []
    seen = set()

    for item in items:
        key = json.dumps(item, sort_keys=True) if isinstance(item, list) else str(item)

        if key in seen:
            continue

        seen.add(key)
        out.append(item)

    return out

def _normalize_models(model: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """
    Normalizes the Gradio multiselect value.

    Gradio multiselect returns:
        []              if nothing is selected
        ["A", "B"]      if multiple models are selected

    This helper also supports the old single-string case.
    """
    if model is None:
        return []

    if isinstance(model, str):
        return [model] if model else []

    return [m for m in model if isinstance(m, str) and m]

def _targets_from_model_config(
    dataset: str,
    model: str | list[str] | tuple[str, ...] | None,
) -> tuple[list[str], list[list[str]], list[str]]:
    """
    Reads Gradio/configs/model_to_data.json and converts required files into
    pipeline targets.

    Supports multiple selected models.

    Returns:
        universe_targets
        market_targets as [[nation, target], ...]
        unsupported_files
    """
    models = _normalize_models(model)

    if not models:
        return ["prices"], [], []

    universe_targets = []
    market_targets = []
    unsupported_files = []

    nation = get_nation_from_universe(dataset)

    for model_name in models:
        if model_name not in CONFIGS:
            unsupported_files.append(f"{model_name} -> model not found in config")
            continue

        raw_files = CONFIGS.get(model_name, [])

        for file_template in raw_files:
            template_key = _template_basename(file_template)

            if template_key in UNIVERSE_FILE_TO_TARGET:
                universe_targets.append(UNIVERSE_FILE_TO_TARGET[template_key])
                continue

            if template_key in MARKET_FILE_TO_TARGET:
                nation_key, target_name = MARKET_FILE_TO_TARGET[template_key]
                resolved_nation = _resolve_market_nation(dataset, nation_key)
                market_targets.append([resolved_nation, target_name])
                continue

            try:
                resolved_name = template_key.format(
                    universe=dataset,
                    nation=nation,
                )
            except Exception:
                resolved_name = template_key

            unsupported_files.append(f"{model_name} -> {resolved_name}")

    if not universe_targets and not market_targets and not unsupported_files:
        universe_targets = ["prices"]

    return (
        _dedupe_keep_order(universe_targets),
        _dedupe_keep_order(market_targets),
        _dedupe_keep_order(unsupported_files),
    )
# =============================================================================
# File map for preview
# =============================================================================

def _build_file_map(universe: str) -> dict[str, str]:
    out_dir = DATA_DIR / universe

    return {
        "Dataset": str(out_dir / f"{universe}.csv"),
        "Info": str(out_dir / f"{universe}_info.csv"),
        "Market Cap": str(out_dir / f"{universe}_market_cap.csv"),
        "Alpha158": str(out_dir / f"{universe}_alpha158.csv"),
        "Alpha360": str(out_dir / f"{universe}_alpha360.csv"),
        "Technicals": str(out_dir / f"{universe}_tech.csv"),
        "US Indexes": str(DATA_DIR / "us_indexes.csv"),
        "EU Indexes": str(DATA_DIR / "eu_indexes.csv"),
        "US Market Info": str(DATA_DIR / "us_market.csv"),
        "EU Market Info": str(DATA_DIR / "eu_market.csv"),
        "CNNPred Market Features": str(DATA_DIR / "cnnpred_market.csv"),
        "News Relations": str(DATA_DIR / "news"),
        "Incidence Matrix": str(out_dir / f"{universe}_inc_matrix.npz"),
        "Correlation Matrix": str(out_dir / f"{universe}_corr_matrix.npz"),
        "Wikidata Matrix": str(out_dir / f"{universe}_wikidata_matrix.npz"),
        "Sector and Industry Relations": str(out_dir / f"{universe}_sector_industry_matrix.npz"),
    }


def dataset_base_path(universe: str) -> str:
    return str(DATA_DIR / universe / f"{universe}.csv")



def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# =============================================================================
# Dependency inspection
# =============================================================================

def _expand_universe_targets(targets: list[str] | tuple[str, ...]) -> list[str]:
    ordered = []
    visited = set()

    def visit(target_name: str) -> None:
        if target_name in visited:
            return

        if target_name not in TARGETS:
            valid = ", ".join(sorted(TARGETS))
            raise ValueError(
                f"Unknown universe target '{target_name}'. Valid targets: {valid}"
            )

        for dep in TARGETS[target_name].deps:
            visit(dep)

        visited.add(target_name)
        ordered.append(target_name)

    for target_name in targets:
        visit(target_name)

    return ordered


def _expand_market_targets(targets: list[str] | tuple[str, ...]) -> list[str]:
    ordered = []
    visited = set()

    def visit(target_name: str) -> None:
        if target_name in visited:
            return

        if target_name not in MARKET_TARGETS:
            valid = ", ".join(sorted(MARKET_TARGETS))
            raise ValueError(
                f"Unknown market target '{target_name}'. Valid targets: {valid}"
            )

        for dep in MARKET_TARGETS[target_name].deps:
            visit(dep)

        visited.add(target_name)
        ordered.append(target_name)

    for target_name in targets:
        visit(target_name)

    return ordered


def _collect_required_files(
    dataset: str,
    universe_targets: list[str],
    market_targets: list[list[str]] | list[tuple[str, str]],
) -> list[dict]:
    rows = []

    universe_ctx = PipelineContext(
        universe=dataset,
        base_dir=EVALUATION_DIR,
    )

    for target_name in _expand_universe_targets(universe_targets):
        target = TARGETS[target_name]

        for path in target.outputs(universe_ctx):
            path = Path(path)

            rows.append(
                {
                    "scope": "dataset",
                    "target": target_name,
                    "path": path,
                    "exists": path.exists(),
                }
            )

    market_targets_by_nation = defaultdict(list)

    for nation, target_name in market_targets:
        market_targets_by_nation[nation].append(target_name)

    for nation, targets in market_targets_by_nation.items():
        market_ctx = MarketContext(
            nation=nation,
            base_dir=EVALUATION_DIR,
        )

        for target_name in _expand_market_targets(targets):
            target = MARKET_TARGETS[target_name]

            for path in target.outputs(market_ctx):
                path = Path(path)

                rows.append(
                    {
                        "scope": "market",
                        "target": target_name,
                        "path": path,
                        "exists": path.exists(),
                    }
                )

    unique_rows = []
    seen_paths = set()

    for row in rows:
        path_key = str(row["path"])

        if path_key in seen_paths:
            continue

        seen_paths.add(path_key)
        unique_rows.append(row)

    return unique_rows


def _format_required_files_report(
    rows: list[dict],
    unsupported_files: list[str] | None = None,
    selected_models: list[str] | None = None,
) -> str:
    unsupported_files = unsupported_files or []
    selected_models = selected_models or []

    if not rows and not unsupported_files:
        return "No required files found."

    lines = []

    if selected_models:
        lines.append(f"Models: {', '.join(selected_models)}")
        lines.append("")

    existing = sum(1 for row in rows if row["exists"])
    missing = len(rows) - existing

    lines.append(f"ℹ️ Required files: {len(rows)} | Available: {existing} | Missing: {missing}")
    lines.append("")

    for row in rows:
        status = "✅ OK" if row["exists"] else "⛔ MISSING"
        path = _relative_path(row["path"])

        lines.append(
            f"[{status}] {row['target']} -> {path}"
        )

    if unsupported_files:
        lines.append("")
        lines.append("Files configured for the selected model(s) but not linked to a download target:")

        for file_name in unsupported_files:
            lines.append(f"[UNSUPPORTED] {file_name}")

    return "\n".join(lines)


def inspect_required_files(
    dataset: str | None,
    model: str | list[str] | tuple[str, ...] | None,
):
    models = _normalize_models(model)

    if not dataset:
        return (
            "",
            gr.update(value="Download Missing Files", interactive=False),
            [],
            [],
        )

    if not models:
        return (
            "",
            gr.update(value="Download Missing Files", interactive=False),
            [],
            [],
        )

    try:
        universe_targets, market_targets, unsupported_files = _targets_from_model_config(
            dataset=dataset,
            model=models,
        )

        rows = _collect_required_files(
            dataset=dataset,
            universe_targets=universe_targets,
            market_targets=market_targets,
        )

        report = _format_required_files_report(
            rows=rows,
            unsupported_files=unsupported_files,
            selected_models=models,
        )

        missing_count = sum(1 for row in rows if not row["exists"])

        if missing_count == 0:
            button = gr.update(
                value="No Missing Files",
                interactive=False,
            )
        else:
            button = gr.update(
                value=f"Download Missing Files ({missing_count})",
                interactive=True,
            )

        return (
            report,
            button,
            universe_targets,
            market_targets,
        )

    except Exception as exc:
        return (
            f"Error while inspecting required files: {exc}",
            gr.update(value="Download Missing Files", interactive=False),
            [],
            [],
        )

def download_missing_files(
    dataset: str | None,
    model: str | list[str] | tuple[str, ...] | None,
    universe_targets: list[str] | None,
    market_targets: list[list[str]] | None,
):
    models = _normalize_models(model)

    if not dataset:
        return (
            "",
            gr.update(value="Download Missing Files", interactive=False),
            [],
            [],
        )

    if not models:
        return (
            "",
            gr.update(value="Download Missing Files", interactive=False),
            [],
            [],
        )

    if not universe_targets and not market_targets:
        universe_targets, market_targets, _ = _targets_from_model_config(
            dataset=dataset,
            model=models,
        )

    universe_targets = universe_targets or []
    market_targets = market_targets or []

    try:
        if universe_targets:
            run_universe_pipeline(
                dataset,
                targets=universe_targets,
                force=False,
                verbose=True,
                workers=4,
            )

        market_targets_by_nation = defaultdict(list)

        for nation, target_name in market_targets:
            market_targets_by_nation[nation].append(target_name)

        for nation, targets in market_targets_by_nation.items():
            run_market_pipeline(
                nation,
                targets=targets,
                force=False,
                verbose=True,
                workers=4,
            )

    except Exception as exc:
        report, button, universe_state, market_state = inspect_required_files(
            dataset,
            models,
        )

        if report:
            report = f"{report}\n\nDownload error: {exc}"
        else:
            report = f"Download error: {exc}"

        return (
            report,
            button,
            universe_state,
            market_state,
        )

    report, button, universe_state, market_state = inspect_required_files(
        dataset,
        models,
    )

    return (
        report,
        button,
        universe_state,
        market_state,
    )
# =============================================================================
# Cache data table html based on file path and modification time
# =============================================================================

def _path_signature(path: str) -> int:
    p = Path(path)

    if not p.exists():
        return 0

    if p.is_file():
        return p.stat().st_mtime_ns

    mtimes = [p.stat().st_mtime_ns]

    for child in p.rglob("*"):
        try:
            if child.is_file():
                mtimes.append(child.stat().st_mtime_ns)
        except OSError:
            pass

    return max(mtimes) if mtimes else p.stat().st_mtime_ns


@lru_cache(maxsize=256)
@lru_cache(maxsize=256)
def _load_preview_file_page_cached(
    path: str,
    signature: int,
    page: int,
    page_size: int,
    start_year=None,
) -> tuple[str, str, bool, bool, int]:
    page = max(0, int(page))

    if path.endswith(".csv"):
        df, total_items = _csv_preview_page_df(
            path=path,
            page=page,
            page_size=page_size,
            start_year=start_year,
        )

    elif path.endswith(".json"):
        df, total_items = _json_preview_page_df(
            path=path,
            page=page,
            page_size=page_size,
        )

    elif path.endswith(".npz"):
        df, total_items = _npz_preview_page_df(path)
        page = 0

    elif os.path.isdir(path):
        df, total_items = _directory_json_preview_page_df(
            path=path,
            page=page,
            page_size=page_size,
        )

    else:
        return (
            empty_preview_table_html(f"Format not supported: {path}"),
            "<span>0 / 0</span>",
            False,
            False,
            0,
        )

    total_pages = max(math.ceil(total_items / page_size), 1)
    page = max(0, min(page, total_pages - 1))

    if df is None or df.empty:
        return (
            empty_preview_table_html("No rows available for this selection."),
            "<span>0 / 0</span>",
            False,
            False,
            0,
        )

    return (
        preview_table_html(df),
        f"<span>{page + 1} / {total_pages}</span>",
        page > 0,
        page < total_pages - 1,
        page,
    )
# =============================================================================
# Preview helpers
# =============================================================================

def empty_preview_table_html(message: str = "Select a dataset and/or a model to preview data.") -> str:
    return f"""
    <div class="preview-table-wrap">
        <div class="preview-empty">
            <div>{escape(message)}</div>
        </div>
    </div>
    """


def preview_table_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return empty_preview_table_html()

    headers = ["#"] + [str(col) for col in df.columns]

    th_cells = "".join(
        f"<th>{escape(col)}</th>" for col in headers
    )

    rows = []
    for display_idx, (_, row) in enumerate(df.iterrows()):
        td_cells = [f"<td class='row-index'>{display_idx}</td>"]

        td_cells.extend(
            f"<td>{escape(str(v)) if pd.notna(v) else ''}</td>"
            for v in row
        )

        rows.append(f"<tr>{''.join(td_cells)}</tr>")

    return f"""
    <div class="preview-table-wrap">
        <div class="preview-table-scroll">
            <table class="preview-html-table">
                <thead>
                    <tr>{th_cells}</tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
    </div>
    """


def get_preview_page(
        df: pd.DataFrame,
        page: int,
) -> tuple[str, str, bool, bool, int]:

    if df is None or df.empty:
        return (
            empty_preview_table_html(),
            _page_label(0, 0),
            False,
            False,
            0,
        )

    total_pages = max(1, (len(df) + PREVIEW_PAGE_SIZE - 1) // PREVIEW_PAGE_SIZE)
    page = max(0, min(page, total_pages - 1))

    start = page * PREVIEW_PAGE_SIZE
    end = start + PREVIEW_PAGE_SIZE
    html = preview_table_html(df.iloc[start:end].reset_index(drop=True))

    label = _page_label(page + 1, total_pages)
    return html, label, page > 0, page < total_pages - 1, page


def _page_label(current: int, total: int) -> str:
    return f"<span>{current} / {total}</span>"



@lru_cache(maxsize=128)
def _csv_total_rows_cached(path: str, signature: int) -> int:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        # -1 per header
        return max(sum(1 for _ in f) - 1, 0)


# def _csv_preview_page_df(
#     path: str,
#     page: int,
#     page_size: int,
# ) -> tuple[pd.DataFrame, int]:
#     signature = _path_signature(path)
#
#     total_items = _csv_total_rows_cached(path, signature)
#
#     start = page * page_size
#
#     df = pd.read_csv(
#         path,
#         skiprows=range(1, start + 1),
#         nrows=page_size,
#     )
#
#     return df, total_items
def _csv_preview_page_df(
    path: str,
    page: int,
    page_size: int,
    start_year=None,
) -> tuple[pd.DataFrame, int]:
    signature = _path_signature(path)
    start_year = _normalize_start_year(start_year)

    columns = _csv_columns_cached(path, signature)
    date_col = _date_column_name(columns)

    # Caso normale: nessun filtro anno oppure CSV senza colonna DATE
    if start_year is None or date_col is None:
        total_items = _csv_total_rows_cached(path, signature)
        start = page * page_size

        df = pd.read_csv(
            path,
            skiprows=range(1, start + 1),
            nrows=page_size,
        )

        return df, total_items

    # Caso filtrato: DATE >= start_year
    target_start = page * page_size
    target_end = target_start + page_size

    total_filtered = 0
    page_chunks = []

    for chunk in pd.read_csv(path, chunksize=50_000):
        parsed_dates = pd.to_datetime(chunk[date_col], errors="coerce")
        mask = parsed_dates.dt.year >= start_year

        filtered_chunk = chunk.loc[mask]
        chunk_count = len(filtered_chunk)

        if chunk_count == 0:
            continue

        chunk_global_start = total_filtered
        chunk_global_end = total_filtered + chunk_count

        if chunk_global_end > target_start and chunk_global_start < target_end:
            local_start = max(target_start - chunk_global_start, 0)
            local_end = max(target_end - chunk_global_start, 0)
            page_chunks.append(filtered_chunk.iloc[local_start:local_end])

        total_filtered += chunk_count

    if page_chunks:
        df = pd.concat(page_chunks, ignore_index=True).head(page_size)
    else:
        df = pd.DataFrame(columns=columns)

    return df, total_filtered

def _safe_json_load(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)


def _flatten_json_records(file_name: str, payload) -> list[dict]:
    if isinstance(payload, dict):
        rows = pd.json_normalize(payload, sep=".").to_dict(orient="records")

        if not rows:
            return [{"file_name": file_name}]

        row = rows[0]
        row["file_name"] = file_name
        return [row]

    if isinstance(payload, list):
        if len(payload) == 0:
            return [
                {
                    "file_name": file_name,
                    "message": "Empty JSON list",
                }
            ]

        if all(isinstance(item, dict) for item in payload):
            rows = pd.json_normalize(payload, sep=".").to_dict(orient="records")

            for row in rows:
                row["file_name"] = file_name

            return rows

        return [
            {
                "file_name": file_name,
                "raw_json": json.dumps(payload, ensure_ascii=False),
            }
        ]

    return [
        {
            "file_name": file_name,
            "raw_json": json.dumps(payload, ensure_ascii=False),
        }
    ]


def _json_preview_page_df(
    path: str,
    page: int = 0,
    page_size: int = PREVIEW_PAGE_SIZE,
) -> tuple[pd.DataFrame, int]:
    payload, err = _safe_json_load(path)

    if err is not None:
        return pd.DataFrame({"file_name": [Path(path).name], "error": [err]}), 1

    rows = _flatten_json_records(Path(path).name, payload)
    total_rows = len(rows)

    page = max(0, int(page))
    start = page * page_size
    end = start + page_size

    return pd.DataFrame(rows[start:end]), total_rows


def _directory_json_preview_page_df(
    path: str,
    page: int = 0,
    page_size: int = PREVIEW_PAGE_SIZE,
    max_rows_per_file: int = 20,
) -> tuple[pd.DataFrame, int]:
    json_files = sorted(
        [
            f for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and f.lower().endswith(".json")
        ]
    )

    total_files = len(json_files)

    if total_files == 0:
        return pd.DataFrame({"message": [f"No JSON files found in: {path}"]}), 0

    page = max(0, int(page))
    start_file = page * page_size
    end_file = start_file + page_size

    selected_files = json_files[start_file:end_file]

    rows = []

    for file_name in selected_files:
        full_path = os.path.join(path, file_name)
        payload, err = _safe_json_load(full_path)

        if err is not None:
            rows.append(
                {
                    "file_name": file_name,
                    "error": err,
                }
            )
            continue

        file_rows = _flatten_json_records(file_name, payload)
        rows.extend(file_rows[:max_rows_per_file])

    if not rows:
        return pd.DataFrame({"message": [f"No JSON rows found in: {path}"]}), total_files

    df = pd.DataFrame(rows)

    preferred = [
        "file_name",
        "date",
        "title",
        "source",
        "ticker",
        "content",
        "summary",
        "url",
        "error",
    ]

    ordered = [c for c in preferred if c in df.columns] + [
        c for c in df.columns if c not in preferred
    ]

    return df[ordered], total_files


def _npz_preview_page_df(path: str) -> tuple[pd.DataFrame, int]:
    rows = []

    try:
        with np.load(path, allow_pickle=True) as data:
            for key in data.files:
                arr = data[key]

                preview_value = ""

                try:
                    flat = arr.ravel()
                    preview_value = str(flat[:5].tolist())
                except Exception:
                    preview_value = ""

                rows.append(
                    {
                        "array": key,
                        "shape": str(arr.shape),
                        "dtype": str(arr.dtype),
                        "preview": preview_value,
                    }
                )

    except Exception as exc:
        rows.append(
            {
                "error": str(exc),
            }
        )

    return pd.DataFrame(rows), len(rows)

def _normalize_start_year(start_year):
    if start_year in [None, "", "All Years"]:
        return None

    try:
        return int(start_year)
    except Exception:
        return None


def _date_column_name(columns) -> str | None:
    for col in columns:
        if str(col).strip().upper() == "DATE":
            return col
    return None


@lru_cache(maxsize=128)
def _csv_columns_cached(path: str, signature: int) -> tuple[str, ...]:
    return tuple(pd.read_csv(path, nrows=0).columns)


@lru_cache(maxsize=128)
def _csv_available_years_cached(path: str, signature: int) -> tuple[int, ...]:
    columns = _csv_columns_cached(path, signature)
    date_col = _date_column_name(columns)

    if date_col is None:
        return tuple()

    try:
        date_series = pd.read_csv(path, usecols=[date_col])[date_col]
        parsed_dates = pd.to_datetime(date_series, errors="coerce")
        years = sorted(parsed_dates.dropna().dt.year.unique().astype(int).tolist())
        return tuple(years)
    except Exception:
        return tuple()


def refresh_preview_year_filter(universe: str, preview_name: str):
    if not universe or not preview_name:
        return gr.update(choices=[], value=None, visible=False, interactive=False)

    file_map = _build_file_map(universe)
    path = file_map.get(preview_name)

    if not path or not Path(path).exists():
        return gr.update(choices=[], value=None, visible=False, interactive=False)

    path_obj = Path(path)

    if path_obj.suffix.lower() != ".csv":
        return gr.update(choices=[], value=None, visible=False, interactive=False)

    path = str(path_obj.resolve())
    signature = _path_signature(path)

    years = list(_csv_available_years_cached(path, signature))

    if not years:
        return gr.update(choices=[], value=None, visible=False, interactive=False)

    choices = ["All Years"] + [str(year) for year in years]

    return gr.update(
        choices=choices,
        value="All Years",
        visible=True,
        interactive=True,
    )

def load_preview_file_page(
    universe: str,
    preview_name: str,
    page: int = 0,
    start_year=None,
    page_size: int = PREVIEW_PAGE_SIZE,
):
    if not universe or not preview_name:
        return (
            empty_preview_table_html(),
            "<span>0 / 0</span>",
            gr.update(interactive=False),
            gr.update(interactive=False),
            0,
        )

    file_map = _build_file_map(universe)
    path = file_map.get(preview_name)

    if not path or not os.path.exists(path):
        return (
            empty_preview_table_html(f"File not found: {preview_name}"),
            "<span>0 / 0</span>",
            gr.update(interactive=False),
            gr.update(interactive=False),
            0,
        )

    path = str(Path(path).resolve())
    signature = _path_signature(path)
    start_year = _normalize_start_year(start_year)

    html, label, can_prev, can_next, page = _load_preview_file_page_cached(
        path=path,
        signature=signature,
        page=int(page),
        page_size=int(page_size),
        start_year=start_year,
    )

    return (
        html,
        label,
        gr.update(interactive=can_prev),
        gr.update(interactive=can_next),
        page,
    )

def reset_preview_page(universe: str, preview_name: str, start_year=None):
    return load_preview_file_page(
        universe=universe,
        preview_name=preview_name,
        page=0,
        start_year=start_year,
    )


def next_preview_page(universe: str, preview_name: str, current_page: int, start_year=None):
    return load_preview_file_page(
        universe=universe,
        preview_name=preview_name,
        page=int(current_page) + 1,
        start_year=start_year,
    )


def prev_preview_page(universe: str, preview_name: str, current_page: int, start_year=None):
    return load_preview_file_page(
        universe=universe,
        preview_name=preview_name,
        page=int(current_page) - 1,
        start_year=start_year,
    )


def refresh_preview_controls(
    universe: str,
    universe_targets: list[str] | None = None,
    market_targets: list[list[str]] | None = None,
):
    """
    Updates the preview dropdown.

    Shows only .csv and .npz files.
    If universe_targets / market_targets are provided, it shows only the files
    required by the selected model and already available on disk.

    Otherwise it falls back to all available preview files for the dataset.
    """
    allowed_suffixes = {".csv", ".npz"}

    if not universe:
        return gr.update(choices=[], value=None)

    universe_targets = universe_targets or []
    market_targets = market_targets or []

    # Fallback: if no model/targets selected, show all available CSV/NPZ files
    if not universe_targets and not market_targets:
        file_map = _build_file_map(universe)

        options = [
            label
            for label, path in file_map.items()
            if Path(path).suffix.lower() in allowed_suffixes
            and Path(path).exists()
        ]

        default_preview = options[0] if options else None
        return gr.update(choices=options, value=default_preview)

    # Build label lookup from the preview file map, only for CSV/NPZ files
    file_map = _build_file_map(universe)

    path_to_label = {
        str(Path(path).resolve()): label
        for label, path in file_map.items()
        if Path(path).suffix.lower() in allowed_suffixes
    }

    try:
        rows = _collect_required_files(
            dataset=universe,
            universe_targets=universe_targets,
            market_targets=market_targets,
        )
    except Exception:
        return gr.update(choices=[], value=None)

    options = []

    for row in rows:
        path = Path(row["path"])

        if not path.exists():
            continue

        if path.suffix.lower() not in allowed_suffixes:
            continue

        resolved_path = str(path.resolve())
        label = path_to_label.get(resolved_path)

        if label is None:
            label = path.name

        if label not in options:
            options.append(label)

    default_preview = options[0] if options else None

    return gr.update(choices=options, value=default_preview)


