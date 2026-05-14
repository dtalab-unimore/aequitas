from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

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

PREVIEW_PAGE_SIZE = 10

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


def _targets_from_model_config(
    dataset: str,
    model: str | None,
) -> tuple[list[str], list[list[str]], list[str]]:
    """
    Reads Gradio/configs/model_to_data.json and converts required files into
    pipeline targets.

    Returns:
        universe_targets
        market_targets as [[nation, target], ...]
        unsupported_files
    """
    if not model or model not in CONFIGS:
        return ["prices"], [], []

    raw_files = CONFIGS.get(model, [])

    universe_targets = []
    market_targets = []
    unsupported_files = []

    nation = get_nation_from_universe(dataset)

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

        unsupported_files.append(resolved_name)

    if not universe_targets and not market_targets and not unsupported_files:
        universe_targets = ["prices"]

    return (
        _dedupe_keep_order(universe_targets),
        _dedupe_keep_order(market_targets),
        unsupported_files,
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


def dataset_exists(universe: str) -> bool:
    if not universe:
        return False

    return Path(dataset_base_path(universe)).exists()


def get_preview_options(universe: str) -> list[str]:
    if not universe:
        return []

    return [
        label
        for label, path in _build_file_map(universe).items()
        if Path(path).exists()
    ]


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
) -> str:
    unsupported_files = unsupported_files or []

    if not rows and not unsupported_files:
        return "No required files found."

    lines = []

    existing = sum(1 for row in rows if row["exists"])
    missing = len(rows) - existing

    lines.append(f"Required files: {len(rows)}")
    lines.append(f"Available: {existing}")
    lines.append(f"Missing: {missing}")
    lines.append("")

    for row in rows:
        status = "OK" if row["exists"] else "MISSING"
        path = _relative_path(row["path"])

        lines.append(
            f"[{status}] {row['target']} -> {path}"
        )

    if unsupported_files:
        lines.append("")
        lines.append("Files configured for the model but not linked to a download target:")

        for file_name in unsupported_files:
            lines.append(f"[UNSUPPORTED] {file_name}")

    return "\n".join(lines)


def inspect_required_files(dataset: str | None, model: str | None):
    if not dataset and not model:
        return (
            "Select a dataset and a model.",
            "",
            gr.update(value="Download Missing Files", interactive=False),
            [],
            [],
        )

    if not dataset:
        return (
            "Select a dataset.",
            "",
            gr.update(value="Download Missing Files", interactive=False),
            [],
            [],
        )

    if not model:
        return (
            "Select a model.",
            "",
            gr.update(value="Download Missing Files", interactive=False),
            [],
            [],
        )

    try:
        universe_targets, market_targets, unsupported_files = _targets_from_model_config(
            dataset=dataset,
            model=model,
        )

        rows = _collect_required_files(
            dataset=dataset,
            universe_targets=universe_targets,
            market_targets=market_targets,
        )

        report = _format_required_files_report(rows, unsupported_files)
        missing_count = sum(1 for row in rows if not row["exists"])

        if missing_count == 0:
            if unsupported_files:
                status = "All downloadable required files are available. Some configured files are unsupported."
            else:
                status = "All required files are already available."

            button = gr.update(
                value="No Missing Files",
                interactive=False,
            )
        else:
            status = f"{missing_count} required file(s) are missing."
            button = gr.update(
                value=f"Download Missing Files ({missing_count})",
                interactive=True,
            )

        return (
            status,
            report,
            button,
            universe_targets,
            market_targets,
        )

    except Exception as exc:
        return (
            f"File inspection failed: {exc}",
            "",
            gr.update(value="Download Missing Files", interactive=False),
            [],
            [],
        )


def download_missing_files(
    dataset: str | None,
    model: str | None,
    universe_targets: list[str] | None,
    market_targets: list[list[str]] | None,
):
    if not dataset:
        return (
            "Select a dataset first.",
            "",
            gr.update(value="Download Missing Files", interactive=False),
            [],
            [],
        )

    if not model:
        return (
            "Select a model first.",
            "",
            gr.update(value="Download Missing Files", interactive=False),
            [],
            [],
        )

    if not universe_targets and not market_targets:
        universe_targets, market_targets, _ = _targets_from_model_config(
            dataset=dataset,
            model=model,
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
        status, report, button, universe_state, market_state = inspect_required_files(
            dataset,
            model,
        )

        return (
            f"Download failed: {exc}",
            report,
            button,
            universe_state,
            market_state,
        )

    status, report, button, universe_state, market_state = inspect_required_files(
        dataset,
        model,
    )

    return (
        "Download completed. Required files are now available.",
        report,
        button,
        universe_state,
        market_state,
    )


# =============================================================================
# Preview helpers
# =============================================================================

@lru_cache(maxsize=256)
def _count_csv_rows(path: str) -> int:
    try:
        with open(path, "rb") as f:
            total_lines = sum(
                chunk.count(b"\n")
                for chunk in iter(lambda: f.read(1024 * 1024), b"")
            )

        return max(total_lines - 1, 0)

    except Exception:
        return 0


def _csv_preview_page_df(
    path: str,
    page: int = 0,
    page_size: int = PREVIEW_PAGE_SIZE,
) -> tuple[pd.DataFrame, int]:
    total_rows = _count_csv_rows(path)
    total_pages = max(math.ceil(total_rows / page_size), 1)

    page = max(0, min(int(page), total_pages - 1))
    start_row = page * page_size

    try:
        df = pd.read_csv(
            path,
            skiprows=range(1, start_row + 1),
            nrows=page_size,
        )
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()

    return df, total_rows


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


def load_preview_file_page(
    universe: str,
    preview_name: str,
    page: int = 0,
    page_size: int = PREVIEW_PAGE_SIZE,
):
    if not universe or not preview_name:
        return (
            pd.DataFrame(),
            "0 / 0",
            gr.update(interactive=False),
            gr.update(interactive=False),
            0,
        )

    path = _build_file_map(universe).get(preview_name)

    if not path or not os.path.exists(path):
        return (
            pd.DataFrame({"message": [f"File not found: {preview_name}"]}),
            "0 / 0",
            gr.update(interactive=False),
            gr.update(interactive=False),
            0,
        )

    page = max(0, int(page))

    if path.endswith(".csv"):
        df, total_items = _csv_preview_page_df(
            path=path,
            page=page,
            page_size=page_size,
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
            pd.DataFrame({"message": [f"Format not supported: {path}"]}),
            "0 / 0",
            gr.update(interactive=False),
            gr.update(interactive=False),
            0,
        )

    total_pages = max(math.ceil(total_items / page_size), 1)
    page = max(0, min(page, total_pages - 1))

    page_label = f"{page + 1} / {total_pages}"

    return (
        df,
        page_label,
        gr.update(interactive=page > 0),
        gr.update(interactive=page < total_pages - 1),
        page,
    )


def reset_preview_page(universe: str, preview_name: str):
    return load_preview_file_page(
        universe=universe,
        preview_name=preview_name,
        page=0,
    )


def next_preview_page(universe: str, preview_name: str, current_page: int):
    return load_preview_file_page(
        universe=universe,
        preview_name=preview_name,
        page=int(current_page) + 1,
    )


def prev_preview_page(universe: str, preview_name: str, current_page: int):
    return load_preview_file_page(
        universe=universe,
        preview_name=preview_name,
        page=int(current_page) - 1,
    )


def load_preview_file(universe: str, preview_name: str) -> pd.DataFrame:
    df, _, _, _, _ = load_preview_file_page(
        universe=universe,
        preview_name=preview_name,
        page=0,
    )

    return df


def update_preview(universe: str, preview_name: str) -> pd.DataFrame:
    return load_preview_file(universe, preview_name)


def refresh_preview_controls(
    universe: str,
    universe_targets: list[str] | None = None,
    market_targets: list[list[str]] | None = None,
):
    """
    Updates the preview dropdown.

    If universe_targets / market_targets are provided, it shows only the files
    required by the selected model and already available on disk.

    Otherwise it falls back to all available preview files for the dataset.
    """
    if not universe:
        return gr.update(choices=[], value=None)

    universe_targets = universe_targets or []
    market_targets = market_targets or []

    # Fallback: if no model/targets selected, show all available files
    if not universe_targets and not market_targets:
        options = get_preview_options(universe)
        default_preview = options[0] if options else None
        return gr.update(choices=options, value=default_preview)

    # Build label lookup from the preview file map
    file_map = _build_file_map(universe)

    path_to_label = {
        str(Path(path).resolve()): label
        for label, path in file_map.items()
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

        resolved_path = str(path.resolve())
        label = path_to_label.get(resolved_path)

        if label is None:
            # fallback if the file exists but is not explicitly mapped
            label = path.name

        if label not in options:
            options.append(label)

    default_preview = options[0] if options else None

    return gr.update(choices=options, value=default_preview)


# =============================================================================
# Backward-compatible helpers for old UI pieces
# =============================================================================

def update_existing_button(universe: str):
    if dataset_exists(universe):
        return gr.update(
            value="Use Existing Dataset",
            interactive=True,
            variant="secondary",
        )

    return gr.update(
        value="Dataset Not Available",
        interactive=False,
        variant="stop",
    )


def update_datasets(market):
    if market == "EU":
        datasets = ["sx5e", "sxxp"]
    elif market == "USA":
        datasets = ["sp500", "nasdaq100", "dji"]
    else:
        datasets = []

    return gr.update(choices=datasets, value=datasets[0] if datasets else None)


def on_dataset_change(dataset_value):
    btn_update = update_existing_button(dataset_value)

    return (
        btn_update,
        "",
        gr.update(choices=[], value=None),
        pd.DataFrame(),
        "",
        gr.update(interactive=False),
        gr.update(interactive=False),
        0,
    )


def on_market_change(market_value):
    new_dataset_update = update_datasets(market_value)

    return (
        new_dataset_update,
        "",
        gr.update(choices=[], value=None),
        pd.DataFrame(),
        "",
        gr.update(interactive=False),
        gr.update(interactive=False),
        0,
    )


def use_existing_dataset(universe: str):
    if not universe:
        return (
            "Select a dataset.",
            gr.update(choices=[], value=None),
            pd.DataFrame(),
            "0 / 0",
            gr.update(interactive=False),
            gr.update(interactive=False),
            0,
            update_existing_button(universe),
        )

    if not dataset_exists(universe):
        return (
            f"The dataset '{universe}' does not exist.",
            gr.update(choices=[], value=None),
            pd.DataFrame(),
            "0 / 0",
            gr.update(interactive=False),
            gr.update(interactive=False),
            0,
            update_existing_button(universe),
        )

    options = get_preview_options(universe)
    default_preview = options[0] if options else None

    if default_preview:
        preview_df, page_label, prev_btn, next_btn, page_state = reset_preview_page(
            universe,
            default_preview,
        )
    else:
        preview_df = pd.DataFrame()
        page_label = "0 / 0"
        prev_btn = gr.update(interactive=False)
        next_btn = gr.update(interactive=False)
        page_state = 0

    return (
        f"Dataset '{universe}' already present, preview loaded.",
        gr.update(choices=options, value=default_preview),
        preview_df,
        page_label,
        prev_btn,
        next_btn,
        page_state,
        update_existing_button(universe),
    )


def run_data_pipeline(universe: str):
    """
    Old button compatibility.
    New UI should prefer inspect_required_files + download_missing_files.
    """
    if not universe:
        return (
            "Select a dataset.",
            update_existing_button(universe),
        )

    try:
        run_universe_pipeline(
            universe,
            targets=["prices"],
            force=False,
            verbose=True,
            workers=4,
        )

        return (
            f"Dataset '{universe}' downloaded.",
            update_existing_button(universe),
        )

    except Exception as exc:
        return (
            f"Error during pipeline: {exc}",
            update_existing_button(universe),
        )


def get_files(universe, models):
    """
    Old Check Files button compatibility.
    """
    if not universe:
        return "Select a dataset."

    if not models:
        return "Select a model."

    _, report, _, _, _ = inspect_required_files(universe, models)
    return report