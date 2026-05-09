import os
import json
import gradio as gr
import pandas as pd
import numpy as np
import math
from functools import lru_cache

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DRY_RUN = False  # True = does not call paid APIs; generates mock data

# -------------------------------------------------
# PATH HELPERS
# -------------------------------------------------
def dataset_base_path(universe: str) -> str:
     return os.path.join("./Evaluation/data", universe, f"{universe}.csv")


def dataset_exists(universe: str) -> bool:
     if not universe:
         return False
     return os.path.exists(dataset_base_path(universe))

def _ensure_dir(path: str) -> None:
     os.makedirs(path, exist_ok=True)

def read_configs():
    with open("./Gradio/configs/model_to_data.json", "r") as f:
        configs = json.load(f)
    return configs

PREVIEW_PAGE_SIZE = 10
CONFIGS = read_configs()
UNIVERSE_TO_NATION = {
    "nasdaq100": "us",
    "sp500": "us",
    "dji": "us",
    "sxxp": "eu",
    "sx5e": "eu"
}

def _build_file_map(universe: str) -> dict:
    out_dir = os.path.join("./Evaluation/data", universe)
    data_dir = "./Evaluation/data"
    return {
        "Dataset":                 os.path.join(out_dir,  f"{universe}.csv"),
        "Info":                    os.path.join(out_dir,  f"{universe}_info.csv"),
        "Market Cap":              os.path.join(out_dir,  f"{universe}_market_cap.csv"),
        "Alpha158":                os.path.join(out_dir,  f"{universe}_alpha158.csv"),
        "Alpha360":                os.path.join(out_dir,  f"{universe}_alpha360.csv"),
        "Technicals":              os.path.join(out_dir,  f"{universe}_tech.csv"),
        "US Indexes":              os.path.join(data_dir, "us_indexes.csv"),
        "EU Indexes":              os.path.join(data_dir, "eu_indexes.csv"),
        "US Market Info":          os.path.join(data_dir, "us_market.csv"),
        "EU Market Info":          os.path.join(data_dir, "eu_market.csv"),
        "CNNPred Market Features": os.path.join(data_dir, "cnnpred_market.csv"),
        "News Relations":          os.path.join(data_dir, "news"),
    }


@lru_cache(maxsize=256)
def _count_csv_rows(path: str) -> int:
    """
    Counts CSV rows without loading the full file into memory.
    Subtracts 1 for the header.
    """
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
    """
    Loads only one page from a CSV file.
    """
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
    """
    Returns a list of table rows based on the JSON content.
    - dict -> 1 row
    - list[dict] -> one row per element
    - other types -> 1 row with raw_json
    """
    if isinstance(payload, dict):
        row = pd.json_normalize(payload, sep=".").to_dict(orient="records")[0]
        row["file_name"] = file_name
        return [row]

    if isinstance(payload, list):
        if len(payload) == 0:
            return [{
                "file_name": file_name,
                "message": "Empty JSON list",
            }]

        if all(isinstance(item, dict) for item in payload):
            rows = pd.json_normalize(payload, sep=".").to_dict(orient="records")
            for row in rows:
                row["file_name"] = file_name
            return rows

        return [{
            "file_name": file_name,
            "raw_json": json.dumps(payload, ensure_ascii=False),
        }]

    return [{
        "file_name": file_name,
        "raw_json": json.dumps(payload, ensure_ascii=False),
    }]


def _directory_json_preview_page_df(
    path: str,
    page: int = 0,
    page_size: int = PREVIEW_PAGE_SIZE,
    max_rows_per_file: int = 20,
) -> tuple[pd.DataFrame, int]:
    """
    Pagetes JSON files inside a directory.
    Each JSON file can produce multiple rows.
    For speed, Pagetion is file-based first, then flattened.
    """
    json_files = sorted([
        f for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and f.lower().endswith(".json")
    ])

    total_files = len(json_files)

    if total_files == 0:
        return pd.DataFrame({
            "message": [f"No JSON files found in: {path}"]
        }), 0

    page = max(0, int(page))
    start_file = page * page_size
    end_file = start_file + page_size

    selected_files = json_files[start_file:end_file]

    rows = []

    for file_name in selected_files:
        full_path = os.path.join(path, file_name)
        payload, err = _safe_json_load(full_path)

        if err is not None:
            rows.append({
                "file_name": file_name,
                "error": err,
            })
            continue

        file_rows = _flatten_json_records(file_name, payload)
        rows.extend(file_rows[:max_rows_per_file])

    if not rows:
        return pd.DataFrame({
            "message": [f"No JSON rows found in: {path}"]
        }), total_files

    df = pd.DataFrame(rows)

    cols = list(df.columns)
    preferred = [
        "file_name", "date", "title", "source", "ticker",
        "content", "summary", "url", "error"
    ]
    ordered = [c for c in preferred if c in cols] + [
        c for c in cols if c not in preferred
    ]

    df = df[ordered]

    return df, total_files


def load_preview_file_page(
    universe: str,
    preview_name: str,
    page: int = 0,
    page_size: int = PREVIEW_PAGE_SIZE,
):
    """
    Loads one preview page and returns:
    dataframe,
    page label,
    previous button update,
    next button update,
    normalized current page.
    """
    if not preview_name:
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
        # item_label = "righe"

    elif os.path.isdir(path):
        df, total_items = _directory_json_preview_page_df(
            path=path,
            page=page,
            page_size=page_size,
        )
        # item_label = "file JSON"

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

    # start_item = page * page_size + 1 if total_items > 0 else 0
    # end_item = min((page + 1) * page_size, total_items)

    page_label = f"{page + 1} / {total_pages}"
        # f"— {item_label} {start_item}-{end_item} di {total_items}"


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
    """
    Backward-compatible function.
    It now returns only the first preview page.
    """
    df, _, _, _, _ = load_preview_file_page(
        universe=universe,
        preview_name=preview_name,
        page=0,
    )
    return df


def update_preview(universe: str, preview_name: str) -> pd.DataFrame:
    """
    Backward-compatible function.
    Prefer reset_preview_page in the Gradio UI.
    """
    return load_preview_file(universe, preview_name)


def refresh_preview_controls(universe: str):
    """
    Only refreshes the dropdown options.
    Does not load the full preview automatically.
    """
    if not universe:
        return gr.update(choices=[], value=None)

    options = get_preview_options(universe)
    default_preview = options[0] if options else None

    return gr.update(choices=options, value=default_preview)

# -------------------------------------------------
# BUTTON STATE
# -------------------------------------------------
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
    else:
        datasets = ["sp500", "nasdaq100", "dji"]
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
            "❌ Select a dataset",
            gr.update(choices=[], value=None),
            pd.DataFrame(),
            "Page 0 / 0",
            gr.update(interactive=False),
            gr.update(interactive=False),
            0,
            update_existing_button(universe),
        )

    if not dataset_exists(universe):
        return (
            f"❌ The dataset '{universe}' does not exist",
            gr.update(choices=[], value=None),
            pd.DataFrame(),
            "Page 0 / 0",
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
        page_label = "Page 0 / 0"
        prev_btn = gr.update(interactive=False)
        next_btn = gr.update(interactive=False)
        page_state = 0

    return (
        f"✅ Dataset '{universe}' already present, preview loaded",
        gr.update(choices=options, value=default_preview),
        preview_df,
        page_label,
        prev_btn,
        next_btn,
        page_state,
        update_existing_button(universe),
    )


# -------------------------------------------------
# MOCK DATA CREATION
# -------------------------------------------------
def _mock_create_dataset_files(universe: str) -> None:
    """
    Creates fake files compatible with the UI preview.
    This does not replace the actual logic in Evaluation/main.py:
    it is intended solely for demonstration purposes without using paid APIs.
    """
    out_dir = os.path.join("./Evaluation/data", universe)
    _ensure_dir(out_dir)
    _ensure_dir("./Evaluation/data")

    rng = np.random.default_rng(42)
    dates = pd.date_range(start="2020-01-01", periods=120, freq="D")
    tickers = ["AAA", "BBB", "CCC", "DDD"]

    # --- price data ---
    rows = []
    for t in tickers:
        base = 100 + rng.normal(0, 2)
        prices = base + np.cumsum(rng.normal(0, 1, size=len(dates)))
        for d, p in zip(dates, prices):
            rows.append({
                "date": d,
                "instrument": t,
                "adj_open":  p + rng.normal(0, 0.5),
                "adj_high":  p + abs(rng.normal(0, 1)),
                "adj_low":   p - abs(rng.normal(0, 1)),
                "adj_close": p,
                "volume":    int(rng.integers(1000, 10000)),
            })

    df_prices = pd.DataFrame(rows)
    df_prices.to_csv(os.path.join(out_dir, f"{universe}.csv"), index=False)

    # --- info ---
    pd.DataFrame({
        "instrument": tickers,
        "Name":        [f"Company {t}" for t in tickers],
        "GicSector":   ["Tech", "Finance", "Tech", "Energy"],
        "GicIndustry": ["Software", "Banks", "Hardware", "Oil & Gas"],
    }).to_csv(os.path.join(out_dir, f"{universe}_info.csv"), index=False)

    # --- market cap ---
    df_mc = df_prices[["date", "instrument"]].copy()
    df_mc["market_cap"] = rng.integers(1_000_000, 10_000_000, size=len(df_mc))
    df_mc.to_csv(os.path.join(out_dir, f"{universe}_market_cap.csv"), index=False)

    # --- feature files ---
    feature_cols = {
        "alpha158": ["alpha_001", "alpha_002", "alpha_003"],
        "alpha360": ["alpha_101", "alpha_102", "alpha_103"],
        "tech":     ["rsi", "macd", "atr"],
    }
    for suffix, cols in feature_cols.items():
        dff = df_prices[["date", "instrument"]].copy()
        for c in cols:
            dff[c] = rng.normal(0, 1, size=len(dff))
        dff.to_csv(os.path.join(out_dir, f"{universe}_{suffix}.csv"), index=False)

    # --- adjacency matrices ---
    _write_sector_matrix(out_dir, universe, tickers, rng)
    _write_corr_matrix(out_dir, universe, tickers, dates, rng)
    _write_wikidata_matrix(out_dir, universe, tickers, dates, rng)
    _write_incidence_matrix(out_dir, universe, tickers, rng)

    # --- market information ---
    for region in ["us", "eu"]:
        pd.DataFrame({
            "date": dates,
            f"{region}_index_1": 100 + np.cumsum(rng.normal(0, 1, len(dates))),
            f"{region}_index_2": 110 + np.cumsum(rng.normal(0, 1, len(dates))),
        }).to_csv(os.path.join("./Evaluation/data", f"{region}_indexes.csv"), index=False)

        pd.DataFrame({
            "date":              dates,
            "market_return":     rng.normal(0, 0.01, len(dates)),
            "market_volatility": np.abs(rng.normal(0.02, 0.01, len(dates))),
        }).to_csv(os.path.join("./Evaluation/data", f"{region}_market.csv"), index=False)

    # --- CNNPred market features ---
    pd.DataFrame({
        "date":           dates,
        "fred_1":         rng.normal(0, 1, len(dates)),
        "commodity_1":    rng.normal(0, 1, len(dates)),
        "world_index_1":  rng.normal(0, 1, len(dates)),
    }).to_csv("./Evaluation/data/cnnpred_market.csv", index=False)

    news_dir = os.path.join("./Evaluation/data", "news")
    _ensure_dir(news_dir)

    # --- fake news json files ---
    for i in range(25):
        payload = {
            "ticker": tickers[i % len(tickers)],
            "title": f"Mock news {i}",
            "date": str(dates[i].date()),
            "source": "demo",
            "sentiment": float(rng.normal(0, 1)),
            "entities": ["macro", "earnings"],
        }
        file_name = f"{tickers[i % len(tickers)]}.{i:02d}.json"
        with open(os.path.join(news_dir, file_name), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_sector_matrix(out_dir, universe, tickers, rng):
    adj = rng.integers(0, 2, size=(len(tickers), len(tickers), 2))
    np.savez_compressed(
        os.path.join(out_dir, f"{universe}_sector_industry_matrix.npz"),
        adj_matrix=adj,
        dates=np.array([np.nan]),
        relations=np.array(["same_sector", "same_industry"], dtype=object),
        tickers=np.array(tickers, dtype=object),
    )


def _write_corr_matrix(out_dir, universe, tickers, dates, rng):
    adj = rng.integers(0, 2, size=(10, len(tickers), len(tickers), 2))
    np.savez_compressed(
        os.path.join(out_dir, f"{universe}_corr_matrix.npz"),
        adj_matrix=adj,
        dates=pd.date_range("2020-01-01", periods=10, freq="7D").values,
        relations=np.array(["pos_corr", "neg_corr"], dtype=object),
        tickers=np.array(tickers, dtype=object),
    )


def _write_wikidata_matrix(out_dir, universe, tickers, dates, rng):
    adj = rng.integers(0, 2, size=(5, len(tickers), len(tickers), 3))
    np.savez_compressed(
        os.path.join(out_dir, f"{universe}_wikidata_matrix.npz"),
        adj_matrix=adj,
        dates=pd.date_range("2020-01-01", periods=5, freq="30D").values,
        relations=np.array(["parent", "subsidiary", "partner"], dtype=object),
        tickers=np.array(tickers, dtype=object),
    )


def _write_incidence_matrix(out_dir, universe, tickers, rng):
    inc = rng.integers(0, 2, size=(len(tickers), 6))
    np.savez_compressed(
        os.path.join(out_dir, f"{universe}_inc_matrix.npz"),
        inc_matrix=inc,
        relations=np.array([f"E{i}" for i in range(6)], dtype=object),
        tickers=np.array(tickers, dtype=object),
    )


# -------------------------------------------------
# PREVIEW HELPERS
# -------------------------------------------------

def get_preview_options(universe: str) -> list[str]:
    return [label for label, path in _build_file_map(universe).items() if os.path.exists(path)]


def _csv_preview_df(path: str, max_rows: int = 20) -> pd.DataFrame:
    return pd.read_csv(path).head(max_rows)


def _safe_json_load(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)

def _flatten_json_records(file_name: str, payload) -> list[dict]:
    """
    Returns a list of table rows based on the JSON content.
    - dict  -> 1 row
    - list[dict] -> one row per element
    - other types -> 1 row with raw_json
    """
    # Case 1: single JSON object
    if isinstance(payload, dict):
        row = pd.json_normalize(payload, sep=".").to_dict(orient="records")[0]
        row["file_name"] = file_name
        return [row]

    # Case 2: JSON object list
    if isinstance(payload, list):
        if len(payload) == 0:
            return [{
                "file_name": file_name,
                "message": "Empty JSON list",
            }]

        # if all or almost all are strings, split them into lines
        if all(isinstance(item, dict) for item in payload):
            rows = pd.json_normalize(payload, sep=".").to_dict(orient="records")
            for row in rows:
                row["file_name"] = file_name
            return rows

        # non-tabular list
        return [{
            "file_name": file_name,
            "raw_json": json.dumps(payload, ensure_ascii=False),
        }]

    # Case 3: type that cannot be handled in a table
    return [{
        "file_name": file_name,
        "raw_json": json.dumps(payload, ensure_ascii=False),
    }]

def _directory_json_preview_df(path: str, max_files: int = 20, max_rows_per_file: int = 20) -> pd.DataFrame:
    json_files = sorted(
        [
            f for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and f.lower().endswith(".json")
        ]
    )[:max_files]

    rows = []
    for file_name in json_files:
        full_path = os.path.join(path, file_name)
        payload, err = _safe_json_load(full_path)

        if err is not None:
            rows.append({
                "file_name": file_name,
                "error": err,
            })
            continue

        file_rows = _flatten_json_records(file_name, payload)

        # limits the number of lines per file, so the preview doesn't get too long
        rows.extend(file_rows[:max_rows_per_file])

    if not rows:
        return pd.DataFrame({
            "message": [f"No JSON files found in: {path}"]
        })

    df = pd.DataFrame(rows)

    cols = list(df.columns)
    preferred = [
        "file_name", "date", "title", "source", "ticker",
        "content", "summary", "url", "error"
    ]
    ordered = [c for c in preferred if c in cols] + [c for c in cols if c not in preferred]
    df = df[ordered]

    return df

def load_preview_file(universe: str, preview_name: str) -> pd.DataFrame:
    if not preview_name:
        return pd.DataFrame()

    path = _build_file_map(universe).get(preview_name)
    if not path or not os.path.exists(path):
        return pd.DataFrame({"message": [f"File not found: {preview_name}"]})

    if path.endswith(".csv"):
        return _csv_preview_df(path, max_rows=20)

    if os.path.isdir(path):
        return _directory_json_preview_df(path, max_files=20)

    return pd.DataFrame({"message": [f"Format not supported: {path}"]})


def update_preview(universe: str, preview_name: str) -> pd.DataFrame:
    return load_preview_file(universe, preview_name)

# -------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------
def run_data_pipeline(universe: str, progress=gr.Progress(track_tqdm=False)):
    if not universe:
        return (
            "❌ Select a dataset",
            update_existing_button(universe),
        )

    try:
        progress(0.0, desc="Pipeline initialisation...")

        if DRY_RUN:
            _run_mock_pipeline(universe, progress)
            status_msg = f"✅ DEMO pipeline completed for “{universe}” (no actual downloads performed)"
        else:
            _run_real_pipeline(universe, progress)
            status_msg = f"✅ Pipeline completed for '{universe}'"

        progress(0.97, desc="Loading preview...")
        options = get_preview_options(universe)
        default_preview = options[0] if options else None
        preview_df = load_preview_file(universe, default_preview) if default_preview else pd.DataFrame()

        progress(1.0, desc="Completed")
        return (
            status_msg,
            update_existing_button(universe),
        )

    except Exception as e:
        return (
            f"❌ Error during the pipeline: {str(e)}",
            update_existing_button(universe),
        )

def refresh_preview_controls(universe: str):
    if not universe:
        return gr.update(choices=[], value=None), pd.DataFrame()

    options = get_preview_options(universe)
    default_preview = options[0] if options else None
    preview_df = load_preview_file(universe, default_preview) if default_preview else pd.DataFrame()

    return (
        gr.update(choices=options, value=default_preview),
        preview_df,
    )

def _run_mock_pipeline(universe: str, progress) -> None:
    steps = [
        (0.10, f"Creating mock dataset for {universe}..."),
        (0.30, "Simulating feature extraction..."),
        (0.45, "Simulating compute relation..."),
        (0.60, "Simulating incidence matrix..."),
        (0.75, "Simulating market information..."),
        (0.90, "Simulating CNNPred market features..."),
    ]
    for pct, desc in steps:
        progress(pct, desc=desc)

    _mock_create_dataset_files(universe)


def _run_real_pipeline(universe: str, progress) -> None:
    from Evaluation import main as eval_main

    steps = [
        (0.05, f"1/6 Download dataset: {universe}", lambda: eval_main.demo_download_dataset(universe)),
        (0.25, f"2/6 Feature extraction: {universe}", lambda: eval_main.demo_feature_extraction(universe)),
        (0.45, f"3/6 Compute relation: {universe}", lambda: eval_main.demo_compute_relation(universe)),
        (0.65, f"4/6 Incidence matrix: {universe}", lambda: eval_main.demo_incidence_matrix(universe)),
        (0.80, "5/6 Market information", lambda: eval_main.demo_market_information()),
        (0.92, "6/6 CNNPred market features", lambda: eval_main.cnnpred_market_features()),
    ]
    for pct, desc, fn in steps:
        progress(pct, desc=desc)
        fn()

def get_models():
    return list(CONFIGS.keys())

def _build_file_lookup(universe: str) -> dict:
    file_map = _build_file_map(universe)

    return {
        os.path.basename(os.path.normpath(path)): {
            "label": label,
            "path": path,
        }
        for label, path in file_map.items()
    }

def get_nation_from_universe(universe: str) -> str:
    try:
        return UNIVERSE_TO_NATION[universe]
    except KeyError:
        raise ValueError(f"Unknown universe: {universe}")

def get_files(universe, models):
    raw_files = CONFIGS[models]

    nation = get_nation_from_universe(universe).lower()
    lookup = _build_file_lookup(universe)

    lines = [
        f"🌍 Universe: {universe}",
        f"🏳️ Nation: {nation.upper()}",
        "",
        "📁 File check:",
        ""
    ]

    for file_template in raw_files:
        filename = file_template.format(
            universe=universe,
            nation=nation
        )

        item = lookup.get(filename)

        if item is None:
            lines.append(f"❓ {filename} — not configured in file map")
            continue

        label = item["label"]
        path = item["path"]

        if os.path.exists(path):
            lines.append(f"✅ {label} ")
        else:
            lines.append(f"❌ {label} ")

    return "\n".join(lines)