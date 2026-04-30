import os
import sys
import json
import gradio as gr
import pandas as pd
import numpy as np

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DRY_RUN = False  # True = non chiama API a pagamento, crea dati mock

# -------------------------------------------------
# PATH HELPERS
# -------------------------------------------------
def dataset_base_path(universe: str) -> str:
    return os.path.join("../Evaluation/data", universe, f"{universe}.csv")


def dataset_exists(universe: str) -> bool:
    if not universe:
        return False
    return os.path.exists(dataset_base_path(universe))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


# -------------------------------------------------
# USE EXISTING DATASET
# -------------------------------------------------
def use_existing_dataset(universe: str):
    if not universe:
        return (
            "❌ Seleziona un dataset",
            gr.update(choices=[], value=None),
            pd.DataFrame(),
            update_existing_button(universe),
        )

    if not dataset_exists(universe):
        return (
            f"❌ Il dataset '{universe}' non è presente",
            gr.update(choices=[], value=None),
            pd.DataFrame(),
            update_existing_button(universe),
        )

    options = get_preview_options(universe)
    default_preview = options[0] if options else None
    preview_df = load_preview_file(universe, default_preview) if default_preview else pd.DataFrame()

    return (
        f"✅ Dataset '{universe}' già presente, preview caricata",
        gr.update(choices=options, value=default_preview),
        preview_df,
        update_existing_button(universe),
    )


# -------------------------------------------------
# MOCK DATA CREATION
# -------------------------------------------------
def _mock_create_dataset_files(universe: str) -> None:
    """
    Crea file fake compatibili con la preview UI.
    Non sostituisce la logica reale di Evaluation/main.py:
    serve solo per demo senza usare API a pagamento.
    """
    out_dir = os.path.join("../Evaluation/data", universe)
    _ensure_dir(out_dir)
    _ensure_dir("../Evaluation/data")

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
        }).to_csv(os.path.join("../Evaluation/data", f"{region}_indexes.csv"), index=False)

        pd.DataFrame({
            "date":              dates,
            "market_return":     rng.normal(0, 0.01, len(dates)),
            "market_volatility": np.abs(rng.normal(0.02, 0.01, len(dates))),
        }).to_csv(os.path.join("../Evaluation/data", f"{region}_market.csv"), index=False)

    # --- CNNPred market features ---
    pd.DataFrame({
        "date":           dates,
        "fred_1":         rng.normal(0, 1, len(dates)),
        "commodity_1":    rng.normal(0, 1, len(dates)),
        "world_index_1":  rng.normal(0, 1, len(dates)),
    }).to_csv("../Evaluation/data/cnnpred_market.csv", index=False)

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
def _build_file_map(universe: str) -> dict:
    out_dir = os.path.join("../Evaluation/data", universe)
    data_dir = "../Evaluation/data"
    return {
        "Dataset":                os.path.join(out_dir,  f"{universe}.csv"),
        "Info":                   os.path.join(out_dir,  f"{universe}_info.csv"),
        "Market Cap":             os.path.join(out_dir,  f"{universe}_market_cap.csv"),
        "Alpha158":               os.path.join(out_dir,  f"{universe}_alpha158.csv"),
        "Alpha360":               os.path.join(out_dir,  f"{universe}_alpha360.csv"),
        "Technicals":             os.path.join(out_dir,  f"{universe}_tech.csv"),
        "US Indexes":             os.path.join(data_dir, "us_indexes.csv"),
        "EU Indexes":             os.path.join(data_dir, "eu_indexes.csv"),
        "US Market Info":         os.path.join(data_dir, "us_market.csv"),
        "EU Market Info":         os.path.join(data_dir, "eu_market.csv"),
        "CNNPred Market Features": os.path.join(data_dir, "cnnpred_market.csv"),
        "News Relations":         os.path.join(data_dir,  "news"),
    }


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
    Restituisce una lista di righe tabellari a partire dal contenuto JSON.
    - dict  -> 1 riga
    - list[dict] -> una riga per elemento
    - altri tipi -> 1 riga con raw_json
    """
    # Caso 1: singolo oggetto JSON
    if isinstance(payload, dict):
        row = pd.json_normalize(payload, sep=".").to_dict(orient="records")[0]
        row["file_name"] = file_name
        return [row]

    # Caso 2: lista di oggetti JSON
    if isinstance(payload, list):
        if len(payload) == 0:
            return [{
                "file_name": file_name,
                "message": "Empty JSON list",
            }]

        # se tutti o quasi sono dict, esplodili in righe
        if all(isinstance(item, dict) for item in payload):
            rows = pd.json_normalize(payload, sep=".").to_dict(orient="records")
            for row in rows:
                row["file_name"] = file_name
            return rows

        # lista non tabellare
        return [{
            "file_name": file_name,
            "raw_json": json.dumps(payload, ensure_ascii=False),
        }]

    # Caso 3: tipo non gestibile tabellarmente
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

        # limita il numero di righe per file, così la preview non esplode
        rows.extend(file_rows[:max_rows_per_file])

    if not rows:
        return pd.DataFrame({
            "message": [f"Nessun file JSON trovato in: {path}"]
        })

    df = pd.DataFrame(rows)

    # ordine colonne più leggibile
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
        return pd.DataFrame({"message": [f"File non trovato: {preview_name}"]})

    if path.endswith(".csv"):
        return _csv_preview_df(path, max_rows=20)

    if os.path.isdir(path):
        return _directory_json_preview_df(path, max_files=20)

    return pd.DataFrame({"message": [f"Formato non supportato: {path}"]})


def update_preview(universe: str, preview_name: str) -> pd.DataFrame:
    return load_preview_file(universe, preview_name)

# -------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------
def run_data_pipeline(universe: str, progress=gr.Progress(track_tqdm=False)):
    if not universe:
        return (
            "❌ Seleziona un dataset",
            gr.update(choices=[], value=None),
            pd.DataFrame(),
            update_existing_button(universe),
        )

    try:
        progress(0.0, desc="Inizializzazione pipeline...")

        if DRY_RUN:
            _run_mock_pipeline(universe, progress)
            status_msg = f"✅ Pipeline DEMO completata per '{universe}' (nessun download reale eseguito)"
        else:
            _run_real_pipeline(universe, progress)
            status_msg = f"✅ Pipeline completata per '{universe}'"

        progress(0.97, desc="Caricamento preview...")
        options = get_preview_options(universe)
        default_preview = options[0] if options else None
        preview_df = load_preview_file(universe, default_preview) if default_preview else pd.DataFrame()

        progress(1.0, desc="Completato")
        return (
            status_msg,
            # gr.update(choices=options, value=default_preview),
            # preview_df,
            update_existing_button(universe),
        )

    except Exception as e:
        return (
            f"❌ Errore durante la pipeline: {str(e)}",
            # gr.update(choices=[], value=None),
            # pd.DataFrame(),
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
        (0.10, f"Creazione dataset mock per {universe}..."),
        (0.30, "Simulazione feature extraction..."),
        (0.45, "Simulazione compute relation..."),
        (0.60, "Simulazione incidence matrix..."),
        (0.75, "Simulazione market information..."),
        (0.90, "Simulazione CNNPred market features..."),
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