import argparse
import os
import re
import logging
import pandas as pd
from typing import Optional, List, Dict, Any
import subprocess
import sys
from pathlib import Path

from paths import (
    EVALUATION_RESULT_DIR,
    EVALUATION_RESULT_QUINTILE_DIR,
    DEFAULT_RESULTS_ROOT,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

THIS_DIR = Path(__file__).resolve().parent
EVALUATION_SCRIPT = THIS_DIR / "evaluation.py"
QUINTILE_SCRIPT = THIS_DIR / "quintile_analysis.py"


# ----------------------------
# functions for lunch evaluation.py and quintile_analysis.py
# ----------------------------

def run_evaluation_before_extract(args):
    cmd = [
        sys.executable,
        str(EVALUATION_SCRIPT),
        "--universe", args.universe,
        "--model", args.model,
        "--type", args.type,
        "--initial_year", str(args.initial_year),
        "--end_year", str(args.end_year),
        "--seed", str(args.seed),
        "--sl", str(args.sl),
        "--pl", str(args.pl),
        "--top_k", str(args.top_k),
        "--short_k", str(args.short_k),
        "--results_root", args.results_root,
    ]

    logging.info("Running evaluation.py before extraction...")
    logging.info("Command: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_quintile_before_extract(args):
    cmd = [
        sys.executable,
        str(QUINTILE_SCRIPT),
        "--universe", args.universe,
        "--model", args.model,
        "--type", args.type,
        "--initial_year", str(args.initial_year),
        "--end_year", str(args.end_year),
        "--seed", str(args.seed),
        "--sl", str(args.sl),
        "--pl", str(args.pl),
        "--results_root", args.results_root,
    ]

    logging.info("Running quintile analysis before extraction...")
    logging.info("Command: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


# ----------------------------
# Helper
# ----------------------------
def _parse_percent_or_none(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "":
        return None
    s = s.replace("%", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


def _safe_read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ----------------------------
# EOY returns aggregation
# ----------------------------

FILENAME_EOY_RE = re.compile(
    r"eoy_returns_sl(?P<sl>\d+)_pl(?P<pl>\d+)_seed(?P<seed>\d+)\.csv"
)


def extract_eoy_returns(
        base_dir: str = "evaluation_result",
        top_k: int = 10,
        short_k: int = 0,
        output_prefix: str = "aggregated_eoy_returns"
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    topk_shortx = f"top{top_k}_short{short_k}"

    for root, _, files in os.walk(base_dir):
        for fname in files:
            match = FILENAME_EOY_RE.match(fname)
            if not match:
                continue

            fpath = os.path.join(root, fname)
            parts = os.path.normpath(fpath).split(os.sep)

            try:
                idx = parts.index(os.path.basename(base_dir))

                universe = parts[idx + 1]
                top_short = parts[idx + 2]
                task_type = parts[idx + 3]
                model = parts[idx + 4]

                config = parts[idx + 5] if len(parts) > idx + 5 else None
                seed_folder = parts[idx + 6] if len(parts) > idx + 6 else None
                period = parts[idx + 7] if len(parts) > idx + 7 else None

                if top_short != topk_shortx:
                    continue

            except Exception:
                logging.warning(f"Skipping path not compliant: {fpath}")
                continue

            sl = int(match.group("sl"))
            pl = int(match.group("pl"))
            seed = int(match.group("seed"))

            try:
                df = _safe_read_csv(fpath)
            except Exception as e:
                logging.warning(f"Read error {fpath}: {e}")
                continue

            if df.shape[1] < 2:
                logging.warning(f"Unexpected CSV format: {fpath}")
                continue

            date_col = df.columns[0]
            value_col = df.columns[1]

            try:
                df["year"] = pd.to_datetime(df[date_col]).dt.year
            except Exception as e:
                logging.warning(f"Cannot parse year in {fpath}: {e}")
                continue

            for _, r in df.iterrows():
                rows.append({
                    "universe": universe,
                    "model": model,
                    "config": config,
                    "task_type": task_type,
                    "topk": top_k,
                    "shortk": short_k,
                    "sl": sl,
                    "pl": pl,
                    "seed": seed,
                    "period": period,
                    "year": int(r["year"]),
                    "value": pd.to_numeric(r[value_col], errors="coerce"),
                })

    df_all = pd.DataFrame(rows)

    if df_all.empty:
        logging.info("No EOY data found.")
        return df_all

    group_cols = [
        "universe",
        "model",
        "config",
        "task_type",
        "topk",
        "shortk",
        "sl",
        "pl",
        "period",
        "year",
    ]

    df_agg = (
        df_all
        .groupby(group_cols, as_index=False)
        .agg(value=("value", "mean"))
    )

    df_final = (
        df_agg
        .pivot_table(
            index=[
                "universe",
                "model",
                "config",
                "task_type",
                "topk",
                "shortk",
                "sl",
                "pl",
                "period",
            ],
            columns="year",
            values="value"
        )
        .reset_index()
    )

    df_final.columns.name = None

    out_csv = f"{output_prefix}_{top_k}_{short_k}.csv"
    df_final.to_csv(out_csv, index=False)

    logging.info(f"Created file: {out_csv} with {len(df_final)} rows")

    return df_final


# ----------------------------
# Metrics extraction
# ----------------------------
TARGET_METRICS = [
    "Cumulative Return",
    "CAGR﹪",
    "Sharpe",
    "Max Drawdown",
    "Volatility (ann.)",
    "Sortino/√2",
    "Avg. Drawdown"
]
METRICS_FILENAME_RE = re.compile(r"metrics_sl(\d+)_pl(\d+)_seed(\d+)\.csv")


def extract_metrics(root_dir: str = "evaluation_result",
                    mean_output: str = "summary_metrics_mean_over_seeds.csv") -> pd.DataFrame:
    rows = []

    for root, _, files in os.walk(root_dir):
        for file in files:
            match = METRICS_FILENAME_RE.match(file)
            if not match:
                continue

            sl, pl, seed = match.groups()
            file_path = os.path.join(root, file)

            parts = os.path.normpath(file_path).split(os.sep)
            try:
                idx = parts.index(os.path.basename(root_dir))

                univers = parts[idx + 1]
                top_short = parts[idx + 2]
                task_type = parts[idx + 3]
                model = parts[idx + 4]

                config = parts[idx + 5] if len(parts) > idx + 5 else None
                seed_folder = parts[idx + 6] if len(parts) > idx + 6 else None
                period = parts[idx + 7] if len(parts) > idx + 7 else None

                m = re.match(r"top(\d+)_short(\d+)", top_short)
                topk, shortk = m.groups() if m else (None, None)

            except Exception:
                logging.warning(f"Skipping path not compliant: {file_path}")
                continue

            try:
                df = _safe_read_csv(file_path)
                metrics_dict = {}
                for metric in TARGET_METRICS:
                    val = df.loc[df.iloc[:, 0] == metric, df.columns[1]]
                    raw = val.values[0] if len(val) > 0 else None
                    metrics_dict[metric] = _parse_percent_or_none(raw)
            except Exception as e:
                logging.warning(f"Read error {file_path}: {e}")
                continue

            row = {
                "universe": univers,
                "model": model,
                "config": config,
                "task_type": task_type,
                "topk": topk,
                "shortk": shortk,
                "seed": int(seed),
                "sl": int(sl),
                "pl": int(pl),
                "period": period,
                **metrics_dict
            }
            rows.append(row)

    final_df = pd.DataFrame(rows)
    if final_df.empty:
        logging.info("Nessun file metrics trovato.")
        return final_df

    int_cols = ["seed", "sl", "pl"]
    for c in ["topk", "shortk"]:
        if c in final_df.columns:
            try:
                final_df[c] = final_df[c].astype(float).astype('Int64')
            except Exception:
                pass
    final_df[int_cols] = final_df[int_cols].astype(int)

    GROUP_COLS = [
        "universe",
        "model",
        "config",
        "task_type",
        "topk",
        "shortk",
        "sl",
        "pl",
        "period",
    ]
    metric_cols = TARGET_METRICS
    final_df[metric_cols] = final_df[metric_cols].apply(pd.to_numeric, errors="coerce")
    grouped_df = (
        final_df
        .groupby(GROUP_COLS, as_index=False)[metric_cols]
        .mean()
    )
    grouped_df.to_csv(mean_output, index=False)
    logging.info(f"Created file {mean_output} with {len(grouped_df)} rows")
    return final_df


# ----------------------------
# Quintile metrics extraction
# ----------------------------
TARGET_METRICS_QUINT = ["CAGR﹪"]
QUINT_FILENAME_RE = re.compile(r"metrics_q(\d+)_sl(\d+)_pl(\d+)_seed(\d+)\.csv")


def extract_metrics_quintile(root_dir: str = "evaluation_result_quintile",
                             mean_output: str = "summary_metrics_quintile_mean_over_seeds.csv") -> pd.DataFrame:
    rows = []

    for root, _, files in os.walk(root_dir):
        for file in files:
            match = QUINT_FILENAME_RE.match(file)
            if not match:
                continue
            quintile, sl, pl, seed = match.groups()
            file_path = os.path.join(root, file)

            parts = os.path.normpath(file_path).split(os.sep)
            try:
                idx = parts.index(os.path.basename(root_dir))

                universe = parts[idx + 1]
                task_type = parts[idx + 2]
                model = parts[idx + 3]

                config = parts[idx + 4] if len(parts) > idx + 4 else None
                seed_folder = parts[idx + 5] if len(parts) > idx + 5 else None
                period = parts[idx + 6] if len(parts) > idx + 6 else None

            except Exception:
                logging.warning(f"Skipping path not compliant: {file_path}")
                continue

            try:
                df = _safe_read_csv(file_path)
                metrics_dict = {}
                for metric in TARGET_METRICS_QUINT:
                    val = df.loc[df.iloc[:, 0] == metric, df.columns[1]]
                    raw = val.values[0] if len(val) > 0 else None
                    metrics_dict[metric] = _parse_percent_or_none(raw)
            except Exception as e:
                logging.warning(f"Read error {file_path}: {e}")
                continue

            row = {
                "universe": universe,
                "model": model,
                "config": config,
                "task_type": task_type,
                "sl": int(sl),
                "pl": int(pl),
                "seed": int(seed),
                "period": period,
                "quintile": int(quintile),
                **metrics_dict
            }

            rows.append(row)

    final_df = pd.DataFrame(rows)
    if final_df.empty:
        logging.info("No quintile files found.")
        return final_df

    GROUP_COLS = [
        "universe",
        "model",
        "config",
        "task_type",
        "sl",
        "pl",
        "period",
    ]
    final_df[TARGET_METRICS_QUINT] = final_df[TARGET_METRICS_QUINT].apply(pd.to_numeric, errors="coerce")
    mean_df = final_df.groupby(GROUP_COLS + ["quintile"], as_index=False)[TARGET_METRICS_QUINT].mean()

    pivot_df = mean_df.pivot_table(
        index=GROUP_COLS,
        columns="quintile",
        values=TARGET_METRICS_QUINT[0]
    ).reset_index()

    pivot_df.columns.name = None
    pivot_df = pivot_df.rename(columns=lambda x: f"q{x}" if str(x).isdigit() else x)
    pivot_df.to_csv(mean_output, index=False)
    logging.info(f"Created file {mean_output} with {len(pivot_df)} rows")
    return final_df


# ----------------------------
# CLI
# ----------------------------

def add_eval_args(parser):
    parser.add_argument("--universe", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["Regression", "Classification", "Ranking"],
    )
    parser.add_argument("--initial_year", type=int, required=True)
    parser.add_argument("--end_year", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sl", type=int, default=5)
    parser.add_argument("--pl", type=int, default=1)
    parser.add_argument(
        "--results_root",
        type=str,
        default=str(DEFAULT_RESULTS_ROOT),
    )
    parser.add_argument(
        "--skip_run",
        action="store_true",
        help="Skip evaluation/quintile run and only extract existing files.",
    )


def main():
    parser = argparse.ArgumentParser(prog="metrics_extraction")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_eoy = sub.add_parser("eoy", help="Run evaluation and aggregate EOY returns")
    add_eval_args(p_eoy)
    p_eoy.add_argument("--base_dir", default=str(EVALUATION_RESULT_DIR))
    p_eoy.add_argument("--top_k", type=int, required=True)
    p_eoy.add_argument("--short_k", type=int, required=True)
    p_eoy.add_argument("--output_prefix", default="aggregated_eoy_returns")

    p_metrics = sub.add_parser("metrics", help="Run evaluation and extract standard metrics")
    add_eval_args(p_metrics)
    p_metrics.add_argument("--root_dir", default="evaluation_result")
    p_metrics.add_argument("--top_k", type=int, required=True)
    p_metrics.add_argument("--short_k", type=int, required=True)
    p_metrics.add_argument("--mean_output", default=f"{str(EVALUATION_RESULT_DIR)}_summary_metrics_mean_over_seeds.csv")

    p_quint = sub.add_parser("quintile", help="Run quintile analysis and extract quintile metrics")
    add_eval_args(p_quint)
    p_quint.add_argument("--root_dir", default="evaluation_result_quintile")
    p_quint.add_argument("--mean_output",
                         default=f"{str(EVALUATION_RESULT_QUINTILE_DIR)}_summary_metrics_quintile_mean_over_seeds.csv")

    args = parser.parse_args()

    if args.cmd == "eoy":
        if not args.skip_run:
            run_evaluation_before_extract(args)

        extract_eoy_returns(
            base_dir=args.base_dir,
            top_k=args.top_k,
            short_k=args.short_k,
            output_prefix=args.output_prefix,
        )

    elif args.cmd == "metrics":
        if not args.skip_run:
            run_evaluation_before_extract(args)

        extract_metrics(
            root_dir=args.root_dir,
            mean_output=args.mean_output,
        )

    elif args.cmd == "quintile":
        if not args.skip_run:
            run_quintile_before_extract(args)

        extract_metrics_quintile(
            root_dir=args.root_dir,
            mean_output=args.mean_output,
        )


if __name__ == "__main__":
    main()
