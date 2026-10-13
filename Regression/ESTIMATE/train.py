"""
ESTIMATE adapter for the Aequitas train/test contract.

ESTIMATE's framework (execute.py, model/, utils/) expects to run from inside
this folder: it has its own `utils` package and reads data_estimate/preprocess/
relative to the working directory. So this module does the preprocessing
into a per-run working folder, runs execute.py as a child process there, and
then moves the checkpoint and the results to the standard locations.
"""
import json
import os
import pickle
import shutil
import subprocess
import sys
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from utils.filter_constituents import filter_constituents_by_date
from utils.load_configuation import create_saving_path
from utils.logger import emit_result
from utils.save_results import save_results

ESTIMATE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ESTIMATE_DIR, "..", ".."))
BACKTEST_CONFIG = os.path.join(ESTIMATE_DIR, "backtest", "config", "normal.yaml")

# Arguments forwarded unchanged to execute.py
FORWARDED_ARGS = [
    "universe", "target_col", "outlier_threshold",
    "close_sma_medium", "close_sma_slow", "rsi_medium", "rsi_slow",
    "macd_medium", "macd_slow", "mfi_medium", "mfi_slow",
    "confidence_threshold", "earlystop", "batch_size", "epochs", "hidden_dim", "rnn_units",
    "learning_rate", "lr_decay", "eval_iter", "dropout", "verify_threshold", "seed",
]


def select_valid_ticker(df, start_date, start_test, end_date, seq_len=20):
    df_train = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    df_test = df[(df['date'] >= start_test) & (df['date'] <= end_date)]
    df_test = df_test.groupby('instrument').filter(lambda x: len(x) > seq_len)

    tickers_train = set(df_train['instrument'].unique())
    tickers_test = set(df_test['instrument'].unique())
    valid_tickers = tickers_train.intersection(tickers_test)

    df = df[df['instrument'].isin(valid_tickers)]
    return df, valid_tickers


def preprocess(args, work_dir, tickers=None):
    """
    Write baseline_data.npy and ticker_info.csv into <work_dir>/data_estimate/preprocess.

    Training selects the tickers (constituents at the test start, with enough
    history); the test reuses the trained list as is, in the same order,
    because the model is built for that exact set of stocks.
    Returns the list of tickers written.
    """
    preprocess_dir = os.path.join(work_dir, "data_estimate", "preprocess")
    os.makedirs(preprocess_dir, exist_ok=True)

    df = pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}.csv')
    df = df[['open', 'close', 'high', 'low', 'volume', 'instrument', 'date', 'adj_close']]

    if tickers is None:
        constituents = pd.read_csv(f'{args.data_path}/constituents/eodhd/{args.universe}.csv')
        members = filter_constituents_by_date(constituents, args.start_test_date)['EODHD'].tolist()
        df = df[df['instrument'].isin(members)]
        df, _ = select_valid_ticker(df, args.start_date, args.start_test_date, args.end_date, args.seq_len)
        candidates = sorted(df['instrument'].unique())
    else:
        df = df[df['instrument'].isin(tickers)]
        candidates = list(tickers)

    dates = df['date'].drop_duplicates().sort_values()
    ticker_dict = {}

    for ticker in candidates:
        sub_df = df[df['instrument'] == ticker].sort_values('date')
        if tickers is None:
            train_mask = (sub_df['date'] >= args.start_date) & (sub_df['date'] <= args.end_train_date)
            if train_mask.sum() < args.pred_len + args.seq_len:
                continue
        elif sub_df.empty:
            raise ValueError(f"No data for the trained ticker {ticker}: ESTIMATE needs every trained ticker.")

        sub_df = pd.merge(pd.DataFrame({'date': dates}), sub_df, on='date', how='left').ffill().infer_objects()
        ticker_dict[ticker] = sub_df.set_index('date').drop(columns=['instrument'])

    np.save(os.path.join(preprocess_dir, "baseline_data.npy"), ticker_dict)

    info = pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}_info.csv')
    info = info[info['instrument'].isin(list(ticker_dict))]
    info = info[['instrument', 'Name', 'GicSector']]
    info = info.rename({'instrument': 'Symbol', 'GicSector': 'Sector'}, axis=1)
    info.to_csv(os.path.join(preprocess_dir, "ticker_info.csv"), index=False)

    return list(ticker_dict)


def run_estimate(args, work_dir, checkpoint_dir, train):
    """Run execute.py (train+test, or test only) in work_dir; returns the results file it writes."""
    cmd = [
        sys.executable, "-u", os.path.join(ESTIMATE_DIR, "execute.py"),
        "--test",
        "--run_name", args.model_name,
        "--model", "estimate",
        "--start_train", args.start_date,
        "--end_train", args.end_train_date,
        "--start_valid", args.start_valid_date,
        "--end_valid", args.end_valid_date,
        "--start_test", args.start_test_date,
        "--end_test", args.end_date,
        "--n_step_ahead", str(args.pred_len),
        "--history_window", str(args.seq_len),
        "--pretrained_log", checkpoint_dir,
        "--config_path", BACKTEST_CONFIG,
        "--save_best",
    ]
    if train:
        cmd.append("--train")
    for name in FORWARDED_ARGS:
        cmd += [f"--{name}", str(getattr(args, name))]

    # Only this folder on the path, so that ESTIMATE's own `utils` package is found.
    env = {**os.environ, "PYTHONPATH": ESTIMATE_DIR}
    subprocess.run(cmd, cwd=work_dir, env=env, check=True)

    year = args.start_test_date.split('-')[0]
    return os.path.join(work_dir, "results", args.universe, "estimate", str(args.seed), f"y{year}",
                        f"results_sl{args.seq_len}_pl{args.pred_len}.pkl")


def clear_data_cache(checkpoint_dir):
    """
    Remove ESTIMATE's data caches from the checkpoint folder.

    The framework caches the full train/valid/test arrays in
    train_<start>_<end>.npz (and backtest_*.npz), keyed only on the training
    dates: a test on another period would silently reuse the old test set.
    """
    for name in os.listdir(checkpoint_dir):
        if name.endswith(".npz") and name.startswith(("train_", "backtest_")):
            os.remove(os.path.join(checkpoint_dir, name))


def save_estimate_results(args, results_file, metrics_path, log_fn=None):
    with open(results_file, "rb") as f:
        results = pickle.load(f)
    return save_results(
        args, results["metrics"], results["preds"], results["labels"],
        results["pred_date"], results["last_date"], results["tickers"], metrics_path, log_fn=log_fn,
    )


def run_train(args):
    model_save_path, metrics_path = create_saving_path(args, "Regression", "ESTIMATE", PROJECT_ROOT)
    work_dir = os.path.join(model_save_path, "work")
    shutil.rmtree(work_dir, ignore_errors=True)

    args.trained_tickers = preprocess(args, work_dir)
    print(f"{len(args.trained_tickers)} tickers with enough data", flush=True)
    with open(os.path.join(model_save_path, "config_params.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    results_file = run_estimate(args, work_dir, model_save_path, train=True)

    val_file = os.path.join(os.path.dirname(results_file), f"val_metrics_sl{args.seq_len}_pl{args.pred_len}.json")
    if os.path.exists(val_file):
        shutil.copy(val_file, metrics_path)

    metrics, results_pkl, metrics_json = save_estimate_results(args, results_file, metrics_path)
    shutil.rmtree(work_dir, ignore_errors=True)
    clear_data_cache(model_save_path)
    return {"model_path": os.path.join(model_save_path, "model-best.pth"), "metrics": metrics,
            "results_path": results_pkl, "metrics_path": metrics_json}


def build_parser():
    args = ArgumentParser(description="Train ESTIMATE")

    # Standard Aequitas arguments
    args.add_argument('--data_path', type=str, default='./Evaluation/data')
    args.add_argument('--universe', type=str, default='sp500')
    args.add_argument('--model_name', type=str, default='ESTIMATE')
    args.add_argument('--seq_len', type=int, default=20, help='history window')
    args.add_argument('--pred_len', type=int, default=5, help='steps ahead')
    args.add_argument('--start_date', default='2018-01-01')
    args.add_argument('--end_train_date', default='2021-12-31')
    args.add_argument('--start_valid_date', default='2022-01-01')
    args.add_argument('--end_valid_date', default='2022-12-31')
    args.add_argument('--start_test_date', default='2023-01-01')
    args.add_argument('--end_date', default='2023-12-31')

    # ESTIMATE data / indicators
    args.add_argument('--target_col', default='trend_return')
    args.add_argument('--outlier_threshold', type=float, default=1000)
    args.add_argument('--close_sma_medium', type=int, default=10)
    args.add_argument('--close_sma_slow', type=int, default=20)
    args.add_argument('--rsi_medium', type=int, default=10)
    args.add_argument('--rsi_slow', type=int, default=20)
    args.add_argument('--macd_medium', type=int, default=10)
    args.add_argument('--macd_slow', type=int, default=20)
    args.add_argument('--mfi_medium', type=int, default=10)
    args.add_argument('--mfi_slow', type=int, default=20)

    # ESTIMATE model
    args.add_argument('--confidence_threshold', type=float, default=0.90)
    args.add_argument('--earlystop', type=int, default=16)
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--epochs', type=int, default=500)
    args.add_argument('--hidden_dim', type=int, default=32)
    args.add_argument('--rnn_units', type=int, default=16)
    args.add_argument('--learning_rate', type=float, default=0.0001)
    args.add_argument('--lr_decay', type=float, default=0.005)
    args.add_argument('--eval_iter', type=int, default=10)
    args.add_argument('--dropout', type=float, default=0.4)
    args.add_argument('--verify_threshold', type=float, default=0.08)
    args.add_argument('--seed', type=int, default=42)
    return args


if __name__ == '__main__':
    output = run_train(build_parser().parse_args())
    emit_result(output)
