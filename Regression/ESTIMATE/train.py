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
from utils.training import save_valid_metrics, ticker_selection_date, training_result

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

    Training selects the tickers (constituents right after validation, with
    enough data in validation); the test reuses the trained list as is, in the
    same order, because the model is built for that exact set of stocks.
    Returns the list of tickers written.
    """
    preprocess_dir = os.path.join(work_dir, "data_estimate", "preprocess")
    os.makedirs(preprocess_dir, exist_ok=True)

    df = pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}.csv')
    df = df[['open', 'close', 'high', 'low', 'volume', 'instrument', 'date', 'adj_close']]

    if tickers is None:
        constituents = pd.read_csv(f'{args.data_path}/constituents/eodhd/{args.universe}.csv')
        members = filter_constituents_by_date(constituents, ticker_selection_date(args))['EODHD'].tolist()
        df = df[df['instrument'].isin(members)]
        df, _ = select_valid_ticker(df, args.start_date, args.start_valid_date, args.end_valid_date, args.seq_len)
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
    """
    Run execute.py in work_dir, to train (train=True) or to test.
    Returns the file it writes: the validation metrics when training, the
    test results otherwise.
    """
    if train:
        # Training has no test period, but ESTIMATE's data loader always builds
        # a test split: give it the validation period. It is never evaluated
        # (no --test) and its cache is removed by clear_data_cache.
        start_test, end_test = args.start_valid_date, args.end_valid_date
    else:
        start_test, end_test = args.start_test_date, args.end_date

    cmd = [
        sys.executable, "-u", os.path.join(ESTIMATE_DIR, "execute.py"),
        "--train" if train else "--test",
        "--run_name", args.model_name,
        "--model", "estimate",
        "--start_train", args.start_date,
        "--end_train", args.end_train_date,
        "--start_valid", args.start_valid_date,
        "--end_valid", args.end_valid_date,
        "--start_test", start_test,
        "--end_test", end_test,
        "--n_step_ahead", str(args.pred_len),
        "--history_window", str(args.seq_len),
        "--pretrained_log", checkpoint_dir,
        "--config_path", BACKTEST_CONFIG,
        "--save_best",
    ]
    for name in FORWARDED_ARGS:
        cmd += [f"--{name}", str(getattr(args, name))]

    # Only this folder on the path, so that ESTIMATE's own `utils` package is found.
    env = {**os.environ, "PYTHONPATH": ESTIMATE_DIR}
    subprocess.run(cmd, cwd=work_dir, env=env, check=True)

    # The supervisor names the folder after the start year of its test window
    year = start_test.split('-')[0]
    name = (f"val_metrics_sl{args.seq_len}_pl{args.pred_len}.json" if train
            else f"results_sl{args.seq_len}_pl{args.pred_len}.pkl")
    return os.path.join(work_dir, "results", args.universe, "estimate", str(args.seed), f"y{year}", name)


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
    model_save_path, _ = create_saving_path(args, "Regression", "ESTIMATE", PROJECT_ROOT)
    work_dir = os.path.join(model_save_path, "work")
    shutil.rmtree(work_dir, ignore_errors=True)

    args.trained_tickers = preprocess(args, work_dir)
    print(f"{len(args.trained_tickers)} tickers with enough data", flush=True)
    with open(os.path.join(model_save_path, "config_params.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    val_file = run_estimate(args, work_dir, model_save_path, train=True)

    # Written when a best checkpoint is saved (validation accuracy improved)
    valid_metrics, valid_path = None, None
    if os.path.exists(val_file):
        with open(val_file) as f:
            valid_metrics = json.load(f)
        valid_path = save_valid_metrics(valid_metrics, model_save_path, args.seq_len, args.pred_len)

    shutil.rmtree(work_dir, ignore_errors=True)
    clear_data_cache(model_save_path)
    return training_result(os.path.join(model_save_path, "model-best.pth"), valid_metrics, valid_path)


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
