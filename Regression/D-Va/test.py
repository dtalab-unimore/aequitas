import os
import shutil
from argparse import ArgumentParser

import pandas as pd
import torch

from .exp.exp_model import Exp_Model
from .preprocessing import read_prices, reset_dir, ticker_csv_name, ticker_frame
from .train import as_2d, get_metrics, ticker_checkpoint
from utils.load_configuation import load_configuration
from utils.logger import _log, emit_result
from utils.save_results import save_results

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))


def write_test_csv(df, args, root_path, ticker):
    """
    Rows for the test of one ticker: the training period (the scaler is fitted
    on it), then the seq_len rows just before the test start, then the test
    period. Dataset_Custom slices its periods by counting rows, so the window
    rows play the role of the validation period and the test windows start
    right at start_test_date, whatever gap there is after training.

    Returns False when the ticker has too little test data for one batch.
    """
    df = df.dropna(axis=0)
    train = df[(df['date'] >= args.start_date) & (df['date'] <= args.end_train_date)]
    window = df[df['date'] < args.start_test_date].tail(args.sequence_length)
    test = df[(df['date'] >= args.start_test_date) & (df['date'] <= args.end_date)]

    if len(window) < args.sequence_length or len(test) - args.prediction_length + 1 < args.batch_size:
        return False

    # If the test starts right after training, the window must not be counted twice.
    train = train[train['date'] < window['date'].min()]

    args.start_valid_date, args.end_valid_date = window['date'].min(), window['date'].max()
    pd.concat([train, window, test]).to_csv(os.path.join(root_path, ticker_csv_name(ticker)), index=False)
    return True


def run_test(task_type, test_model, universe, config_name, sl_value, pl_value,
             start_test_date, end_date, log_fn=None):
    _log(log_fn, "Preparing test...")
    args, model_save_path, metrics_path = load_configuration(
        task_type, test_model, universe, config_name, sl_value, pl_value,
        start_test_date, end_date, PROJECT_ROOT, log_fn
    )
    args.sequence_length, args.prediction_length = args.seq_len, args.pred_len
    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)
    device = "cuda" if args.use_gpu else "cpu"

    data = read_prices(args.data_path, args.universe)
    args.root_path = os.path.join(model_save_path, "data_test")
    reset_dir(args.root_path)

    results = {"preds": [], "labels": [], "pred_date": [], "last_date": [], "tickers": []}

    # One model per ticker: only the tickers trained (and saved) can be tested.
    for ticker in args.trained_tickers:
        checkpoint = ticker_checkpoint(model_save_path, ticker)
        if not os.path.exists(checkpoint) or not write_test_csv(ticker_frame(data, ticker), args, args.root_path, ticker):
            _log(log_fn, f"No test data for ticker {ticker}. Skipping...")
            continue

        args.data_path = ticker_csv_name(ticker)
        exp = Exp_Model(args)
        exp.denoise_net.load_state_dict(torch.load(checkpoint, map_location=exp.device))
        pred, label, pred_date, last_date = exp.test(None, args.sequence_length, args.prediction_length)

        results["preds"].extend(pred)
        results["labels"].extend(label)
        results["pred_date"].extend(pred_date)
        results["last_date"].extend(last_date)
        results["tickers"].extend([ticker] * len(pred))

    shutil.rmtree(args.root_path, ignore_errors=True)

    if not results["preds"]:
        raise RuntimeError("No ticker had enough data in the test period.")

    metrics, results_pkl, metrics_json = save_results(
        args, get_metrics(results["preds"], results["labels"]),
        as_2d(results["preds"]), as_2d(results["labels"]),
        results["pred_date"], results["last_date"], results["tickers"], metrics_path, log_fn=log_fn,
    )
    return {"metrics": metrics, "results_path": results_pkl,
            "metrics_path": metrics_json, "device": device}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task_type", type=str, default='Regression')
    parser.add_argument("--model_name", type=str, default='D-Va')
    parser.add_argument("--universe", type=str, default='sp500')
    parser.add_argument("--config_name", type=str, default='DVa')
    parser.add_argument("--sl_value", type=int, default=2)
    parser.add_argument("--pl_value", type=int, default=1)
    parser.add_argument("--start_test_date", type=str, default='2023-01-01')
    parser.add_argument("--end_date", type=str, default='2023-12-31')
    args = parser.parse_args()

    out = run_test(
        task_type=args.task_type,
        test_model=args.model_name,
        universe=args.universe,
        config_name=args.config_name,
        sl_value=args.sl_value,
        pl_value=args.pl_value,
        start_test_date=args.start_test_date,
        end_date=args.end_date,
    )
    emit_result(out)
