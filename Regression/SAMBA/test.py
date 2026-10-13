import os
from argparse import ArgumentParser

import torch

from .train import build_model, collect_results, load_frame, period_dataset, predict, ticker_checkpoint, ticker_frame
from utils.load_configuation import load_configuration
from utils.logger import _log, emit_result

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))


def run_test(task_type, test_model, universe, config_name, sl_value, pl_value,
             start_test_date, end_date, log_fn=None):
    _log(log_fn, "Preparing test...")
    args, model_save_path, metrics_path = load_configuration(
        task_type, test_model, universe, config_name, sl_value, pl_value,
        start_test_date, end_date, PROJECT_ROOT, log_fn
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df, _ = load_frame(args)
    results = {"preds": [], "labels": [], "pred_date": [], "last_date": [], "tickers": []}

    # One model per ticker: only the tickers trained (and saved) can be tested.
    for ticker in args.trained_tickers:
        checkpoint = ticker_checkpoint(model_save_path, ticker)
        df_ticker = ticker_frame(df, ticker, args)
        if df_ticker is None or not os.path.exists(checkpoint):
            continue

        test_dataset = period_dataset(df_ticker, args, args.start_test_date, args.end_date, 'test')
        if test_dataset.idx <= 0 or len(getattr(test_dataset, "windows", [])) == 0:
            _log(log_fn, f"No test data for ticker {ticker}. Skipping...")
            continue

        model = build_model(args, test_dataset.windows.shape[2], device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))

        y_pred, y_true = predict(model, test_dataset, args, device)
        results["preds"].extend(y_pred)
        results["labels"].extend(y_true)
        results["tickers"].extend([ticker] * len(y_pred))
        results["pred_date"].extend(test_dataset.dates)
        results["last_date"].extend(test_dataset.last_dates)

    metrics, results_pkl, metrics_json = collect_results(results, args, metrics_path)
    return {"metrics": metrics, "results_path": results_pkl,
            "metrics_path": metrics_json, "device": str(device)}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task_type", type=str, default='Regression')
    parser.add_argument("--model_name", type=str, default='SAMBA')
    parser.add_argument("--universe", type=str, default='sp500')
    parser.add_argument("--config_name", type=str, default='SAMBA')
    parser.add_argument("--sl_value", type=int, default=5)
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
