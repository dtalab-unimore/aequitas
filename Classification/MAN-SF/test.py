import json
import os
from argparse import ArgumentParser

import torch

from .train import create_model, eval_model, flatten_dates, prepare_data
from utils.load_configuation import _log, load_configuration
from utils.save_results import save_results

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))


def run_test(task_type, test_model, universe, config_name, sl_value, pl_value,
             start_test_date, end_date, log_fn=None):
    _log(log_fn, "Preparazione del test MAN-SF...")
    args, model_save_path, metrics_path = load_configuration(
        task_type, test_model, universe, config_name, sl_value, pl_value,
        start_test_date, end_date, PROJECT_ROOT, log_fn
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prices, news, last_dates, pred_dates, tickers, adjacency = prepare_data(args)
    trained_tickers = list(getattr(args, "trained_tickers", []))
    if trained_tickers and tickers != trained_tickers:
        raise ValueError("The test universe does not match the stock order saved during MAN-SF training.")
    adjacency = adjacency.to(device)
    model = create_model(args, len(tickers), device)
    checkpoint_path = os.path.join(model_save_path, "model.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    metrics, predictions, labels = eval_model(model, prices[2], news[2], adjacency, device)
    metrics, results_pkl, metrics_json = save_results(
        args, metrics, predictions, labels, flatten_dates(pred_dates), flatten_dates(last_dates),
        [tickers] * len(predictions), metrics_path, log_fn=log_fn
    )
    return {
        "metrics": metrics,
        "results_path": results_pkl,
        "metrics_path": metrics_json,
        "device": str(device),
    }


if __name__ == "__main__":
    parser = ArgumentParser(description="Test a trained MAN-SF model")
    parser.add_argument("--task_type", default="Classification")
    parser.add_argument("--model_name", default="MAN-SF")
    parser.add_argument("--universe", default="sp500")
    parser.add_argument("--config_name", default="MAN-SF")
    parser.add_argument("--sl_value", type=int, default=10)
    parser.add_argument("--pl_value", type=int, default=5)
    parser.add_argument("--start_test_date", default="2023-01-01")
    parser.add_argument("--end_date", default="2023-12-31")
    cli_args = parser.parse_args()
    output = run_test(
        cli_args.task_type, cli_args.model_name, cli_args.universe, cli_args.config_name,
        cli_args.sl_value, cli_args.pl_value, cli_args.start_test_date, cli_args.end_date
    )
    print(json.dumps(output, indent=4))
