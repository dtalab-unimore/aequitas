import os
from argparse import ArgumentParser

import torch

from .train import (
    DEVICE,
    build_dataset,
    build_experiment,
    build_model,
    evaluate,
    prepare_market_data,
    result_dates,
    set_seed,
)
from utils.load_configuation import load_configuration
from utils.logger import _log, emit_result
from utils.save_results import save_results

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))


def run_test(task_type, test_model, universe, config_name, sl_value, pl_value,
             start_test_date, end_date, log_fn=None):
    _log(log_fn, "Preparing test...")
    args, model_save_path, metrics_path = load_configuration(
        task_type, test_model, universe, config_name, sl_value, pl_value,
        start_test_date, end_date, PROJECT_ROOT, log_fn
    )
    set_seed(args.seed)
    ticker_dict, tickers = prepare_market_data(args)
    experiment = build_experiment(args)
    test_dataset = build_dataset(experiment, "test", experiment.test_range, ticker_dict)
    model = build_model(experiment, len(tickers), experiment.window)
    checkpoint_path = os.path.join(model_save_path, "model.pth")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics, predictions, labels = evaluate(model, test_dataset)
    last_dates, pred_dates = result_dates(test_dataset, args.pred_len)
    metrics, results_pkl, metrics_json = save_results(
        args, test_metrics, predictions, labels, pred_dates, last_dates,
        [tickers] * len(predictions), metrics_path, log_fn=log_fn
    )
    return {
        "metrics": metrics,
        "results_path": results_pkl,
        "metrics_path": metrics_json,
        "device": str(DEVICE),
    }


if __name__ == "__main__":
    parser = ArgumentParser(description="Test a trained DGDNN model")
    parser.add_argument("--task_type", default="Classification")
    parser.add_argument("--model_name", default="DGDNN")
    parser.add_argument("--universe", default="sp500")
    parser.add_argument("--config_name", default="DGDNN")
    parser.add_argument("--sl_value", type=int, default=5)
    parser.add_argument("--pl_value", type=int, default=1)
    parser.add_argument("--start_test_date", default="2020-01-01")
    parser.add_argument("--end_date", default="2020-12-31")
    cli_args = parser.parse_args()
    output = run_test(
        cli_args.task_type, cli_args.model_name, cli_args.universe, cli_args.config_name,
        cli_args.sl_value, cli_args.pl_value, cli_args.start_test_date, cli_args.end_date
    )
    emit_result(output)
