import os
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from .load_data import construct_data_warehouse_3d
from .train import get_metrics, predict_all_stocks
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
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    dataset, tickers, dates = construct_data_warehouse_3d(args)
    trained_tickers = list(getattr(args, "trained_tickers", []))
    if trained_tickers and tickers != trained_tickers:
        raise ValueError("The test universe does not match the stock order saved during CNNPred3D training.")

    n_stocks = dataset[0].shape[1]
    n_features = dataset[0].shape[-1]
    predictions, probabilities = predict_all_stocks(
        dataset, args, model_save_path, n_stocks, n_features, 4
    )
    probability_rows = [row.reshape(-1, 1) for row in probabilities]
    label_rows = [row.reshape(-1, 1) for row in np.asarray(dataset[5])]
    metrics, results_pkl, metrics_json = save_results(
        args, get_metrics(predictions, dataset[5]), probability_rows, label_rows,
        dates["pred_date"], dates["last_date"], [tickers] * len(probability_rows),
        metrics_path, log_fn=log_fn
    )
    return {
        "metrics": metrics,
        "results_path": results_pkl,
        "metrics_path": metrics_json,
        "device": str(tf.config.list_physical_devices("GPU") or "cpu"),
    }


if __name__ == "__main__":
    parser = ArgumentParser(description="Test a trained CNNPred3D model")
    parser.add_argument("--task_type", default="Classification")
    parser.add_argument("--model_name", default="CNNPred3D")
    parser.add_argument("--universe", default="sp500")
    parser.add_argument("--config_name", default="CNNPred3D")
    parser.add_argument("--sl_value", type=int, default=60)
    parser.add_argument("--pl_value", type=int, default=5)
    parser.add_argument("--start_test_date", default="2023-01-01")
    parser.add_argument("--end_date", default="2023-12-31")
    cli_args = parser.parse_args()
    output = run_test(
        cli_args.task_type, cli_args.model_name, cli_args.universe, cli_args.config_name,
        cli_args.sl_value, cli_args.pl_value, cli_args.start_test_date, cli_args.end_date
    )
    emit_result(output)
