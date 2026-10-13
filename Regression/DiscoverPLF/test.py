import os
from argparse import ArgumentParser

import numpy as np
import torch

from .train import DEVICE, create_loaders, create_model, get_metrics, test_epoch
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
    _, _, test_loader, concept_matrix, stock_index, last_dates, pred_dates, _ = create_loaders(args)
    inverse_stock_index = {value: key for key, value in stock_index.items()}
    concept_matrix = torch.as_tensor(concept_matrix, dtype=torch.float32, device=DEVICE)
    model = create_model(args)
    checkpoint_path = os.path.join(model_save_path, "model.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=False))
    predictions, labels, stock_indexes, _, _, mask = test_epoch(
        model, test_loader, args, concept_matrix, prefix="Test"
    )
    tickers = [[inverse_stock_index.get(int(value), str(value)) for value in row] for row in stock_indexes]
    filtered_pred_dates = [date for index, date in enumerate(pred_dates) if mask[index]]
    filtered_last_dates = [date for index, date in enumerate(last_dates) if mask[index]]
    prediction_rows = [np.expand_dims(value, axis=1) for value in predictions]
    label_rows = [np.expand_dims(value, axis=1) for value in labels]
    metrics, results_pkl, metrics_json = save_results(
        args, get_metrics(predictions, labels), prediction_rows, label_rows,
        filtered_pred_dates, filtered_last_dates, tickers, metrics_path, log_fn=log_fn
    )
    return {"metrics": metrics, "results_path": results_pkl,
            "metrics_path": metrics_json, "device": str(DEVICE)}


if __name__ == "__main__":
    parser = ArgumentParser(description="Test a trained DiscoverPLF model")
    parser.add_argument("--task_type", default="Regression")
    parser.add_argument("--model_name", default="DiscoverPLF")
    parser.add_argument("--universe", default="sp500")
    parser.add_argument("--config_name", default="DiscoverPLF")
    parser.add_argument("--sl_value", type=int, default=1)
    parser.add_argument("--pl_value", type=int, default=5)
    parser.add_argument("--start_test_date", default="2023-01-01")
    parser.add_argument("--end_date", default="2023-12-31")
    cli_args = parser.parse_args()
    output = run_test(
        cli_args.task_type, cli_args.model_name, cli_args.universe, cli_args.config_name,
        cli_args.sl_value, cli_args.pl_value, cli_args.start_test_date, cli_args.end_date
    )
    emit_result(output)
