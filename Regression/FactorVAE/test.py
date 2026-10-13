import os
from argparse import ArgumentParser

import torch

from .train import build_model, loader, metrics_of, predict, prepare_dataset
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = prepare_dataset(args)
    all_dates = dataset.index.get_level_values('datetime').unique().sort_values()
    test_loader = loader(dataset, args, args.start_test_date, args.end_date)

    model = build_model(args).to(device)
    model.load_state_dict(torch.load(os.path.join(model_save_path, 'model.pth'), map_location=device))

    preds, labels, pred_dates, last_dates, tickers = predict(model, test_loader, all_dates, args.pred_len, device)
    metrics, results_pkl, metrics_json = save_results(
        args, metrics_of(preds, labels), preds, labels, pred_dates, last_dates, tickers,
        metrics_path, log_fn=log_fn
    )
    return {"metrics": metrics, "results_path": results_pkl,
            "metrics_path": metrics_json, "device": str(device)}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task_type", type=str, default='Regression')
    parser.add_argument("--model_name", type=str, default='FactorVAE')
    parser.add_argument("--universe", type=str, default='sp500')
    parser.add_argument("--config_name", type=str, default='FactorVAE')
    parser.add_argument("--sl_value", type=int, default=10)
    parser.add_argument("--pl_value", type=int, default=5)
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
