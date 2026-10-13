import os
from argparse import ArgumentParser

import numpy as np
import torch

from Ranking.SVAT.experiments.exp_svat import Exp_SVAT
from Ranking.SVAT.utils.config import build_param_dict
from Ranking.SVAT.utils.load_data import save_dates
from Ranking.SVAT.utils.metric import get_metrics
from Ranking.SVAT.utils.optimize import set_seed
from utils.load_configuation import load_configuration
from utils.logger import _log, emit_result
from utils.save_results import save_results

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))


def run_test(
    task_type,
    test_model,
    universe,
    config_name,
    sl_value,
    pl_value,
    start_test_date,
    end_date,
    log_fn=None,
):
    _log(log_fn, "Preparing test...")

    args, model_save_path, metrics_path = load_configuration(
        task_type,
        test_model,
        universe,
        config_name,
        sl_value,
        pl_value,
        start_test_date,
        end_date,
        PROJECT_ROOT,
        log_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpus = 0 if device.type == "cuda" else None

    set_seed(args.seed)
    exp = Exp_SVAT(build_param_dict(args), gpus=gpus)
    test_pred, test_gt, _ = exp.test(model_save_path)

    test_pred = test_pred.swapaxes(0, 1)
    test_gt = test_gt.swapaxes(0, 1)
    test_metrics = get_metrics(test_gt, test_pred)
    predictions = [np.expand_dims(row, axis=1) for row in test_pred]
    labels = [np.expand_dims(row, axis=1) for row in test_gt]
    dates_gt, last_seq_date, tickers = save_dates(
        exp.dates, exp.tickers, exp.test_index, args.pred_len, args.seq_len
    )

    metrics, results_pkl, metrics_json = save_results(
        args,
        test_metrics,
        predictions,
        labels,
        dates_gt,
        last_seq_date,
        tickers,
        metrics_path,
        log_fn=log_fn,
    )

    torch.cuda.empty_cache()
    return {
        "metrics": metrics,
        "results_path": results_pkl,
        "metrics_path": metrics_json,
        "device": str(device),
    }


if __name__ == "__main__":
    parser = ArgumentParser(description="Test a trained SVAT ranking model")
    parser.add_argument("--task_type", type=str, default="Ranking")
    parser.add_argument("--model_name", type=str, default="SVAT")
    parser.add_argument("--universe", type=str, default="sx5e")
    parser.add_argument("--config_name", type=str, default="SVAT")
    parser.add_argument("--sl_value", type=int, default=10)
    parser.add_argument("--pl_value", type=int, default=5)
    parser.add_argument("--start_test_date", type=str, default="2023-01-01")
    parser.add_argument("--end_date", type=str, default="2023-12-31")
    cli_args = parser.parse_args()

    output = run_test(
        task_type=cli_args.task_type,
        test_model=cli_args.model_name,
        universe=cli_args.universe,
        config_name=cli_args.config_name,
        sl_value=cli_args.sl_value,
        pl_value=cli_args.pl_value,
        start_test_date=cli_args.start_test_date,
        end_date=cli_args.end_date,
    )
    emit_result(output)
