import json
import os
from argparse import ArgumentParser
from datetime import date

import numpy as np
import torch

from Ranking.SVAT.experiments.exp_svat import Exp_SVAT
from Ranking.SVAT.utils.config import build_param_dict
from Ranking.SVAT.utils.load_data import save_dates
from Ranking.SVAT.utils.metric import get_metrics
from Ranking.SVAT.utils.optimize import set_seed
from utils.load_configuation import create_saving_path
from utils.logger import emit_result
from utils.save_results import save_results

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))


def validate_training_dates(args):
    start_date = date.fromisoformat(args.start_date)
    end_train_date = date.fromisoformat(args.end_train_date)
    start_valid_date = date.fromisoformat(args.start_valid_date)
    end_valid_date = date.fromisoformat(args.end_valid_date)

    if not start_date <= end_train_date < start_valid_date <= end_valid_date:
        raise ValueError(
            "Gli intervalli devono rispettare: start train <= end train "
            "< start validation <= end validation."
        )


def run_train(args):
    validate_training_dates(args)

    model_save_path, metrics_path = create_saving_path(
        args, "Ranking", "SVAT", PROJECT_ROOT
    )

    with open(os.path.join(model_save_path, "config_params.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpus = 0 if device.type == "cuda" else None

    set_seed(args.seed)
    exp = Exp_SVAT(build_param_dict(args), gpus=gpus)
    test_pred, test_gt, _, valid_pred, valid_gt, _ = exp.train(model_save_path)

    valid_pred = valid_pred.swapaxes(0, 1)
    valid_gt = valid_gt.swapaxes(0, 1)
    valid_metrics = {k: float(v) for k, v in get_metrics(valid_gt, valid_pred).items()}

    with open(
        os.path.join(metrics_path, f"val_metrics_sl{args.seq_len}_pl{args.pred_len}.json"),
        "w",
    ) as f:
        json.dump(valid_metrics, f, indent=4)

    if test_pred is not None and test_pred.size > 0:
        test_pred = test_pred.swapaxes(0, 1)
        test_gt = test_gt.swapaxes(0, 1)
        test_metrics = get_metrics(test_gt, test_pred)
        predictions = [np.expand_dims(row, axis=1) for row in test_pred]
        labels = [np.expand_dims(row, axis=1) for row in test_gt]
        dates_gt, last_seq_date, tickers = save_dates(
            exp.dates, exp.tickers, exp.test_index, args.pred_len, args.seq_len
        )

        save_results(
            args,
            test_metrics,
            predictions,
            labels,
            dates_gt,
            last_seq_date,
            tickers,
            metrics_path,
        )

    torch.cuda.empty_cache()
    return {"Model:": f"{args.model_name} saved to {model_save_path}"}


def build_parser():
    parser = ArgumentParser(description="Train the SVAT ranking model")
    parser.add_argument("--data_path", type=str, default="./Evaluation/data")
    parser.add_argument("--universe", type=str, default="sx5e")
    parser.add_argument("--model_name", type=str, default="SVAT")
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--start_date", type=str, default="2019-01-01")
    parser.add_argument("--end_train_date", type=str, default="2021-12-31")
    parser.add_argument("--start_valid_date", type=str, default="2022-01-01")
    parser.add_argument("--end_valid_date", type=str, default="2022-12-31")
    parser.add_argument("--start_test_date", type=str, default="2023-01-01")
    parser.add_argument("--end_date", type=str, default="2023-12-31")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--adv_eps", type=float, default=0.005)
    parser.add_argument("--reg_alpha", type=float, default=0.5)
    parser.add_argument("--kl_lambda", type=float, default=1.0)
    return parser


if __name__ == "__main__":
    output = run_train(build_parser().parse_args())
    emit_result(output)
