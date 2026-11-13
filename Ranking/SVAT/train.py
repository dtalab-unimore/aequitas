import json
import os
from argparse import ArgumentParser
from datetime import date

import torch

from Ranking.SVAT.experiments.exp_svat import Exp_SVAT
from Ranking.SVAT.utils.config import build_param_dict
from Ranking.SVAT.utils.metric import get_metrics
from Ranking.SVAT.utils.optimize import set_seed
from utils.load_configuation import create_saving_path
from utils.logger import emit_result
from utils.training import save_valid_metrics, training_result

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

    model_save_path, _ = create_saving_path(
        args, "Ranking", "SVAT", PROJECT_ROOT
    )

    with open(os.path.join(model_save_path, "config_params.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpus = 0 if device.type == "cuda" else None

    set_seed(args.seed)
    exp = Exp_SVAT(build_param_dict(args), gpus=gpus)
    valid_pred, valid_gt, _ = exp.train(model_save_path)

    valid_pred = valid_pred.swapaxes(0, 1)
    valid_gt = valid_gt.swapaxes(0, 1)
    valid_metrics = get_metrics(valid_gt, valid_pred)
    valid_path = save_valid_metrics(valid_metrics, model_save_path, args.seq_len, args.pred_len)

    torch.cuda.empty_cache()
    return training_result(os.path.join(model_save_path, "model.pth"), valid_metrics, valid_path)


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
