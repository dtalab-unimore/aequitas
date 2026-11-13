#!/usr/bin/env python
import json
import os
from argparse import ArgumentParser

from Ranking.RTGCN.processor.recognition import REC_Processor
from utils.load_configuation import create_saving_path
from utils.logger import emit_result
from utils.training import training_result

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))


def run_train(
    universe,
    model_name,
    seq_len,
    pred_len,
    start_date,
    end_train_date,
    start_valid_date,
    end_valid_date,
):
    processor_args = [
        "--model_name", str(model_name),
        "--universe", str(universe),
        "--seq_len", str(seq_len),
        "--pred_len", str(pred_len),
        "--start_date", str(start_date),
        "--end_train_date", str(end_train_date),
        "--start_valid_date", str(start_valid_date),
        "--end_valid_date", str(end_valid_date),
    ]

    p = REC_Processor(processor_args)

    model_save_path, _ = create_saving_path(p.arg, "Ranking", "RTGCN", PROJECT_ROOT)

    with open(f'{model_save_path}/config_params.json', 'w') as f:
        json.dump(vars(p.arg), f, indent=4)

    # Writes val_metrics_sl*_pl*.json in model_save_path
    valid_metrics = p.start(model_save_path)
    valid_path = os.path.join(model_save_path, f"val_metrics_sl{p.arg.seq_len}_pl{p.arg.pred_len}.json")

    return training_result(os.path.join(model_save_path, "model.pth"), valid_metrics, valid_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--universe", type=str, default='sx5e')
    parser.add_argument("--model_name", type=str, default='RT-GCN')
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--start_date", type=str, default='2023-01-01')
    parser.add_argument("--end_train_date", type=str, default='2023-12-31')
    parser.add_argument('--start_valid_date', type=str, default='2024-01-01')
    parser.add_argument('--end_valid_date', type=str, default='2024-06-30')
    args = parser.parse_args()

    out = run_train(
        universe=args.universe,
        model_name=args.model_name,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        start_date=args.start_date,
        end_train_date=args.end_train_date,
        start_valid_date=args.start_valid_date,
        end_valid_date=args.end_valid_date,
    )
    emit_result(out)



