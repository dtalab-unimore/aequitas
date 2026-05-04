import json
import os
from argparse import ArgumentParser

import numpy as np
import torch

from .evaluator import get_metrics
from .model import ReRaLSTM
from .load_datasets import load_dataset, load_incidence_matrix, save_dates
from utils.load_configuation import load_configuration, _log
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
    _log(log_fn, "Preparazione del test...")

    args, model_save_path, metrics_path = load_configuration(
        task_type, test_model, universe, config_name, sl_value, pl_value, start_test_date, end_date, PROJECT_ROOT, log_fn
    )

    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'

    eod_data, gt_data, price_data, valid_index, test_index, dates, tickers = load_dataset(
        args.data_path, args.universe, args.start_date, args.end_train_date, args.end_valid_date, args.start_test_date,
        args.end_date, args.pred_len, args.seq_len)

    mask_data = np.ones((eod_data.shape[0], eod_data.shape[1]), dtype=float)

    inci_matr = load_incidence_matrix(tickers, args)

    RR_LSTM = ReRaLSTM(
        eod_data=eod_data,
        gt_data=gt_data,
        price_data=price_data,
        mask_data=mask_data,
        tickers=tickers,
        inci_matrix=inci_matr,
        seq_len=args.seq_len,
        steps=args.pred_len,
        valid_index=valid_index,
        test_index=test_index,
        epochs=args.epochs,
        gpu=args.gpu,
        device=device,
        alpha=args.a,
        lr=args.lr,
        early_stop=args.early_stopping
    )

    test_gt, test_pred = RR_LSTM.test(model_save_path)

    test_gt = test_gt.swapaxes(0, 1)
    test_pred = test_pred.swapaxes(0, 1)

    test_metrics = get_metrics(test_gt, test_pred)
    test_pred = [np.expand_dims(test_pred[i], axis=1) for i in range(test_pred.shape[0])]
    test_gt = [np.expand_dims(test_gt[i], axis=1) for i in range(test_gt.shape[0])]

    dates_gt, last_seq_date, tickers = save_dates(dates, tickers, test_index, args.pred_len, args.seq_len)

    metrics, results_pkl, metrics_json = save_results(
        args,
        test_metrics,
        test_pred,
        test_gt,
        dates_gt.to_list(),
        last_seq_date.to_list(),
        tickers,
        metrics_path,
        log_fn=log_fn
    )

    return {
        "metrics": metrics,
        "results_path": results_pkl,
        "metrics_path": metrics_json,
        "device": str(device),
    }

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--task_type", type=str, default='Ranking')
    parser.add_argument("--model_name", type=str, default='STHAN-SR')
    parser.add_argument("--universe", type=str, default='sx5e')
    parser.add_argument("--config_name", type=str, default='STHAN-SR')
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
    print(json.dumps(out, indent=4))