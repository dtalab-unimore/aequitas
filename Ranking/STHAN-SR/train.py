import json
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch.random

from .evaluator import get_metrics
from .load_datasets import load_dataset, load_incidence_matrix, save_dates
from .model import ReRaLSTM
from utils.load_configuation import create_saving_path
from utils.logger import emit_result
from utils.save_results import save_results

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if __name__=='__main__':
    
    args = ArgumentParser(description='Train a relational rank lstm model')

    args.add_argument('--data_path', type=str, default='./Evaluation/data', help='Path to the dataset')
    args.add_argument('--universe', type=str, default='sx5e', help='Universe of stocks to use')
    args.add_argument('--model_name', type=str, default='STHAN-SR', help='Name of the model to use')

    args.add_argument('--pred_len', type=int, default=5, help='Steps for future prediction')
    args.add_argument('--seq_len', type=int, default=10, help='Lookback length for the model')
    args.add_argument('--start_date', type=str, default='2019-01-01', help='Start date for the dataset')
    args.add_argument('--end_train_date', type=str, default='2021-12-31', help='End date for training set')
    args.add_argument('--start_valid_date', type=str, default='2022-01-01', help='Start date for validation set')
    args.add_argument('--end_valid_date', type=str, default='2022-12-31', help='End date for validation set')
    args.add_argument('--start_test_date', type=str, default='2023-01-01', help='Start date for the test set')
    args.add_argument('--end_date', type=str, default='2023-12-31', help='End date for the dataset')

    args.add_argument('--lr', type=float, default=0.001,  help='learning rate')
    args.add_argument('--a', type=float, default=0.1, help='alpha, the weight of ranking loss')
    args.add_argument('--gpu', type=int, default=0, help='use gpu')
    args.add_argument('--seed', type=int, default=42, help='random seed')
    args.add_argument('--epochs', type=int, default=500, help='number of epochs')

    args.add_argument('--early_stopping', type=int, default=5, help='Early stopping criteria.')

    args = args.parse_args()

    model_save_path, metrics_path = create_saving_path(args, "Ranking", "STHAN-SR", PROJECT_ROOT)

    with open(f'{model_save_path}/config_params.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

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

    test_gt, test_pred, valid_gt, valid_pred = RR_LSTM.train(model_save_path)

    valid_metrics = get_metrics(valid_gt, valid_pred)
    valid_metrics = {k: float(v) for k, v in valid_metrics.items()}
    print(f'Validation Metrics: {valid_metrics}')

    with open(f'{metrics_path}/val_metrics_sl{args.seq_len}_pl{args.pred_len}.json', 'w') as f:
        json.dump(valid_metrics, f, indent=4)

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
        metrics_path
    )

    print(f'Metrics: {test_metrics}')

    emit_result({
        "model_path": f"{model_save_path}/model.pth",
        "metrics": metrics,
        "results_path": results_pkl,
        "metrics_path": metrics_json,
    })
