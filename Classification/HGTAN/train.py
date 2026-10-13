import json
import os
from argparse import ArgumentParser

import numpy as np
import torch
from torch import optim

from .model.Optim import ScheduledOptim
from .model.models import HGTAN
from .tool import train, eval_model, get_metrics, prepare_data
from .load_data import load_data, load_matrices, extract_dates
from utils.load_configuation import create_saving_path
from utils.logger import emit_result
from utils.save_results import save_results

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if __name__ == '__main__':
    args = ArgumentParser()

    args.add_argument('--data_path', type=str, default='./Evaluation/data', help='Path to the dataset')
    args.add_argument('--universe', type=str, default='sx5e', help='Universe of stocks to use')
    args.add_argument('--model_name', type=str, default='HGTAN', help='Name of the model to use')

    args.add_argument('--pred_len', type=int, default=5, help='Steps for future prediction')
    args.add_argument('--seq_len', type=int, default=20, help='Lookback length for the model')
    args.add_argument('--start_date', type=str, default='2018-01-01', help='Start date for the dataset')
    args.add_argument('--end_train_date', type=str, default='2021-12-31', help='End date for training set')
    args.add_argument('--start_valid_date', type=str, default='2022-01-01', help='Start date for validation set')
    args.add_argument('--end_valid_date', type=str, default='2022-12-31', help='End date for validation set')
    args.add_argument('--start_test_date', type=str, default='2023-01-01', help='Start date for the test set')
    args.add_argument('--end_date', type=str, default='2023-12-31', help='End date for the dataset')

    args.add_argument('--feature', type=int, default=9, help='Number of features to use')
    args.add_argument('--n_class', type=int, default=3, help='Number of classes to use')
    args.add_argument('--n_epochs', type=int, default=600, help='Number of epochs')
    args.add_argument('--batch_size', type=int, default=64, help='Batch length')
    args.add_argument('--rnn_unit', type=int, default=32, help='Number of GRU hidden units.')
    args.add_argument('--d_model', type=int, default=16)
    args.add_argument('--d_k', type=int, default=8)
    args.add_argument('--d_v', type=int, default=8)
    args.add_argument('--n_head', type=int, default=4)
    args.add_argument('--n_layers', type=int, default=3)
    args.add_argument('--hidden', type=int, default=8, help='Number of hidden units')
    args.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    args.add_argument('--proj_share_weight', default=True, action='store_true')
    args.add_argument('--seed', type=int, default=42, help='Random seed')

    args.add_argument('--job_id', type=int, default=0, help='Job identifier to save the model')

    args.add_argument('--label_smoothing', default=True, action='store_true')
    args.add_argument('--n_warmup_steps', type=int, default=4000)
    args.add_argument('--steps', type=int, default=1, help='Steps to make prediction')

    args = args.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    args.d_word_vec = args.d_model

    model_save_path, metrics_path = create_saving_path(args, "Classification", "HGTAN", PROJECT_ROOT)

    with open(f'{model_save_path}/config_params.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    print('Extracting data...')
    eod_data, ground_truth, valid_index, test_index, dates, tickers = load_data(args)
    pred_dates, last_dates = extract_dates(test_index, dates, args)
    adj, H = load_matrices(tickers, args)

    len_data = eod_data.shape[1] - args.seq_len - args.pred_len + 1
    train_loader = prepare_data(eod_data, ground_truth, 0, valid_index, args)
    valid_loader = prepare_data(eod_data, ground_truth, valid_index, test_index, args, flag='valid')
    test_loader = prepare_data(eod_data, ground_truth, test_index, len_data, args, flag='test')

    model = HGTAN(
        rnn_unit=args.rnn_unit,
        n_hid=args.hidden,
        n_class=args.n_class,
        feature=args.feature,
        tgt_emb_prj_weight_sharing=args.proj_share_weight,
        d_k=args.d_k,
        d_v=args.d_v,
        d_model=args.d_model,
        d_word_vec=args.d_word_vec,
        n_head=args.n_head,
        dropout=args.dropout
    ).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        args.d_model, args.n_warmup_steps)

    train(model, train_loader, valid_loader, adj, H, optimizer, device, args, model_save_path,metrics_path)

    preds, labels = eval_model(model, test_loader, adj, H, device, args, model_save_path)

    metrics = get_metrics(preds, labels)

    metrics, results_pkl, metrics_json = save_results(
        args,
        metrics,
        preds,
        labels,
        pred_dates.dt.strftime("%Y-%m-%d").to_list(),
        last_dates.dt.strftime("%Y-%m-%d").to_list(),
        [tickers] * len(preds),
        metrics_path
    )

    print('Test Metrics:\n', metrics)

    emit_result({
        "model_path": f"{model_save_path}/model.pth",
        "metrics": metrics,
        "results_path": results_pkl,
        "metrics_path": metrics_json,
    })