import json
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from Regression.FinFormer.finformer import Finformer_Model
from Regression.FinFormer.load_dataset import RobustZScoreNormalization, CustomDataset
from Regression.FinFormer.train import select_valid_ticker, preprocess_matrix, custom_collate
from utils.filter_constituents import filter_constituents_by_date
from utils.load_configuation import _log, load_configuration
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

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    df_alpha = pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}_alpha360.csv')
    tickers = filter_constituents_by_date(pd.read_csv(f'{args.data_path}/constituents/eodhd/{args.universe}.csv'),
                                          args.start_test_date)
    df_alpha = df_alpha[df_alpha['instrument'].isin(tickers['EODHD'].tolist())]

    # Extract labels
    df_close = pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}.csv')[['date', 'instrument', 'adj_close']]
    df_close = df_close.sort_values(['instrument', 'date'])
    df_close['Label'] = df_close.groupby('instrument')['adj_close'].transform(
        lambda x: (x.shift(-args.pred_len) - x) / x)
    df_alpha = df_alpha.merge(df_close[['date', 'instrument', 'Label']], on=['date', 'instrument'], how='left')

    df_alpha = select_valid_ticker(df_alpha, args.start_date, args.end_train_date)

    robust_z_score = RobustZScoreNormalization(
        df_alpha[(df_alpha['date'] >= args.start_date) & (df_alpha['date'] <= args.end_train_date)])

    df_test = CustomDataset(
        df_alpha,
        d_feat=args.d_feat,
        pred_len=args.pred_len,
        start_date=args.start_test_date,
        end_date=args.end_date,
        z_score=robust_z_score,
        period='test'
    )
    test_loader = DataLoader(df_test, shuffle=False, drop_last=False, collate_fn=custom_collate)

    matrix = np.load(f'{args.data_path}/{args.universe}/{args.universe}_sector_industry_matrix.npz')
    adj_matrix, ticker_list = matrix['adj_matrix'][:11, :, :], matrix['tickers']
    adj_matrix = preprocess_matrix(adj_matrix)

    model = Finformer_Model(d_feat=args.d_feat,
                            hidden_size=args.hidden_size,
                            num_layers=args.num_layers,
                            temporal_dropout=args.temporal_dropout,
                            snum_head=args.snum_head,
                            device=device)

    chk = torch.load(f'{model_save_path}/model.pth', weights_only=False)
    model.finformer.load_state_dict(chk)

    _, _, _, preds, labels, tickers = model.tester(test_loader, adj_matrix, ticker_list)

    test_metrics = model.get_metrics(preds, labels)

    preds = [np.expand_dims(i, axis=1) for i in preds]
    labels = [np.expand_dims(i, axis=1) for i in labels]

    metrics, results_pkl, metrics_json = save_results(
        args,
        test_metrics,
        preds,
        labels,
        test_loader.dataset.valid_dates[args.pred_len:],
        test_loader.dataset.valid_dates[:-args.pred_len],
        tickers,
        metrics_path
    )

    return {
        "metrics": metrics,
        "results_path": results_pkl,
        "metrics_path": metrics_json,
        "device": str(device),
    }


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--task_type", type=str, default='Regression')
    parser.add_argument("--model_name", type=str, default='FinFormer')
    parser.add_argument("--universe", type=str, default='sx5e')
    parser.add_argument("--config_name", type=str, default='FinFormer')
    parser.add_argument("--sl_value", type=int, default=None)
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