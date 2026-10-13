import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data_loader import CustomDataset
from .model.Thgnn import StockHeteGAT
from .train import extract_labels, select_valid_ticker, calcultate_matrics
from .trainer.trainer import eval_model, bce_loss
from utils.filter_constituents import filter_constituents_by_date
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
        task_type, test_model, universe, config_name, sl_value, pl_value, start_test_date, end_date, PROJECT_ROOT, log_fn
    )

    dataset = pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}.csv')
    dataset = dataset[['date', 'instrument', 'open', 'high', 'low', 'close', 'volume', 'adj_close']]
    constituents = pd.read_csv(f'{args.data_path}/constituents/eodhd/{args.universe}.csv')
    tickers = filter_constituents_by_date(constituents, args.start_test_date)['EODHD'].tolist()
    dataset = dataset[dataset['instrument'].isin(tickers)]

    corr_matrix = dict(np.load(f'{args.data_path}/{args.universe}/{args.universe}_corr_matrix.npz', allow_pickle=True))

    dataset = extract_labels(dataset, args)
    dataset, corr_matrix, tickers, dates = select_valid_ticker(dataset, corr_matrix, args.start_date, args.end_date)

    all_keys = pd.DataFrame(tickers, columns=['instrument']).merge(dates, how='cross')
    dataset = pd.merge(all_keys, dataset, on=['instrument', 'date'], how='left')
    dataset = dataset.sort_values(['instrument', 'date'])
    df_filled = dataset.groupby('instrument').ffill()
    dataset[df_filled.columns] = df_filled

    dl_test = CustomDataset(
        dataset=dataset.copy(),
        correlation_matrix=corr_matrix.copy(),
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        start_date=args.start_test_date,
        end_date=args.end_date,
        period='test'
    )
    test_loader = DataLoader(dl_test, batch_size=args.batch_size, pin_memory=True)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StockHeteGAT(hidden_dim=args.hidden_dim, num_heads=args.num_heads,
                         out_features=args.out_features).to(args.device)

    state = torch.load(f"{model_save_path}/model.pth", map_location=args.device)
    model.load_state_dict(state['model'])

    _, preds, labels = eval_model(args=args, model=model, dataset_eval=test_loader, loss_fcn=bce_loss)

    metrics = calcultate_matrics(preds, labels)

    metrics, results_pkl, metrics_json = save_results(
        args,
        metrics,
        preds,
        labels,
        dl_test.pred_dates,
        dl_test.last_seq_dates,
        [tickers]*len(preds),
        metrics_path,
        log_fn=log_fn
    )

    return {
        "metrics": metrics,
        "results_path": results_pkl,
        "metrics_path": metrics_json,
        "device": str(args.device),
    }

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--task_type", type=str, default='Classification')
    parser.add_argument("--model_name", type=str, default='THGNN')
    parser.add_argument("--universe", type=str, default='sx5e')
    parser.add_argument("--config_name", type=str, default='THGNN')
    parser.add_argument("--sl_value", type=int, default=20)
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