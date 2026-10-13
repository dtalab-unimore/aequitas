import os
from argparse import ArgumentParser

import pandas as pd
import torch

from utils.filter_constituents import filter_constituents_by_date
from utils.load_configuation import load_configuration
from utils.logger import _log, emit_result
from utils.save_results import save_results
from utils.z_score import RobustZScoreNormalization
from .load_dataset import CSVDataset
from .model.MATCC import MATCC
from .train import select_valid_ticker, _init_data_loader, valid_epoch

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

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    df_alpha = pd.read_csv(f"{args.data_path}/{args.universe}/{args.universe}_alpha158.csv")
    tickers = filter_constituents_by_date(pd.read_csv(f'{args.data_path}/constituents/eodhd/{args.universe}.csv'), args.start_test_date)
    df_alpha = df_alpha[df_alpha['instrument'].isin(tickers['EODHD'].tolist())]

    market_index = pd.read_csv(f'{args.data_path}/{args.nation}_market.csv')
    df_alpha = pd.merge(df_alpha, market_index, how='left', on='date')

    df_close = pd.read_csv(f"{args.data_path}/{args.universe}/{args.universe}.csv")[['date', 'instrument', 'adj_close']]
    df_close = df_close.sort_values(['instrument', 'date'])
    df_close['Label'] = df_close.groupby('instrument')['adj_close'].transform(lambda x: (x.shift(-args.pred_len) - x) / x)
    df_alpha = df_alpha.merge(df_close[['date', 'instrument', 'Label']], on=['date', 'instrument'], how='left')

    df_alpha = select_valid_ticker(df_alpha, args.start_date, args.end_train_date)

    robust_z_score = RobustZScoreNormalization(
        df_alpha[(df_alpha['date'] >= args.start_date) & (df_alpha['date'] <= args.end_train_date)])

    dl_test = CSVDataset(
        df_alpha=df_alpha,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        start_date=args.start_test_date,
        end_date=args.end_date,
        z_score=robust_z_score,
        period='test'
    )

    test_loader = _init_data_loader(dl_test, shuffle=False, drop_last=False, num_workers=args.num_workers)

    model = MATCC(
        d_model=args.d_model,
        d_feat=args.gate_input_start_index,
        seq_len=args.seq_len,
        t_nhead=args.n_head,
        S_dropout_rate=args.dropout,
        gate_input_start_index=args.gate_input_start_index,
        gate_input_end_index=args.gate_input_end_index
    ).to(device)
    model.load_state_dict(torch.load(f'{model_save_path}/model.pth', map_location=device))

    test_loss, metrics, preds, labels, mask = valid_epoch(test_loader, model, device)
    output_dates = [date for i, date in enumerate(dl_test.output_dates) if mask[i]]
    input_dates = [date for i, date in enumerate(dl_test.input_dates) if mask[i]]

    metrics, results_pkl, metrics_json = save_results(
        args,
        metrics,
        preds,
        labels,
        output_dates,
        input_dates,
        dl_test.tickers_to_date,
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
    parser.add_argument("--task_type", type=str, default='Regression')
    parser.add_argument("--model_name", type=str, default='MATCC')
    parser.add_argument("--universe", type=str, default='sxxp')
    parser.add_argument("--config_name", type=str, default='MATCC')
    parser.add_argument("--sl_value", type=int, default=5)
    parser.add_argument("--pl_value", type=int, default=1)
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