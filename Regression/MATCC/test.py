import json
import os
import pickle
from argparse import Namespace, ArgumentParser

import pandas as pd
import torch

from .load_dataset import RobustZScoreNormalization, CSVDataset
from .model.MATCC import MATCC
from .train import create_saving_path, select_valid_ticker, _init_data_loader, valid_epoch
from .utils import filter_constituents_by_date

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))


def _log(log_fn, msg):
    if log_fn is not None:
        log_fn(msg)


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

    config_dir = os.path.join(
        PROJECT_ROOT,
        task_type,
        test_model,
        "model_params",
        universe,
        config_name,
        f"sl_{sl_value}",
        f"pl_{pl_value}",
    )

    config_path = os.path.join(config_dir, "config_params.json")
    _log(log_fn, f"Lettura configurazione da: {config_path}")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config_params.json non trovato: {config_path}")

    with open(config_path, "r", encoding="utf-8-sig") as f:
        saved_args = json.load(f)

    args_dict = saved_args.copy()
    args_dict["start_test_date"] = start_test_date
    args_dict["end_date"] = end_date
    args_dict["model_name"] = config_name
    args_dict["universe"] = universe
    args_dict["seq_len"] = sl_value
    args_dict["pred_len"] = pl_value
    args_dict["data_path"] = "../Evaluation/data"

    args = Namespace(**args_dict)

    _log(log_fn, "Creazione dei path di salvataggio...")
    model_save_path, metrics_path = create_saving_path(args)

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

    test_loss, metrics, preds, labels = valid_epoch(test_loader, model, device)

    metrics = {k: float(v) for k, v in metrics.items()}
    results = {
        'metrics': metrics,
        'preds': preds,
        'labels': labels,
        'pred_date': dl_test.output_dates,
        'last_date': dl_test.input_dates,
        'tickers': dl_test.tickers_to_date
    }

    results_pkl = f"{metrics_path}/results_sl{args.seq_len}_pl{args.pred_len}.pkl"
    metrics_json = f"{metrics_path}/metrics_sl{args.seq_len}_pl{args.pred_len}.json"

    with open(results_pkl, "wb") as f:
        pickle.dump(results, f)

    with open(metrics_json, "w") as f:
        json.dump(metrics, f)

    _log(log_fn, "Test completato con successo.")

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
    parser.add_argument("--end_date", type=str, default='2023-01-01')
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