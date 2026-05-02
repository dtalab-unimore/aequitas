import json
import os
from argparse import ArgumentParser

import pandas as pd
import torch

from utils.load_configuation import _log, load_configuration
from utils.save_results import save_results
from .load_dataset import RobustZScoreNormalization, CSVDataset
from .master import MASTERModel
from .train import extract_labels, select_valid_ticker
from .utils import filter_constituents_by_date

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
        task_type, test_model, universe, config_name, sl_value, pl_value, start_test_date, end_date, PROJECT_ROOT,
        log_fn
    )
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    _log(log_fn, "Caricamento dati alpha...")
    alpha_path = f"{args.data_path}/{args.universe}/{args.universe}_alpha158.csv"
    df_alpha = pd.read_csv(alpha_path)
    _log(log_fn, f"Dati alpha caricati: {len(df_alpha)} righe")

    constituents_path = f"{args.data_path}/constituents/eodhd/{args.universe}.csv"
    tickers = filter_constituents_by_date(
        pd.read_csv(constituents_path),
        args.start_test_date
    )
    selected_tickers = tickers["EODHD"].tolist()
    df_alpha = df_alpha[df_alpha["instrument"].isin(selected_tickers)]

    _log(log_fn, "Caricamento indice di mercato...")
    market_index_path = f"{args.data_path}/{args.nation}_market.csv"
    market_index = pd.read_csv(market_index_path)
    df_alpha = pd.merge(df_alpha, market_index, how="left", on="date")
    _log(log_fn, "Merge con indice di mercato completata.")

    _log(log_fn, "Estrazione label...")
    df_alpha = extract_labels(df_alpha, args)

    _log(log_fn, "Selezione ticker validi sul train...")
    df_alpha = select_valid_ticker(df_alpha, args.start_date, args.end_train_date)

    _log(log_fn, "Calcolo robust z-score sul train set...")
    train_slice = df_alpha[
        (df_alpha["date"] >= args.start_date) & (df_alpha["date"] <= args.end_train_date)
    ]
    robust_z_score = RobustZScoreNormalization(train_slice)

    _log(log_fn, "Creazione dataset di test...")
    dl_test = CSVDataset(
        df_alpha=df_alpha,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        start_date=args.start_test_date,
        end_date=args.end_date,
        z_score=robust_z_score,
        period="test"
    )

    _log(log_fn, "Inizializzazione modello MASTER...")
    model = MASTERModel(
        d_feat=args.gate_input_start_index,
        d_model=args.d_model,
        t_nhead=args.t_nhead,
        s_nhead=args.s_nhead,
        T_dropout_rate=args.dropout,
        S_dropout_rate=args.dropout,
        beta=args.beta,
        n_epochs=args.n_epoch,
        lr=args.lr,
        gate_input_end_index=args.gate_input_end_index,
        gate_input_start_index=args.gate_input_start_index,
        save_path=f"{model_save_path}",
        GPU=args.gpu,
        train_stop_loss_thred=args.train_stop_loss_thred
    )

    model_weights_path = f"{model_save_path}/model.pth"
    _log(log_fn, f"Caricamento pesi da: {model_weights_path}")
    model.load_param(model_weights_path)
    _log(log_fn, "Pesi caricati correttamente.")

    _log(log_fn, "Avvio predizione...")
    preds, labels, metrics = model.predict(dl_test, args.num_workers)

    metrics, results_pkl, metrics_json = save_results(
        args,
        metrics,
        preds,
        labels,
        dl_test.output_dates,
        dl_test.input_dates,
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


if __name__ == "__main__":
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
    print(json.dumps(out, indent=4))