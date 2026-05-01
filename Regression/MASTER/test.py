import json
import os
import pickle
from argparse import ArgumentParser, Namespace

import pandas as pd
import torch

from .load_dataset import RobustZScoreNormalization, CSVDataset
from .master import MASTERModel
from .train import extract_labels, select_valid_ticker, create_saving_path
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
    model_save_path = os.path.abspath(os.path.join(PROJECT_ROOT, task_type, test_model, model_save_path))
    metrics_path = os.path.abspath(os.path.join(PROJECT_ROOT, task_type, test_model, metrics_path))

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
    predictions, labels, metrics = model.predict(dl_test, args.num_workers)

    metrics = {k: float(v) for k, v in metrics.items()}
    _log(log_fn, f"Metriche calcolate: {metrics}")

    results = {
        "metrics": metrics,
        "preds": predictions,
        "labels": labels,
        "pred_date": dl_test.output_dates,
        "last_date": dl_test.input_dates,
        "tickers": dl_test.tickers_to_date
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task_type", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--universe", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--sl_value", type=int, required=True)
    parser.add_argument("--pl_value", type=int, required=True)
    parser.add_argument("--start_test_date", type=str, required=True)
    parser.add_argument("--end_date", type=str, required=True)
    args = parser.parse_args()

    out = run_test(
        task_type=args.task_type,
        model_name=args.model_name,
        universe=args.universe,
        config_name=args.config_name,
        sl_value=args.sl_value,
        pl_value=args.pl_value,
        start_test_date=args.start_test_date,
        end_date=args.end_date,
    )
    print(json.dumps(out, indent=4))