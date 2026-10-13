# -*-Encoding: utf-8 -*-
import argparse
import json
import os
import random
import shutil

import numpy as np
import torch
from sklearn.metrics import r2_score

from .exp.exp_model import Exp_Model
from .preprocessing import load_dataset, ticker_csv_name
from utils.load_configuation import create_saving_path
from utils.logger import emit_result
from utils.save_results import save_results

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))


def get_metrics(preds, labels):
    preds_flat = np.array(preds)
    labels_flat = np.array(labels)

    metrics = {
        'MSE': np.mean((preds_flat - labels_flat) ** 2),
        'MAE': np.mean(np.abs(preds_flat - labels_flat)),
        'RMSE': np.sqrt(np.mean((preds_flat - labels_flat) ** 2)),
        'R2': r2_score(preds_flat, labels_flat)
    }
    return {k: float(v) for k, v in metrics.items()}


def as_2d(values):
    """(samples, pred_len), also for pred_len 1: the Evaluation step slices [:, :pl]."""
    return np.array(values).reshape(len(values), -1)


def ticker_checkpoint(model_save_path, ticker):
    return os.path.join(model_save_path, "tickers", f"{ticker.replace('.', '_')}.pth")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def check_lengths(args):
    # The hierarchical VAE decodes the latent into a target sequence as long as
    # the input, halving the resolution at each scale: other lengths fail deep
    # inside the network with a tensor size mismatch.
    if args.prediction_length != args.sequence_length or args.sequence_length % 2:
        raise ValueError(
            f"D-Va needs prediction length = sequence length, and an even length "
            f"(got sequence length {args.sequence_length}, prediction length {args.prediction_length})."
        )


def run_train(args):
    """D-Va trains one model per ticker; each best checkpoint is kept for test.py."""
    check_lengths(args)
    set_seed(args.seed)
    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)
    args.seq_len, args.pred_len = args.sequence_length, args.prediction_length

    model_save_path, metrics_path = create_saving_path(args, "Regression", "D-Va", PROJECT_ROOT)
    os.makedirs(os.path.join(model_save_path, "tickers"), exist_ok=True)

    data_path = args.data_path
    args.root_path = os.path.join(model_save_path, "data_preprocessed")
    args.checkpoints = os.path.join(model_save_path, "checkpoints")

    tickers = load_dataset(args.start_date, args.end_date, args.end_train_date, args.end_valid_date,
                           args.start_test_date, data_path, args.universe, args.root_path,
                           args.batch_size, args.sequence_length, args.prediction_length)
    tickers = sorted(set(tickers))
    print(f"{len(tickers)} tickers with enough data", flush=True)

    results = {"preds": [], "labels": [], "pred_date": [], "last_date": [], "tickers": []}
    val_preds, val_labels = [], []
    trained = []

    for i, ticker in enumerate(tickers, start=1):
        print(f"Training ticker {ticker} ({i}/{len(tickers)})", flush=True)

        args.data_path = ticker_csv_name(ticker)
        setting = f"{args.data_path}_sl{args.sequence_length}_pl{args.prediction_length}_seed{args.seed}"
        exp = Exp_Model(args)
        exp.train(setting)
        pred, label, pred_date, last_date = exp.test(setting, args.sequence_length, args.prediction_length)

        if np.isnan(pred).any() or np.isnan(label).any():
            print(f'NaN predictions for ticker {ticker}, skipping...', flush=True)
        else:
            # exp.train leaves the best checkpoint loaded in denoise_net
            torch.save(exp.denoise_net.state_dict(), ticker_checkpoint(model_save_path, ticker))
            trained.append(ticker)

            results["preds"].extend(pred)
            results["labels"].extend(label)
            results["pred_date"].extend(pred_date)
            results["last_date"].extend(last_date)
            results["tickers"].extend([ticker] * len(pred))

            val_pred, val_label, _, _ = exp.validate(setting, args.sequence_length, args.prediction_length)
            val_preds.extend(val_pred)
            val_labels.extend(val_label)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        os.remove(os.path.join(args.root_path, args.data_path))
        shutil.rmtree(os.path.join(args.checkpoints, setting), ignore_errors=True)

    shutil.rmtree(args.root_path, ignore_errors=True)
    shutil.rmtree(args.checkpoints, ignore_errors=True)

    if not results["preds"]:
        raise RuntimeError("No ticker had enough data to produce D-Va predictions.")

    args.data_path = data_path
    args.trained_tickers = trained
    with open(f'{model_save_path}/config_params.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    with open(f'{metrics_path}/val_metrics_sl{args.seq_len}_pl{args.pred_len}.json', 'w') as f:
        json.dump(get_metrics(val_preds, val_labels), f, indent=4)

    metrics, results_pkl, metrics_json = save_results(
        args, get_metrics(results["preds"], results["labels"]),
        as_2d(results["preds"]), as_2d(results["labels"]),
        results["pred_date"], results["last_date"], results["tickers"], metrics_path,
    )
    return {"model_path": os.path.join(model_save_path, "tickers"), "metrics": metrics,
            "results_path": results_pkl, "metrics_path": metrics_json}


def build_parser():
    parser = argparse.ArgumentParser(description='Train D-Va (one model per ticker)')

    # Load data
    parser.add_argument('--universe', type=str, default='sx5e')
    parser.add_argument('--model_name', type=str, default='DVa')
    parser.add_argument('--data_path', type=str, default='./Evaluation/data', help='root path of the data files')

    parser.add_argument('--seq_len', '--sequence_length', dest='sequence_length', type=int, default=2,
                        help='length of input sequence')
    parser.add_argument('--pred_len', '--prediction_length', dest='prediction_length', type=int, default=1,
                        help='prediction sequence length')
    parser.add_argument('--start_date', type=str, default='2018-01-01', help='start date')
    parser.add_argument('--end_train_date', type=str, default='2021-12-31', help='train end date')
    parser.add_argument('--start_valid_date', type=str, default='2022-01-01', help='validation start date')
    parser.add_argument('--end_valid_date', type=str, default='2022-12-31', help='validation end date')
    parser.add_argument('--start_test_date', type=str, default='2023-01-01', help='test start date')
    parser.add_argument('--end_date', type=str, default='2023-12-31', help='end date')

    parser.add_argument('--target_dim', type=int, default=1, help='dimension of target')
    parser.add_argument('--input_dim', type=int, default=6, help='dimension of input')
    parser.add_argument('--hidden_size', type=int, default=128, help='encoder dimension')
    parser.add_argument('--embedding_dimension', type=int, default=64, help='feature embedding dimension')

    # Diffusion process
    parser.add_argument('--diff_steps', type=int, default=1000, help='number of the diff step')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout')
    parser.add_argument('--beta_schedule', type=str, default='linear', help='the schedule of beta')
    parser.add_argument('--beta_start', type=float, default=0.0, help='start of the beta')
    parser.add_argument('--beta_end', type=float, default=1.0, help='end of the beta')
    parser.add_argument('--scale', type=float, default=0.1, help='adjust diffusion scale')

    # Bidirectional VAE
    parser.add_argument('--arch_instance', type=str, default='res_mbconv', help='path to the architecture instance')
    parser.add_argument('--mult', type=float, default=1, help='mult of channels')
    parser.add_argument('--num_layers', type=int, default=2, help='num of RNN layers')
    parser.add_argument('--num_channels_enc', type=int, default=32, help='number of channels in encoder')
    parser.add_argument('--channel_mult', type=int, default=2, help='number of channels in encoder')
    parser.add_argument('--num_preprocess_blocks', type=int, default=1, help='number of preprocessing blocks')
    parser.add_argument('--num_preprocess_cells', type=int, default=3, help='number of cells per block')
    parser.add_argument('--groups_per_scale', type=int, default=2, help='number of cells per block')
    parser.add_argument('--num_postprocess_blocks', type=int, default=1, help='number of postprocessing blocks')
    parser.add_argument('--num_postprocess_cells', type=int, default=2, help='number of cells per block')
    parser.add_argument('--num_channels_dec', type=int, default=32, help='number of channels in decoder')
    parser.add_argument('--num_latent_per_group', type=int, default=8, help='number of channels in latent variables per group')

    # Training settings
    parser.add_argument('--num_workers', type=int, default=5, help='data loader num workers')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0000, help='weight decay')
    parser.add_argument('--zeta', type=float, default=0.5, help='trade off parameter zeta')
    parser.add_argument('--eta', type=float, default=1.0, help='trade off parameter eta')
    parser.add_argument('--seed', type=int, default=100, help='random seed for reproducibility')

    # Device
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    return parser


if __name__ == '__main__':
    output = run_train(build_parser().parse_args())
    emit_result(output)
