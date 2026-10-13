import json
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from .config.model_config import ModelArgs
from .models.samba import SAMBA
from .preprocessing import CustomDataset
from .training import Trainer
from utils.filter_constituents import filter_constituents_by_date
from utils.load_configuation import create_saving_path
from utils.logger import emit_result
from utils.save_results import save_results

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))


def apply_normalization(df, start_date, end_date):
    # Handle of infinity values
    df.iloc[:, 1:] = df.iloc[:, 1:].replace([np.inf, -np.inf], np.nan)

    # Apply Min-Max scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].iloc[:, 1:]
    scaler.fit(train_df)
    df = scaler.transform(df.iloc[:, 1:])
    return df


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


def ticker_checkpoint(model_save_path, ticker):
    return os.path.join(model_save_path, "tickers", f"{ticker.replace('.', '_')}.pth")


def load_frame(args):
    """Technical indicators + close + CNNPred market features of the universe tickers."""
    df_close = pd.read_csv(f"{args.data_path}/{args.universe}/{args.universe}.csv")
    df_close = df_close[['date', 'instrument', 'adj_close']]

    df_tech = pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}_tech.csv')
    df_tech = df_tech[['instrument', 'date', 'vol', 'mom1', 'mom2', 'mom3', 'roc5', 'roc10', 'roc15', 'roc20', 'ema10',
                       'ema20', 'ema50', 'ema200']]

    df_cnnpred = pd.read_csv(f'{args.data_path}/cnnpred_market.csv')

    df = df_tech.merge(df_close, how='inner', on=['instrument', 'date'])
    df = df.merge(df_cnnpred, on='date', how='inner')

    constituents = pd.read_csv(f'{args.data_path}/constituents/eodhd/{args.universe}.csv')
    tickers = filter_constituents_by_date(constituents, args.start_test_date)['EODHD'].tolist()
    df = df[df['instrument'].isin(tickers)]

    df = df[(df['date'] >= args.start_date) & (df['date'] <= args.end_date)]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df, tickers


def ticker_frame(df, ticker, args):
    """Normalized features + label for one ticker, or None if it has too little training data."""
    df_ticker = df[df['instrument'] == ticker].copy()

    train_df = df_ticker[(df_ticker['date'] >= args.start_date) & (df_ticker['date'] <= args.end_train_date)]
    if train_df.shape[0] <= args.seq_len + args.pred_len:
        print(f"Skip {ticker}: only {train_df.shape[0]} samples (<={args.seq_len + args.pred_len})", flush=True)
        return None

    df_ticker = df_ticker.drop(columns=['instrument'])
    df_ticker['Label'] = (df_ticker['adj_close'].shift(-args.pred_len) - df_ticker['adj_close']) / df_ticker['adj_close']

    df_ticker['vol'] = df_ticker['vol'].astype(float)
    df_ticker.iloc[:, 1:-1] = apply_normalization(df_ticker.iloc[:, :-1], args.start_date, args.end_train_date)
    return df_ticker


def period_dataset(df_ticker, args, start, end, period):
    return CustomDataset(X=df_ticker, seq_len=args.seq_len, pred_len=args.pred_len,
                         start_date=start, end_date=end, period=period)


def build_model(args, n_features, device):
    return SAMBA(
        args=ModelArgs(
            d_model=args.d_model,
            n_layer=args.n_layer,
            vocab_size=n_features,
            seq_in=args.seq_len,
            seq_out=args.pred_len,
        ),
        hidden=args.d_model,
        inp=args.seq_len,
        out=args.pred_len,
        embed=args.embed_dim,
        cheb_k=args.d_conv,
    ).to(device)


def predict(model, dataset, args, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    y_pred, y_true = [], []
    with torch.no_grad():
        for data, target in loader:
            y_pred.append(model(data.to(device)).cpu())
            y_true.append(target)
    y_pred = torch.cat(y_pred, dim=0)[:, 0, 0].numpy().tolist()
    y_true = torch.cat(y_true, dim=0).squeeze(-1).numpy().tolist()
    return y_pred, y_true


def collect_results(results, args, metrics_path):
    preds, labels = results["preds"], results["labels"]
    if not preds:
        raise RuntimeError("No ticker had enough data to produce SAMBA predictions.")
    return save_results(
        args, get_metrics(preds, labels),
        np.expand_dims(np.array(preds), axis=1), np.expand_dims(np.array(labels), axis=1),
        results["pred_date"], results["last_date"], results["tickers"], metrics_path,
    )


def run_train(args):
    """SAMBA trains one model per ticker; each best checkpoint is kept for test.py."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_save_path, metrics_path = create_saving_path(args, "Regression", "SAMBA", PROJECT_ROOT)
    os.makedirs(os.path.join(model_save_path, "tickers"), exist_ok=True)

    df, tickers = load_frame(args)

    # The Trainer reads its settings from a dict.
    trainer_args = {**vars(args), "device": device, "debug": True,
                    "log_dir": model_save_path, "model": "SAMBA"}

    results = {"preds": [], "labels": [], "pred_date": [], "last_date": [], "tickers": []}
    val_preds, val_labels = [], []
    trained = []

    for i, ticker in enumerate(tickers, start=1):
        df_ticker = ticker_frame(df, ticker, args)
        if df_ticker is None:
            continue

        train_dataset = period_dataset(df_ticker, args, args.start_date, args.end_train_date, 'train')
        valid_dataset = period_dataset(df_ticker, args, args.start_valid_date, args.end_valid_date, 'valid')
        test_dataset = period_dataset(df_ticker, args, args.start_test_date, args.end_date, 'test')

        if valid_dataset.idx <= 0 or test_dataset.idx <= 0:
            continue
        if any(len(ds.windows) < args.batch_size for ds in (train_dataset, valid_dataset, test_dataset)):
            print(f'Not enough data for ticker {ticker}. Skipping...', flush=True)
            continue

        print(f"Training ticker {ticker} ({i}/{len(tickers)})", flush=True)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                 num_workers=args.num_workers, pin_memory=True)

        model = build_model(args, train_dataset.windows.shape[2], device)
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)

        if args.loss_func == 'mae':
            loss = torch.nn.L1Loss().to(device)
        else:
            loss = torch.nn.MSELoss().to(device)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, eps=1.0e-8, weight_decay=0,
                                     amsgrad=False)

        lr_scheduler = None
        if args.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=[0.5 * args.epochs, 0.7 * args.epochs, 0.9 * args.epochs],
                gamma=0.1
            )

        trainer = Trainer(model, loss, optimizer, train_loader, val_loader, test_loader, args=trainer_args,
                          lr_scheduler=lr_scheduler)
        trainer.train()  # leaves the best validation state loaded in the model
        torch.save(model.state_dict(), ticker_checkpoint(model_save_path, ticker))
        trained.append(ticker)

        y_pred, y_true = predict(model, test_dataset, args, device)
        val_y_pred, val_y_true = predict(model, valid_dataset, args, device)

        results["preds"].extend(y_pred)
        results["labels"].extend(y_true)
        results["tickers"].extend([ticker] * len(y_pred))
        results["pred_date"].extend(test_dataset.dates)
        results["last_date"].extend(test_dataset.last_dates)
        val_preds.extend(val_y_pred)
        val_labels.extend(val_y_true)

    args.trained_tickers = trained
    with open(f'{model_save_path}/config_params.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    with open(f'{metrics_path}/val_metrics_sl{args.seq_len}_pl{args.pred_len}.json', 'w') as f:
        json.dump(get_metrics(val_preds, val_labels), f, indent=4)

    metrics, results_pkl, metrics_json = collect_results(results, args, metrics_path)
    return {"model_path": os.path.join(model_save_path, "tickers"), "metrics": metrics,
            "results_path": results_pkl, "metrics_path": metrics_json}


def build_parser():
    args = ArgumentParser()

    # Dataset parameters
    args.add_argument('--data_path', type=str, default='./Evaluation/data')
    args.add_argument('--universe', type=str, default='sp500')
    args.add_argument('--model_name', type=str, default='SAMBA')

    # Prediction parameters
    args.add_argument('--seq_len', type=int, default=5)
    args.add_argument('--pred_len', type=int, default=1)
    args.add_argument('--start_date', type=str, default='2018-01-01')
    args.add_argument('--end_train_date', type=str, default='2021-12-31')
    args.add_argument('--start_valid_date', type=str, default='2022-01-01')
    args.add_argument('--end_valid_date', type=str, default='2022-12-31')
    args.add_argument('--start_test_date', type=str, default='2023-01-01')
    args.add_argument('--end_date', type=str, default='2023-12-31')

    # Model Hyperparameters
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--seed', type=int, default=1)
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--epochs', type=int, default=1500)
    args.add_argument('--d_model', type=int, default=32)
    args.add_argument('--n_layer', type=int, default=3)
    args.add_argument('--embed_dim', type=int, default=10)
    args.add_argument('--d_conv', type=int, default=3)
    args.add_argument('--loss_func', type=str, default='mae', choices=('mse', 'mae'))
    args.add_argument('--lr_decay', type=bool, default=True)
    args.add_argument('--early_stop', type=bool, default=True)
    args.add_argument('--early_stop_patience', type=int, default=200)
    args.add_argument('--grad_norm', type=bool, default=False)
    args.add_argument('--max_grad_norm', type=float, default=5.0)
    args.add_argument('--log_step', type=int, default=20)
    args.add_argument('--num_workers', type=int, default=4)
    return args


if __name__ == '__main__':
    output = run_train(build_parser().parse_args())
    emit_result(output)
