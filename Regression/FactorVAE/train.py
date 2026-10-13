import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from .dataset import init_data_loader
from .model import train, validate
from .module import FactorVAE, FeatureExtractor, FactorDecoder, FactorEncoder, FactorPredictor, AlphaLayer, BetaLayer
from .utils import set_seed

from utils.filter_constituents import filter_constituents_by_date
from utils.load_configuation import create_saving_path
from utils.logger import emit_result
from utils.save_results import save_results
from utils.z_score import RobustZScoreNormalization

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))


def extract_labels(df, args, pred_len):
    df_close = pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}.csv')[['date', 'instrument', 'adj_close']]
    df_close.rename(columns={'date': 'datetime'}, inplace=True)
    df_close['datetime'] = pd.to_datetime(df_close['datetime'])
    df_close = df_close.sort_values(['instrument', 'datetime'])
    df_close['Label'] = df_close.groupby('instrument')['adj_close'].transform(lambda x: (x.shift(-pred_len) - x) / x)
    df = df.merge(df_close[['datetime', 'instrument', 'Label']], on=['datetime', 'instrument'], how='left')
    return df


def select_valid_ticker(df, start_date, end_date):
    df_train = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
    tickers = df_train['instrument'].drop_duplicates().tolist()
    df = df[df['instrument'].isin(tickers)]
    return df


def cs_zscore(df, clip=None):
    def zscore(group):
        mean = group.mean()
        std = group.std()
        z = (group - mean) / std
        if clip is not None:
            z = z.clip(lower=-clip, upper=clip)
        return z

    return df.groupby(level='datetime')['Label'].transform(zscore)


def prepare_dataset(args):
    """Alpha158 features + labels, robust z-scored on the training period."""
    dataset = pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}_alpha158.csv')
    tickers = filter_constituents_by_date(
        pd.read_csv(f'{args.data_path}/constituents/eodhd/{args.universe}.csv'), args.start_test_date
    )
    dataset = dataset[dataset['instrument'].isin(tickers['EODHD'].tolist())].copy()

    dataset.rename(columns={'date': 'datetime'}, inplace=True)
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])

    dataset = extract_labels(dataset, args, pred_len=args.pred_len)
    dataset = select_valid_ticker(dataset, pd.to_datetime(args.start_date), pd.to_datetime(args.end_train_date))

    dataset.set_index(['datetime', 'instrument'], inplace=True)
    dataset = dataset.reorder_levels(['datetime', 'instrument']).sort_index()
    dates = dataset.index.get_level_values('datetime')
    train_mask = (dates >= pd.to_datetime(args.start_date)) & (dates <= pd.to_datetime(args.end_train_date))
    robust_z = RobustZScoreNormalization(dataset[train_mask])
    dataset = robust_z.robust_zscore(dataset)

    dataset = dataset[dataset['Label'].notna()].fillna(0)
    dataset['Label'] = cs_zscore(dataset, clip=3.0)
    return dataset


def build_model(args):
    feature_extractor = FeatureExtractor(num_latent=args.num_latent, hidden_size=args.hidden_size)
    factor_encoder = FactorEncoder(num_factors=args.num_factor, num_portfolio=args.num_portfolio,
                                   hidden_size=args.hidden_size)
    alpha_layer = AlphaLayer(args.hidden_size)
    beta_layer = BetaLayer(args.hidden_size, args.num_factor)
    factor_decoder = FactorDecoder(alpha_layer, beta_layer)
    factor_predictor = FactorPredictor(args.hidden_size, args.num_factor)
    return FactorVAE(feature_extractor, factor_encoder, factor_decoder, factor_predictor)


def loader(dataset, args, start, end, shuffle=False):
    return init_data_loader(dataset, shuffle=shuffle, step_len=args.seq_len, start=start, end=end)


@torch.no_grad()
def predict(model, dataloader, all_dates, pred_len, device):
    """
    Out-of-sample predictions, one batch per date.

    Uses FactorVAE.prediction (prior factors from the inputs only); the
    forward pass reconstructs returns from the true labels and must not be
    used for predictions.
    Dates follow the original convention: pred_date is the sample date and
    last_date the date pred_len trading days before it.
    """
    model.eval()
    date_pos = {date: i for i, date in enumerate(all_dates)}
    preds, labels, pred_dates, last_dates, tickers = [], [], [], [], []

    for char_with_label, index in dataloader:
        inputs = char_with_label[:, :, :-1].to(device).float()
        label = char_with_label[:, -1, -1].reshape(-1, 1)

        preds.append(model.prediction(inputs).cpu().numpy().reshape(-1, 1))
        labels.append(label.numpy())

        # Each sample's index lists the (datetime, instrument) of every row of
        # its input window; the last one is the sample itself.
        date = pd.Timestamp(index[0][-1][0])
        pred_dates.append(date.strftime('%Y-%m-%d'))
        last_dates.append(pd.Timestamp(all_dates[date_pos[date] - pred_len]).strftime('%Y-%m-%d'))
        tickers.append([window[-1][1] for window in index])

    return preds, labels, pred_dates, last_dates, tickers


def metrics_of(preds, labels):
    from sklearn.metrics import r2_score

    preds_flat, labels_flat = np.concatenate(preds), np.concatenate(labels)
    return {
        'MSE': float(np.mean((preds_flat - labels_flat) ** 2)),
        'MAE': float(np.mean(np.abs(preds_flat - labels_flat))),
        'RMSE': float(np.sqrt(np.mean((preds_flat - labels_flat) ** 2))),
        'R2': float(r2_score(preds_flat, labels_flat)),
    }


def run_train(args):
    set_seed(args.seed)
    model_save_path, metrics_path = create_saving_path(args, "Regression", "FactorVAE", PROJECT_ROOT)
    with open(f'{model_save_path}/config_params.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    dataset = prepare_dataset(args)
    all_dates = dataset.index.get_level_values('datetime').unique().sort_values()

    train_loader = loader(dataset, args, args.start_date, args.end_train_date, shuffle=True)
    valid_loader = loader(dataset, args, args.start_valid_date, args.end_valid_date)
    test_loader = loader(dataset, args, args.start_test_date, args.end_date)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}", flush=True)
    args.device = device

    model = build_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.num_epochs)
    checkpoint_path = os.path.join(model_save_path, 'model.pth')

    if args.wandb:
        import wandb
        wandb.init(project="FactorVAE", config=args, name=f"{args.run_name}")

    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, optimizer, scheduler, args)
        val_loss, val_metrics, _, _ = validate(model, valid_loader, args)
        print(f"Epoch {epoch + 1}/{args.num_epochs}: train_loss={train_loss:.4f}, valid_loss={val_loss:.4f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            with open(f'{metrics_path}/val_metrics_sl{args.seq_len}_pl{args.pred_len}.json', 'w') as f:
                json.dump({k: float(v) for k, v in val_metrics.items()}, f, indent=4)

        if args.wandb:
            wandb.log({"Train Loss": train_loss, "Validation Loss": val_loss,
                       "Learning Rate": scheduler.get_last_lr()[0]})

    if args.wandb:
        wandb.log({"Best Validation Loss": best_val_loss})
        wandb.finish()

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    preds, labels, pred_dates, last_dates, tickers = predict(model, test_loader, all_dates, args.pred_len, device)
    metrics, results_pkl, metrics_json = save_results(
        args, metrics_of(preds, labels), preds, labels, pred_dates, last_dates, tickers, metrics_path
    )
    return {"model_path": checkpoint_path, "metrics": metrics,
            "results_path": results_pkl, "metrics_path": metrics_json}


def build_parser():
    parser = argparse.ArgumentParser(description='Train a FactorVAE model on stock data')

    parser.add_argument('--data_path', type=str, default='./Evaluation/data', help='path to the dataset')
    parser.add_argument('--universe', type=str, default='sp500')
    parser.add_argument('--model_name', type=str, default='FactorVAE')

    parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')

    parser.add_argument('--num_latent', type=int, default=157, help='number of variables')
    parser.add_argument('--num_portfolio', type=int, default=128, help='number of stocks')

    parser.add_argument('--seq_len', type=int, default=10, help='sequence length')
    parser.add_argument('--pred_len', type=int, default=5, help='prediction length')
    parser.add_argument('--num_factor', type=int, default=96, help='number of factors')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')

    parser.add_argument('--start_date', type=str, default='2018-01-01', help='start time')
    parser.add_argument('--end_train_date', type=str, default='2021-12-31', help='fit end time')
    parser.add_argument('--start_valid_date', type=str, default='2022-01-01', help='validation start time')
    parser.add_argument('--end_valid_date', type=str, default='2022-12-31', help='validation end time')
    parser.add_argument('--start_test_date', type=str, default='2023-01-01', help='test start time')
    parser.add_argument('--end_date', type=str, default='2023-12-31', help='end time')

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--run_name', type=str, default='VAE-Revision2', help='name of the run')
    parser.add_argument('--wandb', action='store_true', help='whether to use wandb')
    return parser


if __name__ == '__main__':
    output = run_train(build_parser().parse_args())
    emit_result(output)
