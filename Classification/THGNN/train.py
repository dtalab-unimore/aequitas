import json
import os
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from utils.save_results import save_results
from .data_loader import CustomDataset
from .trainer.trainer import train_epoch, bce_loss, eval_model
from .model.Thgnn import StockHeteGAT
from utils.filter_constituents import filter_constituents_by_date
from utils.load_configuation import create_saving_path
from utils.logger import emit_result

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

def select_valid_ticker(df, corr_matr, start_date, end_date):
    tickers_before = df[df['date'] < start_date]['instrument'].drop_duplicates().reset_index(drop=True)
    tickers_after = df[df['date'] > end_date]['instrument'].drop_duplicates().reset_index(drop=True)
    tickers = pd.merge(tickers_before, tickers_after, on='instrument')['instrument'].to_list()
    dates = df['date'].drop_duplicates().reset_index(drop=True)
    df = df[df['instrument'].isin(tickers)]
    mask = np.array([t in tickers for t in corr_matr['tickers']])
    corr_matr['adj_matrix'] = corr_matr['adj_matrix'][:, :, mask][:, :, :, mask]
    corr_matr['tickers'] = [t for t in corr_matr['tickers'] if t in tickers]
    return df, corr_matr, tickers, dates

def extract_labels(df, args):
    df['Label'] = df.groupby('instrument')['adj_close'].transform(lambda x: ((x.shift(-args.pred_len) - x) / x) * 100)
    df = df.drop(columns=['adj_close'])
    return df

def calcultate_matrics(preds, labels):
    preds = np.concatenate(preds, axis=0).reshape(-1)
    preds = np.where(preds > 0.5, 1, 0)
    labels = np.concatenate(labels, axis=0).reshape(-1)

    metrics = {
        'F1': f1_score(labels, preds, average='macro'),
        'Accuracy': accuracy_score(labels, preds),
        'Precision': precision_score(labels, preds),
        'Recall': recall_score(labels, preds),
        'MCC': matthews_corrcoef(labels, preds)
    }
    return metrics

def train(args, model, train_loader, optimizer, bce_loss, default_scheduler, valid_loader, metrics_path, model_save_path):
    print('Start Training...')
    for epoch in range(args.n_epochs):
        train_loss = train_epoch(args=args, model=model, dataset_train=train_loader,
                                 optimizer=optimizer, scheduler=default_scheduler, loss_fcn=bce_loss)
        if epoch % args.epochs_eval == 0:
            eval_loss, val_preds, val_labels = eval_model(args=args, model=model, dataset_eval=valid_loader,
                                                          loss_fcn=bce_loss)
            print('Epoch: {}/{}, train loss: {:.6f}, val loss: {:.6f}'.format(epoch + 1, args.n_epochs, train_loss,
                                                                              eval_loss))

            val_metrics = calcultate_matrics(val_preds, val_labels)
            val_metrics = {k: float(v) for k, v in val_metrics.items()}

            with open(f'{metrics_path}/val_metrics_sl{args.seq_len}_pl{args.pred_len}.json', 'w') as f:
                json.dump(val_metrics, f, indent=4)

        else:
            print('Epoch: {}/{}, train loss: {:.6f}'.format(epoch + 1, args.n_epochs, train_loss))
        if (epoch + 1) % args.n_epochs == 0:
            print("save model!")
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
            torch.save(state, os.path.join(model_save_path, "model.pth"))

        print(f'Epoch {epoch + 1}/{args.n_epochs} completed.')

if __name__=='__main__':
    args = ArgumentParser()

    # Dataset Parameters
    args.add_argument('--data_path', type=str, default='./Evaluation/data')
    args.add_argument('--universe', type=str, default='sx5e')
    args.add_argument('--model_name', type=str, default='THGNN')
    args.add_argument('--seed', type=int, default=42)

    # Prediction Parameters
    args.add_argument('--seq_len', type=int, default=20)
    args.add_argument('--pred_len', type=int, default=5)
    args.add_argument('--start_date', type=str, default='2020-01-01')
    args.add_argument('--end_train_date', type=str, default='2020-12-31')
    args.add_argument('--start_valid_date', type=str, default='2021-01-01')
    args.add_argument('--end_valid_date', type=str, default='2022-12-31')
    args.add_argument('--start_test_date', type=str, default='2023-01-01')
    args.add_argument('--end_date', type=str, default='2023-12-31')

    # Model Hyperparameters
    args.add_argument('--n_epochs', type=int, default=60)
    args.add_argument('--epochs_eval', type=int, default=10)
    args.add_argument('--lr', type=float, default=0.0002)
    args.add_argument('--gamma', type=float, default=0.3)
    args.add_argument('--hidden_dim', type=int, default=128)
    args.add_argument('--num_heads', type=int, default=8)
    args.add_argument('--out_features', type=int, default=32)
    args.add_argument('--batch_size', type=int, default=1)

    args = args.parse_args()

    model_save_path, metrics_path = create_saving_path(args, 'Classification', 'THGNN', PROJECT_ROOT)

    with open(f'{model_save_path}/config_params.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    train_dataset = CustomDataset(
        dataset=dataset.copy(),
        correlation_matrix=corr_matrix.copy(),
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        start_date=args.start_date,
        end_date=args.end_train_date
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, collate_fn=lambda x: x)

    valid_dataset = CustomDataset(
        dataset=dataset.copy(),
        correlation_matrix=corr_matrix.copy(),
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        start_date=args.start_valid_date,
        end_date=args.end_valid_date,
        period='valid'
    )
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, pin_memory=True)

    test_dataset = CustomDataset(
        dataset=dataset.copy(),
        correlation_matrix=corr_matrix.copy(),
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        start_date=args.start_test_date,
        end_date=args.end_date,
        period='test'
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StockHeteGAT(hidden_dim=args.hidden_dim, num_heads=args.num_heads,
                                  out_features=args.out_features).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    cold_scheduler = StepLR(optimizer=optimizer, step_size=5000, gamma=0.9, last_epoch=-1)
    default_scheduler = cold_scheduler

    train(args, model, train_loader, optimizer, bce_loss, default_scheduler, valid_loader, metrics_path, model_save_path)

    print('Start Testing...')

    _, preds, labels = eval_model(args=args, model=model, dataset_eval=test_loader, loss_fcn=bce_loss)

    test_metrics = calcultate_matrics(preds, labels)
    preds = [np.expand_dims(p, axis=1) for p in preds]
    labels = [np.expand_dims(label, axis=1) for label in labels]

    metrics, results_pkl, metrics_json = save_results(
        args,
        test_metrics,
        preds,
        labels,
        test_dataset.pred_dates,
        test_dataset.last_seq_dates,
        [tickers] * len(preds),
        metrics_path
    )

    print(f"Test Metrics: {test_metrics}")

    emit_result({
        "model_path": os.path.join(model_save_path, "model.pth"),
        "metrics": metrics,
        "results_path": results_pkl,
        "metrics_path": metrics_json,
    })




