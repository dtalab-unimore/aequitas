import json
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from torch import optim

from .load_data import (
    adj_to_dense_matrix,
    build_news_array,
    encode_news,
    filter_data_by_available_news,
    filter_data_by_available_news_tickers,
    filter_nan_windows,
    load_adj,
    load_prices,
    train_valid_test_split,
    windows_extraction,
)
from .model.model import GAT
from utils.load_configuation import create_saving_path
from utils.logger import emit_result
from utils.save_results import save_results

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))


def get_metrics(preds, labels):
    predicted_classes = np.asarray(preds).argmax(axis=-1).reshape(-1)
    true_classes = np.asarray(labels).reshape(-1)
    mask = ~np.isnan(true_classes)
    predicted_classes = predicted_classes[mask]
    true_classes = true_classes[mask]
    return {
        "F1": f1_score(true_classes, predicted_classes, average="macro", zero_division=0),
        "Accuracy": accuracy_score(true_classes, predicted_classes),
        "Precision": precision_score(true_classes, predicted_classes, average="macro", zero_division=0),
        "Recall": recall_score(true_classes, predicted_classes, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(true_classes, predicted_classes),
    }


def train_batch(model, prices, news, adjacency, device):
    time_index = np.random.randint(prices.shape[1])
    text = news[:, time_index].to(device)
    price = prices[:, time_index, :, :-2].to(device)
    labels = prices[:, time_index, -1, -2].long().to(device)
    return model(text, price, adjacency), labels


def eval_model(model, prices, news, adjacency, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for time_index in range(prices.shape[1]):
            text = news[:, time_index].to(device)
            price = prices[:, time_index, :, :-2].to(device)
            predictions.append(model(text, price, adjacency).cpu().numpy())
    labels = prices[:, :, -1, -1].cpu().numpy().swapaxes(0, 1)
    return get_metrics(predictions, labels), predictions, labels


def prepare_data(args):
    os.makedirs(f"{args.data_path}_news/{args.universe}", exist_ok=True)
    prices_array, tickers, dates = load_prices(args)
    if not tickers:
        raise ValueError("No tickers found in the specified universe and date range.")
    adjacency, ordered_tickers = load_adj(tickers, args)
    news_dict = encode_news(tickers, dates, args)
    news_list, news_tickers = build_news_array(news_dict, dates, tickers, args, num_news=args.n_news)
    prices_array, adjacency, tickers, ordered_tickers = filter_data_by_available_news_tickers(
        news_tickers, tickers, prices_array, adjacency, ordered_tickers
    )
    split_prices, split_news, split_dates = train_valid_test_split(
        prices_array, news_list, tickers, dates, args
    )
    split_prices, split_news, last_dates, pred_dates = windows_extraction(
        split_prices, split_news, split_dates, args
    )
    split_prices, split_news, last_dates, pred_dates, tickers, ordered_tickers, adjacency = (
        filter_data_by_available_news(
            split_prices, split_news, last_dates, pred_dates,
            list(tickers), ordered_tickers, adjacency
        )
    )
    split_prices, split_news = filter_nan_windows(split_prices, split_news)
    if any(values.shape[1] == 0 for values in split_news):
        raise ValueError("No news data is available after MAN-SF preprocessing.")
    return split_prices, split_news, last_dates, pred_dates, list(tickers), adj_to_dense_matrix(adjacency)


def create_model(args, stock_count, device):
    return GAT(
        nfeat=64, nhid=args.hidden, nclass=2, dropout=args.dropout,
        nheads=args.nb_heads, alpha=args.alpha, stock_num=stock_count
    ).to(device)


def flatten_dates(date_map):
    if not date_map:
        return []
    values = list(date_map.values())
    common = set(values[0])
    for dates in values[1:]:
        common &= set(dates)
    return sorted(common)


def run_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model_save_path, metrics_path = create_saving_path(args, "Classification", "MAN-SF", PROJECT_ROOT)
    prices, news, last_dates, pred_dates, tickers, adjacency = prepare_data(args)
    args.trained_tickers = tickers
    with open(os.path.join(model_save_path, "config_params.json"), "w") as handle:
        json.dump(vars(args), handle, indent=4)

    adjacency = adjacency.to(device)
    model = create_model(args, len(tickers), device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0], device=device))
    best_mcc = float("-inf")
    stop = 0
    checkpoint_path = os.path.join(model_save_path, "model.pth")
    best_valid_metrics = None

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output, labels = train_batch(model, prices[0], news[0], adjacency, device)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        valid_metrics, _, _ = eval_model(model, prices[1], news[1], adjacency, device)
        print(
            f"Epoch {epoch + 1}/{args.epochs}: loss={loss.item():.6f}, "
            f"val_mcc={valid_metrics['MCC']:.6f}", flush=True
        )
        if valid_metrics["MCC"] > best_mcc:
            best_mcc = valid_metrics["MCC"]
            best_valid_metrics = valid_metrics
            stop = 0
            torch.save({"model": model.state_dict(), "epoch": epoch + 1}, checkpoint_path)
        else:
            stop += 1
        if stop >= args.early_stopping:
            break

    if best_valid_metrics is None:
        raise RuntimeError("MAN-SF training did not produce a valid checkpoint.")
    with open(os.path.join(metrics_path, f"val_metrics_sl{args.seq_len}_pl{args.pred_len}.json"), "w") as handle:
        json.dump({key: float(value) for key, value in best_valid_metrics.items()}, handle, indent=4)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics, predictions, labels = eval_model(model, prices[2], news[2], adjacency, device)
    pred_date_values = flatten_dates(pred_dates)
    last_date_values = flatten_dates(last_dates)
    metrics, results_pkl, metrics_json = save_results(
        args, test_metrics, predictions, labels, pred_date_values, last_date_values,
        [tickers] * len(predictions), metrics_path
    )
    return {"model_path": checkpoint_path, "metrics": metrics,
            "results_path": results_pkl, "metrics_path": metrics_json}


def build_parser():
    parser = ArgumentParser(description="Train the MAN-SF classification model")
    parser.add_argument("--data_path", default="./Evaluation/data")
    parser.add_argument("--universe", default="sp500")
    parser.add_argument("--model_name", default="MAN-SF")
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--start_date", default="2019-01-01")
    parser.add_argument("--end_train_date", default="2021-12-31")
    parser.add_argument("--start_valid_date", default="2022-01-01")
    parser.add_argument("--end_valid_date", default="2022-12-31")
    parser.add_argument("--start_test_date", default="2023-01-01")
    parser.add_argument("--end_date", default="2023-12-31")
    parser.add_argument("--n_news", type=int, default=5)
    parser.add_argument("--sparse", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--nb_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.38)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--early_stopping", type=int, default=50)
    return parser


if __name__ == "__main__":
    output = run_train(build_parser().parse_args())
    emit_result(output)
