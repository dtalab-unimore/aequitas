"""Train and evaluate the Dynamic Graph Diffusion Neural Network."""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from torch.nn import functional as F

from .dataset_gen import CustomDataset, DatasetConfig
from .model.dgdnn import DGDNN
from utils.filter_constituents import filter_constituents_by_date
from utils.load_configuation import create_saving_path
from utils.logger import emit_result
from utils.save_results import save_results

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class ExperimentConfig:
    root: Path
    destination: Path
    market: str
    train_range: Tuple[str, str]
    val_range: Tuple[str, str]
    test_range: Tuple[str, str]
    window: int = 19
    pred_len: int = 1
    fast_approximation: bool = False
    layers: int = 6
    expansion_step: int = 7
    num_heads: int = 2
    embedding_hidden: int = 1024
    embedding_output: int = 256
    raw_feature_size: int = 64
    classes: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 1.5e-5
    epochs: int = 1200


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_df(df):
    ticker_dict = {
        ticker: frame.set_index("date").sort_index().drop(columns=["instrument"])
        for ticker, frame in df.groupby("instrument")
    }
    return ticker_dict


def prepare_market_data(args):
    df = pd.read_csv(f"{args.data_path}/{args.universe}/{args.universe}.csv")
    trained_tickers = getattr(args, "trained_tickers", None)
    if trained_tickers:
        requested_tickers = list(trained_tickers)
    else:
        constituents = pd.read_csv(f"{args.data_path}/constituents/eodhd/{args.universe}.csv")
        requested_tickers = filter_constituents_by_date(
            constituents, args.start_test_date
        )["EODHD"].tolist()
    df = df[df["instrument"].isin(requested_tickers)]
    df = df[["open", "close", "high", "low", "volume", "instrument", "date"]]

    periods = [
        (args.start_date, args.end_train_date),
        (args.start_valid_date, args.end_valid_date),
        (args.start_test_date, args.end_date),
    ]
    valid = set(requested_tickers)
    minimum_rows = args.seq_len + args.pred_len
    for start, end in periods:
        frame = df[(df["date"] >= start) & (df["date"] <= end)]
        counts = frame.groupby("instrument").size()
        valid &= set(counts[counts > minimum_rows].index)
    ordered_tickers = [ticker for ticker in requested_tickers if ticker in valid]
    if not ordered_tickers:
        raise ValueError("No ticker has enough data in all DGDNN periods.")
    if trained_tickers and ordered_tickers != list(trained_tickers):
        raise ValueError("The requested test period does not contain every ticker used to train DGDNN.")
    return preprocess_df(df[df["instrument"].isin(ordered_tickers)]), ordered_tickers


def build_experiment(args):
    return ExperimentConfig(
        root=Path(args.data_path) / args.universe,
        destination=CURRENT_DIR / "preprocessed" / f"DGDNN_{args.job_id}" / args.universe,
        window=args.seq_len,
        pred_len=args.pred_len,
        market=args.universe,
        train_range=(args.start_date, args.end_train_date),
        val_range=(args.start_valid_date, args.end_valid_date),
        test_range=(args.start_test_date, args.end_date),
        epochs=args.n_epochs,
        expansion_step=args.expansion_step,
    )


def build_dataset(config, mode, date_range, ticker_dict):
    dataset_config = DatasetConfig(
        root=str(config.root), dest=str(config.destination), market=config.market,
        tickers=list(ticker_dict.keys()), start=date_range[0], end=date_range[1],
        window=config.window, pred_len=config.pred_len, mode=mode,
        fast_approx=config.fast_approximation,
    )
    return CustomDataset(ticker_dict, dataset_config)


def build_model(config, num_nodes, timestamp):
    diffusion_size = [timestamp * 5, 64, 128, 256, 256, 256, 128]
    embedding_size = [128, 384, 512, 512, 512, 384]
    emb_output = config.embedding_output
    raw_feature_size = config.raw_feature_size
    if config.num_heads != 2:
        scale = config.num_heads / 2.0
        emb_output = int(round(emb_output * scale))
        raw_feature_size = int(round(raw_feature_size * scale))
        diffusion_size = [diffusion_size[0]] + [int(round(value * scale)) for value in diffusion_size[1:]]
        embedding_size = [int(round(value * scale)) for value in embedding_size]
    return DGDNN(
        diffusion_size, embedding_size, config.embedding_hidden, emb_output,
        raw_feature_size, config.classes, config.layers, num_nodes,
        config.expansion_step, config.num_heads, active=[True] * config.layers,
    ).to(DEVICE)


def train_epoch(model, dataset, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for sample in dataset:
        features = sample["X"].to(DEVICE)
        adjacency = sample["A"].to(DEVICE)
        target = sample["Y"].to(DEVICE).long()
        optimizer.zero_grad()
        logits = model(features, adjacency)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        correct += int((logits.argmax(dim=1) == target).sum())
        total += target.size(0)
    return total_loss, correct / total if total else 0.0


def get_metrics(preds, labels):
    predicted_classes = np.asarray(preds).argmax(axis=-1).reshape(-1)
    true_classes = np.asarray(labels).reshape(-1)
    return {
        "F1": f1_score(true_classes, predicted_classes, average="macro", zero_division=0),
        "Accuracy": accuracy_score(true_classes, predicted_classes),
        "Precision": precision_score(true_classes, predicted_classes, average="macro", zero_division=0),
        "Recall": recall_score(true_classes, predicted_classes, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(true_classes, predicted_classes),
    }


def evaluate(model, dataset):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for sample in dataset:
            logits = model(sample["X"].to(DEVICE), sample["A"].to(DEVICE))
            predictions.append(logits.cpu().numpy())
            targets.append(np.expand_dims(sample["Y"].cpu().numpy(), axis=1))
    if not predictions:
        raise ValueError("The selected DGDNN period produced no evaluation windows.")
    return get_metrics(predictions, targets), predictions, targets


def result_dates(dataset, pred_len):
    count = len(dataset)
    last_dates = dataset.dates[dataset.window - 1: dataset.window - 1 + count]
    pred_dates = dataset.dates[dataset.window + pred_len - 1: dataset.window + pred_len - 1 + count]
    return [str(value.date()) for value in last_dates], [str(value.date()) for value in pred_dates]


def run_train(args):
    set_seed(args.seed)
    model_save_path, metrics_path = create_saving_path(args, "Classification", "DGDNN", str(PROJECT_ROOT))
    ticker_dict, tickers = prepare_market_data(args)
    args.trained_tickers = tickers
    with open(os.path.join(model_save_path, "config_params.json"), "w") as handle:
        json.dump(vars(args), handle, indent=4)

    experiment = build_experiment(args)
    train_dataset = build_dataset(experiment, "train", experiment.train_range, ticker_dict)
    valid_dataset = build_dataset(experiment, "validation", experiment.val_range, ticker_dict)
    test_dataset = build_dataset(experiment, "test", experiment.test_range, ticker_dict)
    model = build_model(experiment, len(tickers), experiment.window)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=experiment.learning_rate, weight_decay=experiment.weight_decay
    )
    for epoch in range(1, experiment.epochs + 1):
        loss, accuracy = train_epoch(model, train_dataset, optimizer)
        print(f"Epoch {epoch:04d}: loss={loss:.4f}, acc={accuracy:.4f}", flush=True)

    checkpoint = os.path.join(model_save_path, "model.pth")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": experiment.epochs}, checkpoint)
    valid_metrics, _, _ = evaluate(model, valid_dataset)
    with open(os.path.join(metrics_path, f"val_metrics_sl{args.seq_len}_pl{args.pred_len}.json"), "w") as handle:
        json.dump({key: float(value) for key, value in valid_metrics.items()}, handle, indent=4)
    test_metrics, predictions, labels = evaluate(model, test_dataset)
    last_dates, pred_dates = result_dates(test_dataset, args.pred_len)
    metrics, results_pkl, metrics_json = save_results(
        args, test_metrics, predictions, labels, pred_dates, last_dates,
        [tickers] * len(predictions), metrics_path
    )
    return {"model_path": checkpoint, "metrics": metrics,
            "results_path": results_pkl, "metrics_path": metrics_json}


def build_parser():
    parser = ArgumentParser(description="Train the DGDNN classification model")
    parser.add_argument("--data_path", default="./Evaluation/data")
    parser.add_argument("--universe", default="sp500")
    parser.add_argument("--model_name", default="DGDNN")
    parser.add_argument("--start_date", default="2015-01-01")
    parser.add_argument("--end_train_date", default="2018-12-31")
    parser.add_argument("--start_valid_date", default="2019-01-01")
    parser.add_argument("--end_valid_date", default="2019-12-31")
    parser.add_argument("--start_test_date", default="2020-01-01")
    parser.add_argument("--end_date", default="2020-12-31")
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--pred_len", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_epochs", type=int, default=1200)
    parser.add_argument("--expansion_step", type=int, default=7)
    parser.add_argument("--job_id", type=int, default=0)
    return parser


if __name__ == "__main__":
    output = run_train(build_parser().parse_args())
    emit_result(output)
