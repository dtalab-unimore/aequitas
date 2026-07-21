import json
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch

from .evaluator import evaluate_regression
from .load_datasets import load_dataset
from .model import StockMixer, get_loss
from utils.load_configuation import create_saving_path
from utils.save_results import save_results

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args, stock_num, feature_num, device):
    if args.seq_len < 2:
        raise ValueError("StockMixer requires seq_len >= 2.")
    return StockMixer(
        stocks=stock_num, time_steps=args.seq_len, channels=feature_num,
        market=args.market_num, scale=max(1, args.seq_len // 2),
    ).to(device)


def get_batch(offset, args, eod_data, mask_data, price_data, gt_data):
    mask_batch = mask_data[:, offset + args.seq_len - 1]
    return (
        eod_data[:, offset: offset + args.seq_len, :],
        np.expand_dims(mask_batch, axis=1),
        np.expand_dims(price_data[:, offset + args.seq_len - 1], axis=1),
        np.expand_dims(gt_data[:, offset + args.seq_len - 1], axis=1),
    )


def evaluate_range(model, args, arrays, device, start_index, end_index):
    eod_data, gt_data, price_data, mask_data = arrays
    predictions, labels, masks = [], [], []
    losses, reg_losses, rank_losses = [], [], []
    model.eval()
    stop = min(end_index, eod_data.shape[1] - args.seq_len - args.pred_len + 1)
    with torch.no_grad():
        for offset in range(start_index, stop):
            data, mask, price, ground_truth = map(
                lambda value: torch.as_tensor(value, dtype=torch.float32, device=device),
                get_batch(offset, args, eod_data, mask_data, price_data, gt_data),
            )
            prediction = model(data)
            loss, reg_loss, rank_loss, return_ratio = get_loss(
                prediction, ground_truth, price, mask, eod_data.shape[0], args.alpha
            )
            losses.append(float(loss.item()))
            reg_losses.append(float(reg_loss.item()))
            rank_losses.append(float(rank_loss.item()))
            predictions.append(return_ratio.cpu().numpy())
            labels.append(ground_truth.cpu().numpy())
            masks.append(mask.cpu().numpy())
    if not predictions:
        raise ValueError("The selected StockMixer period produced no evaluation windows.")
    pred_matrix = np.asarray(predictions).squeeze(-1).T
    label_matrix = np.asarray(labels).squeeze(-1).T
    mask_matrix = np.asarray(masks).squeeze(-1).T
    metrics = evaluate_regression(pred_matrix, label_matrix, mask_matrix)
    return (
        float(np.mean(losses)), float(np.mean(reg_losses)), float(np.mean(rank_losses)),
        metrics, predictions, labels,
    )


def result_dates(dates, tickers, start_index, args):
    last_dates = dates[start_index + args.seq_len - 1: -args.pred_len].astype(str).tolist()
    pred_dates = dates[start_index + args.seq_len + args.pred_len - 1:].astype(str).tolist()
    return pred_dates, last_dates, [tickers] * len(pred_dates)


def run_train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path, metrics_path = create_saving_path(
        args, "Regression", "StockMixer", PROJECT_ROOT
    )
    eod_data, gt_data, price_data, valid_index, test_index, dates, tickers = load_dataset(args)
    args.trained_tickers = list(tickers)
    with open(os.path.join(model_save_path, "config_params.json"), "w") as handle:
        json.dump(vars(args), handle, indent=4)

    stock_num, feature_num = eod_data.shape[0], eod_data.shape[-1]
    mask_data = np.isfinite(gt_data).astype(np.float32)
    gt_data = np.nan_to_num(gt_data, nan=0.0)
    arrays = (eod_data, gt_data, price_data, mask_data)
    model = build_model(args, stock_num, feature_num, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_valid_loss = float("inf")
    best_payload = None
    checkpoint_path = os.path.join(model_save_path, "model.pth")

    train_offsets = np.arange(valid_index, dtype=int)
    for epoch in range(args.epochs):
        model.train()
        np.random.shuffle(train_offsets)
        losses = []
        for offset in train_offsets:
            data, mask, price, ground_truth = map(
                lambda value: torch.as_tensor(value, dtype=torch.float32, device=device),
                get_batch(offset, args, eod_data, mask_data, price_data, gt_data),
            )
            optimizer.zero_grad()
            prediction = model(data)
            loss, _, _, _ = get_loss(
                prediction, ground_truth, price, mask, stock_num, args.alpha
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        valid_loss, _, _, valid_metrics, _, _ = evaluate_range(
            model, args, arrays, device, valid_index, test_index
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs}: train_loss={np.mean(losses):.6f}, "
            f"valid_loss={valid_loss:.6f}", flush=True
        )
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            test_output = evaluate_range(model, args, arrays, device, test_index, eod_data.shape[1])
            best_payload = (valid_metrics, test_output[3], test_output[4], test_output[5])
            torch.save({"model": model.state_dict(), "epoch": epoch + 1}, checkpoint_path)

    if best_payload is None:
        raise RuntimeError("StockMixer training did not produce a checkpoint.")
    valid_metrics, test_metrics, predictions, labels = best_payload
    with open(os.path.join(metrics_path, f"val_metrics_sl{args.seq_len}_pl{args.pred_len}.json"), "w") as handle:
        json.dump({key: float(value) for key, value in valid_metrics.items()}, handle, indent=4)
    pred_dates, last_dates, tickers_to_date = result_dates(dates, tickers, test_index, args)
    metrics, results_pkl, metrics_json = save_results(
        args, test_metrics, predictions, labels, pred_dates, last_dates,
        tickers_to_date, metrics_path
    )
    return {"model_path": checkpoint_path, "metrics": metrics,
            "results_path": results_pkl, "metrics_path": metrics_json}


def build_parser():
    parser = ArgumentParser(description="Train the StockMixer regression model")
    parser.add_argument("--data_path", default="./Evaluation/data")
    parser.add_argument("--universe", default="nasdaq100")
    parser.add_argument("--model_name", default="StockMixer")
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--start_date", default="2018-01-01")
    parser.add_argument("--end_train_date", default="2021-12-31")
    parser.add_argument("--start_valid_date", default="2022-01-01")
    parser.add_argument("--end_valid_date", default="2022-12-31")
    parser.add_argument("--start_test_date", default="2023-01-01")
    parser.add_argument("--end_date", default="2023-12-31")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--activation", default="GELU")
    parser.add_argument("--market_num", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    output = run_train(build_parser().parse_args())
    print("__TRAIN_JSON_START__", flush=True)
    print(json.dumps(output, ensure_ascii=False), flush=True)
    print("__TRAIN_JSON_END__", flush=True)
