import argparse
import collections
import copy
import datetime
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import r2_score
from tqdm import tqdm

from .dataloader import DataLoader, RobustZScoreNormalization
from .factormodel import FACTORMODEL
from .utils import cal_beta, metric_fn, mse, select_valid_ticker
from utils.filter_constituents import filter_constituents_by_date
from utils.load_configuation import create_saving_path
from utils.save_results import save_results

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def average_params(params_list):
    if not isinstance(params_list, (tuple, list, collections.deque)) or not params_list:
        raise ValueError("params_list must contain at least one model state")
    if len(params_list) == 1:
        return params_list[0]
    averaged = collections.OrderedDict()
    for params in params_list:
        for key, value in params.items():
            averaged[key] = averaged.get(key, 0) + value / len(params_list)
    return averaged


def loss_fn(prediction, label, args):
    del args
    mask = ~torch.isnan(label)
    return mse(prediction[mask], label[mask])


def pprint(*values):
    timestamp = str(datetime.datetime.utcnow())[:19]
    print(f"[{timestamp}] -", *values, flush=True)


def zscore(values):
    std = values.std()
    if not torch.isfinite(std) or std == 0:
        return values - values.mean()
    return (values - values.mean()).div(std)


def train_epoch(epoch, model, optimizer, train_loader, args, concept_matrix):
    model.train()
    capacity = None
    if args.beta_strategy == "B":
        maximum = torch.tensor(0.50, device=DEVICE)
        capacity = torch.clamp(maximum / args.n_epochs * epoch, 0, maximum)
    for _, selection in tqdm(train_loader.iter_batch(), total=train_loader.batch_length):
        feature, label, market_value, stock_index, _ = train_loader.get(selection)
        if label.shape[0] <= 1:
            continue
        label = zscore(label)
        prediction, vae_losses = model(feature, concept_matrix[stock_index], market_value)
        prediction_loss = loss_fn(prediction, label, args)
        if args.beta_strategy == "B":
            loss = prediction_loss + (
                vae_losses[0] + vae_losses[2] + (vae_losses[1] - capacity).abs()
            ) * 5e-2
        elif args.beta_strategy == "increasing":
            beta = cal_beta(epoch)
            loss = prediction_loss + (
                vae_losses[0] + beta * vae_losses[1] + vae_losses[2]
            ) * 5e-2
        else:
            raise ValueError(f"Unknown beta strategy: {args.beta_strategy}")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
        optimizer.step()


def test_epoch(model, data_loader, args, concept_matrix, prefix="Test"):
    model.eval()
    losses, frames, predictions, labels, stock_indexes, mask = [], [], [], [], [], []
    for _, selection in tqdm(data_loader.iter_daily(), desc=prefix, total=data_loader.daily_length):
        feature, label, market_value, stock_index, index = data_loader.get(selection)
        if label.shape[0] <= 1:
            mask.append(False)
            continue
        label = zscore(label)
        with torch.no_grad():
            prediction, _ = model(feature, concept_matrix[stock_index], market_value)
            loss = loss_fn(prediction, label, args)
        losses.append(float(loss.item()))
        prediction_array = prediction.cpu().numpy()
        label_array = label.cpu().numpy()
        frames.append(pd.DataFrame({"score": prediction_array, "label": label_array}, index=index))
        predictions.append(prediction_array)
        labels.append(label_array)
        stock_indexes.append(stock_index.cpu().numpy())
        mask.append(True)
    if not predictions:
        raise ValueError(f"The {prefix} period produced no DiscoverPLF daily batches.")
    prediction_frame = pd.concat(frames, axis=0)
    _, _, ic, _ = metric_fn(prediction_frame)
    return predictions, labels, stock_indexes, float(np.mean(losses)), float(ic), mask


def create_loaders(args):
    alpha = pd.read_csv(f"{args.data_path}/{args.universe}/{args.universe}_alpha360.csv")
    trained_tickers = getattr(args, "trained_tickers", None)
    if trained_tickers:
        selected_tickers = list(trained_tickers)
    else:
        constituents = pd.read_csv(f"{args.data_path}/constituents/eodhd/{args.universe}.csv")
        selected_tickers = filter_constituents_by_date(
            constituents, args.start_test_date
        )["EODHD"].tolist()
    alpha = alpha[alpha["instrument"].isin(selected_tickers)]

    close = pd.read_csv(
        f"{args.data_path}/{args.universe}/{args.universe}.csv",
        usecols=["date", "instrument", "adj_close"],
    ).sort_values(["instrument", "date"])
    close["Label"] = close.groupby("instrument")["adj_close"].transform(
        lambda values: (values.shift(-args.pred_len) - values) / values
    )
    alpha = alpha.rename(columns={"date": "datetime"})
    close = close.rename(columns={"date": "datetime"})
    alpha = alpha.merge(close[["datetime", "instrument", "Label"]], on=["datetime", "instrument"], how="left")
    alpha = select_valid_ticker(alpha, args.start_date, args.end_train_date)
    normalizer = RobustZScoreNormalization(
        alpha[(alpha["datetime"] >= args.start_date) & (alpha["datetime"] <= args.end_train_date)]
    )
    alpha = alpha.groupby("instrument", group_keys=False).apply(lambda frame: frame.iloc[:-args.pred_len])
    alpha = normalizer.robust_zscore(alpha).fillna(0)
    alpha = alpha[(alpha["datetime"] >= args.start_date) & (alpha["datetime"] <= args.end_date)]
    alpha = alpha.set_index(["datetime", "instrument"])

    concept_data = np.load(f"{args.data_path}/{args.universe}/{args.universe}_inc_matrix.npz")
    matrix_tickers = list(concept_data["tickers"])
    valid_matrix_tickers = set(matrix_tickers)
    alpha = alpha[alpha.index.get_level_values("instrument").isin(valid_matrix_tickers)]
    used_tickers = sorted(alpha.index.get_level_values("instrument").unique())
    if not used_tickers:
        raise ValueError("No DiscoverPLF ticker is shared by Alpha360 and the incidence matrix.")
    if trained_tickers and used_tickers != sorted(trained_tickers):
        raise ValueError("The test period does not contain every ticker used to train DiscoverPLF.")
    stock_index = {str(ticker): index for index, ticker in enumerate(matrix_tickers)}
    alpha["stock_index"] = alpha.index.get_level_values("instrument").map(stock_index).astype(int)

    market_value = pd.read_csv(f"{args.data_path}/{args.universe}/{args.universe}_market_cap.csv")
    market_value = market_value.rename(columns={"date": "datetime"}).set_index(["datetime", "instrument"])
    market_value = market_value[["market_cap"]] / 1_000_000_000
    all_dates = pd.to_datetime(alpha.index.get_level_values("datetime").unique().sort_values())
    valid_index = all_dates.searchsorted(pd.to_datetime(args.start_valid_date))
    test_index = all_dates.searchsorted(pd.to_datetime(args.start_test_date))
    valid_context = all_dates[max(0, valid_index - args.pred_len)].strftime("%Y-%m-%d")
    test_context = all_dates[max(0, test_index - args.pred_len)].strftime("%Y-%m-%d")

    def prepare_frame(frame):
        frame = frame.join(market_value, how="inner").rename(columns={"market_cap": "market_value"})
        frame["market_value"] = frame["market_value"].fillna(frame["market_value"].mean())
        return frame.sort_index(level=["datetime", "instrument"])

    train_frame = prepare_frame(alpha[alpha.index.get_level_values("datetime") <= args.end_train_date])
    train_frame = train_frame.groupby(level="instrument", group_keys=False).apply(
        lambda frame: frame.iloc[:-args.pred_len]
    ).sort_index(level=["datetime", "instrument"])
    valid_frame = prepare_frame(alpha[
        (alpha.index.get_level_values("datetime") >= valid_context)
        & (alpha.index.get_level_values("datetime") <= args.end_valid_date)
    ])
    valid_frame = valid_frame.groupby(level="instrument", group_keys=False).apply(
        lambda frame: frame.iloc[:-args.pred_len]
    ).sort_index(level=["datetime", "instrument"])
    test_frame = prepare_frame(alpha[
        (alpha.index.get_level_values("datetime") >= test_context)
        & (alpha.index.get_level_values("datetime") <= args.end_date)
    ])
    unique_test_dates = test_frame.index.get_level_values("datetime").unique().tolist()
    last_dates = unique_test_dates[:-args.pred_len]
    pred_dates = unique_test_dates[args.pred_len:]
    test_frame = test_frame.groupby(level="instrument", group_keys=False).apply(
        lambda frame: frame.iloc[:-args.pred_len]
    ).sort_index(level=["datetime", "instrument"])

    def loader(frame, batch_size=-1):
        return DataLoader(
            frame.iloc[:, :-3], pd.DataFrame(frame["Label"]), frame["market_value"],
            frame["stock_index"], batch_size=batch_size, pin_memory=args.pin_memory,
            device=DEVICE,
        )

    return (
        loader(train_frame, args.batch_size), loader(valid_frame), loader(test_frame),
        concept_data["inc_matrix"], stock_index, last_dates, pred_dates, used_tickers,
    )


def get_metrics(predictions, labels):
    prediction_array = np.concatenate(predictions, axis=0)
    label_array = np.concatenate(labels, axis=0)
    return {
        "MSE": np.mean((prediction_array - label_array) ** 2),
        "MAE": np.mean(np.abs(prediction_array - label_array)),
        "RMSE": np.sqrt(np.mean((prediction_array - label_array) ** 2)),
        "R2": r2_score(label_array, prediction_array),
    }


def checkpoint_selection(valid_ic, valid_loss):
    """Prefer finite IC; fall back to the lowest finite validation loss."""
    if np.isfinite(valid_ic):
        value = float(valid_ic)
        return (1, value), "validation IC", value
    if np.isfinite(valid_loss):
        value = float(valid_loss)
        return (0, -value), "validation loss", value
    return None


def create_model(args):
    return FACTORMODEL(
        d_feat=args.d_feat, hidden_size=args.hidden_size, num_layers=args.num_layers,
        dilations=[int(value) for value in args.stride.split("+")],
        num_days=int(args.num_days), K=args.K,
    ).to(DEVICE)


def run_train(args):
    if args.n_epochs < 1:
        raise ValueError("DiscoverPLF requires n_epochs >= 1.")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model_save_path, metrics_path = create_saving_path(
        args, "Regression", "DiscoverPLF", PROJECT_ROOT
    )
    loaders = create_loaders(args)
    train_loader, valid_loader, test_loader, concept_matrix, stock_index, last_dates, pred_dates, used_tickers = loaders
    args.trained_tickers = list(used_tickers)
    with open(os.path.join(model_save_path, "config_params.json"), "w") as handle:
        json.dump(vars(args), handle, indent=4)
    inverse_stock_index = {value: key for key, value in stock_index.items()}
    concept_matrix = torch.as_tensor(concept_matrix, dtype=torch.float32, device=DEVICE)
    model = create_model(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_selection, best_epoch, stop_round, best_param = None, -1, 0, None
    best_result = None
    parameter_history = collections.deque(maxlen=args.smooth_steps)
    evaluation_start_epoch = min(max(0, args.warmup_epochs), args.n_epochs - 1)

    for epoch in range(args.n_epochs):
        pprint("Epoch:", epoch + 1)
        train_epoch(epoch, model, optimizer, train_loader, args, concept_matrix)
        current_params = copy.deepcopy(model.state_dict())
        parameter_history.append(current_params)
        averaged_params = average_params(parameter_history)
        if epoch < evaluation_start_epoch:
            continue
        model.load_state_dict(averaged_params)
        valid_output = test_epoch(model, valid_loader, args, concept_matrix, prefix="Valid")
        test_output = test_epoch(model, test_loader, args, concept_matrix, prefix="Test")
        model.load_state_dict(current_params)
        valid_predictions, valid_labels, _, valid_loss, valid_score, _ = valid_output
        test_predictions, test_labels, test_stock_indexes, test_loss, _, test_mask = test_output
        pprint(f"Valid loss: {valid_loss:.4f}, Test loss: {test_loss:.4f}")
        selection = checkpoint_selection(valid_score, valid_loss)
        if selection is not None and (
            best_selection is None or selection[0] > best_selection[0]
        ):
            best_selection, best_epoch, stop_round = selection, epoch, 0
            pprint(f"Checkpoint updated using {selection[1]}: {selection[2]:.6f}")
            best_param = copy.deepcopy(averaged_params)
            valid_metrics = get_metrics(valid_predictions, valid_labels)
            test_metrics = get_metrics(test_predictions, test_labels)
            tickers = [[inverse_stock_index.get(int(value), str(value)) for value in row] for row in test_stock_indexes]
            filtered_pred_dates = [date for index, date in enumerate(pred_dates) if test_mask[index]]
            filtered_last_dates = [date for index, date in enumerate(last_dates) if test_mask[index]]
            prediction_rows = [np.expand_dims(value, axis=1) for value in test_predictions]
            label_rows = [np.expand_dims(value, axis=1) for value in test_labels]
            with open(os.path.join(metrics_path, f"val_metrics_sl1_pl{args.pred_len}.json"), "w") as handle:
                json.dump({key: float(value) for key, value in valid_metrics.items()}, handle, indent=4)
            metrics, results_pkl, metrics_json = save_results(
                args, test_metrics, prediction_rows, label_rows,
                filtered_pred_dates, filtered_last_dates, tickers, metrics_path
            )
            best_result = (metrics, results_pkl, metrics_json)
        else:
            stop_round += 1
            if stop_round >= args.early_stop:
                pprint("early stop")
                break

    if best_param is None or best_result is None:
        raise RuntimeError("DiscoverPLF training produced neither a finite validation IC nor a finite validation loss.")
    checkpoint_path = os.path.join(model_save_path, "model.pth")
    torch.save(best_param, checkpoint_path)
    pprint(f"best {best_selection[1]}: {best_selection[2]:.6f} @ {best_epoch + 1}")
    metrics, results_pkl, metrics_json = best_result
    return {"model_path": checkpoint_path, "metrics": metrics,
            "results_path": results_pkl, "metrics_path": metrics_json}


def build_parser():
    parser = argparse.ArgumentParser(description="Train the DiscoverPLF regression model")
    parser.add_argument("--data_path", default="./Evaluation/data")
    parser.add_argument("--model_name", default="DiscoverPLF")
    parser.add_argument("--universe", default="sp500")
    parser.add_argument("--start_date", default="2015-01-01")
    parser.add_argument("--end_train_date", default="2018-12-31")
    parser.add_argument("--start_valid_date", default="2019-01-01")
    parser.add_argument("--end_valid_date", default="2019-12-31")
    parser.add_argument("--start_test_date", default="2023-01-01")
    parser.add_argument("--end_date", default="2023-12-31")
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--d_feat", type=int, default=6)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--early_stop", type=int, default=25)
    parser.add_argument("--smooth_steps", type=int, default=5)
    parser.add_argument("--metric", default="IC")
    parser.add_argument("--loss", default="mse")
    parser.add_argument("--beta_strategy", default="increasing")
    parser.add_argument("--pin_memory", action="store_false", default=True)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--label", default="")
    parser.add_argument("--stride", default="1+2+5")
    parser.add_argument("--num_days", default="60")
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    output = run_train(build_parser().parse_args())
    print("__TRAIN_JSON_START__", flush=True)
    print(json.dumps(output, ensure_ascii=False), flush=True)
    print("__TRAIN_JSON_END__", flush=True)
