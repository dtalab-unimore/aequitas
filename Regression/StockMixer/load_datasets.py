import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.filter_constituents import filter_constituents_by_date


def find_max_values(df):
    values = {
        column: float(df[column].abs().max())
        for column in ["adj_open", "adj_close", "adj_high", "adj_low", "volume"]
    }
    return {column: value if np.isfinite(value) and value > 0 else 1.0 for column, value in values.items()}


def dataset_normalization(dataset, max_values):
    dataset = dataset.copy()
    for column, maximum in max_values.items():
        dataset[column] = dataset[column].astype("float32") / maximum
    return dataset


def _context_index(dates, boundary, seq_len, pred_len):
    positions = np.flatnonzero(dates.to_numpy() >= boundary)
    if len(positions) == 0:
        raise ValueError(f"No StockMixer data is available on or after {boundary}.")
    return max(0, int(positions[0]) - seq_len - pred_len + 1)


def load_dataset(args):
    dataset = pd.read_csv(f"{args.data_path}/{args.universe}/{args.universe}.csv")
    trained_tickers = getattr(args, "trained_tickers", None)
    if trained_tickers:
        requested_tickers = list(trained_tickers)
    else:
        constituents = pd.read_csv(f"{args.data_path}/constituents/eodhd/{args.universe}.csv")
        requested_tickers = filter_constituents_by_date(
            constituents, args.start_test_date
        )["EODHD"].tolist()

    dataset = dataset[dataset["instrument"].isin(requested_tickers)]
    dataset = dataset[["date", "instrument", "adj_open", "adj_close", "adj_high", "adj_low", "volume"]]
    dataset = dataset[dataset["volume"] > 0].sort_values(["date", "instrument"])
    all_dates = dataset["date"].drop_duplicates().sort_values().reset_index(drop=True)
    ticker_order = list(trained_tickers) if trained_tickers else sorted(dataset["instrument"].unique())

    all_data, ground_truth, price_data, available_tickers = [], [], [], []
    for ticker in tqdm(ticker_order):
        frame = dataset[dataset["instrument"] == ticker].copy().sort_values("date")
        if frame.empty or frame["date"].min() > args.start_date or frame["date"].max() < args.end_date:
            continue
        frame["label"] = (frame["adj_close"].shift(-args.pred_len) - frame["adj_close"]) / frame["adj_close"]
        frame = pd.DataFrame({"date": all_dates}).merge(frame, on="date", how="left")
        feature_columns = ["instrument", "adj_open", "adj_close", "adj_high", "adj_low", "volume"]
        frame[feature_columns] = frame[feature_columns].ffill().bfill()
        frame = frame[(frame["date"] >= args.start_date) & (frame["date"] <= args.end_date)]
        maximums = find_max_values(
            frame[(frame["date"] >= args.start_date) & (frame["date"] <= args.end_train_date)]
        )
        ground_truth.append(frame["label"].to_numpy(dtype=np.float32))
        normalized = dataset_normalization(frame, maximums)
        features = normalized.drop(columns=["instrument", "date", "label"])
        all_data.append(features.to_numpy(dtype=np.float32))
        price_data.append(normalized["adj_close"].to_numpy(dtype=np.float32))
        available_tickers.append(ticker)

    if not available_tickers:
        raise ValueError("No ticker has enough data for the requested StockMixer periods.")
    if trained_tickers and available_tickers != list(trained_tickers):
        raise ValueError("The test period does not contain every ticker used to train StockMixer.")

    dates = all_dates[(all_dates >= args.start_date) & (all_dates <= args.end_date)].reset_index(drop=True)
    valid_index = _context_index(dates, args.start_valid_date, args.seq_len, args.pred_len)
    test_index = _context_index(dates, args.start_test_date, args.seq_len, args.pred_len)
    if not 0 < valid_index < test_index < len(dates):
        raise ValueError("StockMixer periods do not produce ordered train/validation/test windows.")

    return (
        np.asarray(all_data), np.asarray(ground_truth), np.asarray(price_data),
        valid_index, test_index, dates, available_tickers,
    )
