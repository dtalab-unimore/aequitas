import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from tqdm import tqdm

from utils.filter_constituents import filter_constituents_by_date
from utils.training import ticker_selection_date


def read_datasets(args):
    df_close = pd.read_csv(f"{args.data_path}/{args.universe}/{args.universe}.csv")
    df_close = df_close[["instrument", "date", "close"]]
    df_tech = pd.read_csv(f"{args.data_path}/{args.universe}/{args.universe}_tech.csv")
    df_tech = df_tech[[
        "instrument", "date", "vol", "mom1", "mom2", "mom3", "roc5", "roc10",
        "roc15", "roc20", "ema10", "ema20", "ema50", "ema200",
    ]]
    market = pd.read_csv(f"{args.data_path}/cnnpred_market.csv")
    dataset = df_tech.merge(df_close, how="inner", on=["instrument", "date"])
    dataset = dataset.merge(market, on="date", how="inner")

    trained_tickers = getattr(args, "trained_tickers", None)
    if trained_tickers:
        tickers = list(trained_tickers)
    else:
        constituents = pd.read_csv(f"{args.data_path}/constituents/eodhd/{args.universe}.csv")
        tickers = filter_constituents_by_date(constituents, ticker_selection_date(args))["EODHD"].tolist()
    dataset = dataset[dataset["instrument"].isin(tickers)]
    dataset = dataset[dataset["date"] >= args.start_date]
    # Training has no end_date: the rows after validation are kept, as the
    # labels of the last validation days look pred_len days ahead.
    if getattr(args, "end_date", None):
        dataset = dataset[dataset["date"] <= args.end_date]
    return dataset.replace([np.inf, -np.inf], np.nan)


def split_train_val_test_ticker(df, args):
    train_data = df[(df.index >= args.start_date) & (df.index <= args.end_train_date)]
    if len(train_data) <= args.seq_len + args.pred_len:
        return None
    train_data = scale(train_data[:-args.pred_len].iloc[:, :-1])
    train_labels = df[(df.index >= args.start_date) & (df.index <= args.end_train_date)]["label"]
    train_labels = train_labels[args.seq_len + args.pred_len - 1:]

    valid_candidates = df.index[df.index >= args.start_valid_date]
    if len(valid_candidates) == 0:
        return None
    valid_index = max(0, df.index.get_loc(valid_candidates.min()) - args.seq_len - args.pred_len + 1)
    valid_frame = df.iloc[valid_index:]
    valid_frame = valid_frame[valid_frame.index <= args.end_valid_date]
    if len(valid_frame) <= args.pred_len:
        return None
    valid_data = scale(valid_frame[:-args.pred_len].iloc[:, :-1])
    valid_labels = df[(df.index >= args.start_valid_date) & (df.index <= args.end_valid_date)]["label"]

    # Training (no start_test_date): train and validation only
    if not getattr(args, "start_test_date", None):
        return [train_data, train_labels, valid_data, valid_labels]

    test_candidates = df.index[df.index >= args.start_test_date]
    if len(test_candidates) == 0:
        return None
    test_index = max(0, df.index.get_loc(test_candidates.min()) - args.seq_len - args.pred_len + 1)
    test_frame = df.iloc[test_index:]
    test_frame = test_frame[test_frame.index <= args.end_date]
    if len(test_frame) <= args.pred_len:
        return None
    test_data = test_frame[:-args.pred_len].iloc[:, :-1]
    test_labels = df[(df.index >= args.start_test_date) & (df.index <= args.end_date)]["label"]
    return [train_data, train_labels, valid_data, valid_labels, test_data, test_labels]


def construct_data_warehouse_3d(args):
    dataset = read_datasets(args)
    all_dates = dataset["date"].drop_duplicates().sort_values()
    all_dates = all_dates[all_dates >= args.start_date]
    if getattr(args, "end_date", None):
        all_dates = all_dates[all_dates <= args.end_date]
    # 6 splits (train, valid, test data and labels), 4 in training: no test
    grouped = None
    tickers = []
    dates = {}

    for ticker, frame in tqdm(dataset.groupby("instrument")):
        frame = frame.fillna(0).drop(columns=["instrument"])
        frame = pd.DataFrame({"date": all_dates}).merge(frame, on="date", how="left").ffill()
        if frame.isnull().values.any():
            continue
        frame = frame.set_index("date").sort_index()
        future_close = frame["close"].shift(-args.pred_len)
        frame["label"] = (future_close / frame["close"] >= 1.0).astype(int)
        split = split_train_val_test_ticker(frame, args)
        if split is None:
            continue
        if grouped is None:
            grouped = [[] for _ in split]
        for index, value in enumerate(split):
            grouped[index].append(value)
        tickers.append(ticker)
        if not dates and len(split) == 6:
            dates["pred_date"] = [str(value) for value in split[5].index]
            dates["last_date"] = [str(value) for value in split[4].index[args.seq_len - 1:]]

    if not tickers:
        raise ValueError("No ticker has enough data for the requested CNNPred3D periods.")
    arrays = [np.asarray(values) for values in grouped]
    arrays = extract_sequences_3d(arrays, args.seq_len)
    return arrays, tickers, dates


def extract_sequences_3d(dataset, seq_len):
    for index in range(len(dataset)):
        if index % 2 == 0:
            dataset[index] = np.lib.stride_tricks.sliding_window_view(
                dataset[index], window_shape=seq_len, axis=1
            ).swapaxes(2, 3)
        dataset[index] = dataset[index].swapaxes(0, 1)
    return dataset
