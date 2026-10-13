import glob
import os

import pandas as pd

from utils.filter_constituents import filter_constituents_by_date


def ticker_csv_name(ticker):
    return f"{ticker.replace('.', '_')}.csv"


def read_prices(path, universe):
    data = pd.read_csv(f'{path}/{universe}/{universe}.csv')
    return data[data['volume'] > 0]


def ticker_frame(data, ticker):
    """D-Va input features of one ticker: OHLC relative to close, volume, returns, target."""
    df = data[data['instrument'] == ticker]
    df = df[['date', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'volume']]

    df['adj_open'] = df['adj_open'] / df['adj_close']
    df['adj_high'] = df['adj_high'] / df['adj_close']
    df['adj_low'] = df['adj_low'] / df['adj_close']
    df['adj_abs_ret'] = df['adj_close'] - df['adj_close'].shift(1)
    df['target'] = df['adj_close'] / df['adj_close'].shift(1)
    df = df.drop(['adj_close'], axis=1)
    return df


def reset_dir(root_path):
    if os.path.exists(root_path):
        for f in glob.glob(os.path.join(root_path, '*')):
            os.remove(f)
    else:
        os.makedirs(root_path)


def load_dataset(start_date, end_date, train_end_date, val_end_date, start_test_date, path, universe, root_path, batch_size, pred_len, seq_len):
    data = read_prices(path, universe)
    constituents = pd.read_csv(f'{path}/constituents/eodhd/{universe}.csv')
    tickers = filter_constituents_by_date(constituents, start_test_date)['EODHD'].tolist()

    available_tickers = []
    reset_dir(root_path)

    for ticker in tickers:
        df = ticker_frame(data, ticker)
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        df = df.dropna(axis=0)

        train_df = df[df['date'] <= train_end_date]
        val_df = df[(df['date'] > train_end_date) & (df['date'] <= val_end_date)]
        test_df = df[df['date'] > val_end_date]

        if len(df) > 0 and len(train_df) >= 251 and len(val_df) >= len(train_df) * 0.1 and len(test_df) - pred_len - seq_len >= batch_size:
            df.to_csv(f'{root_path}/{ticker_csv_name(ticker)}', index=False)
            available_tickers.append(ticker)

    return available_tickers
