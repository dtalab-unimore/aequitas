import glob
import os

import pandas as pd
from tqdm import tqdm

from utils.filter_constituents import filter_constituents_by_date


def load_dataset(start_date, end_date, train_end_date, val_end_date, start_test_date, path, universe, root_path, batch_size, pred_len, seq_len):
    data = pd.read_csv(f'{path}/{universe}/{universe}.csv')
    constituents = pd.read_csv(f'{path}/constituents/eodhd/{universe}.csv')
    tickers = filter_constituents_by_date(constituents, start_test_date)['EODHD'].tolist()
    data = data[data['volume'] > 0]

    available_tickers = []

    if os.path.exists(root_path):
        files = glob.glob(os.path.join(root_path, '*'))
        for f in files:
            os.remove(f)
    else:
        os.makedirs(root_path)

    for ticker in tqdm(tickers, desc='Processing tickers'):
        df = data[data['instrument'] == ticker]
        df = df[['date', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'volume']]

        df['adj_open'] = df['adj_open'] / df['adj_close']
        df['adj_high'] = df['adj_high'] / df['adj_close']
        df['adj_low'] = df['adj_low'] / df['adj_close']
        df['adj_abs_ret'] = df['adj_close'] - df['adj_close'].shift(1)
        df['target'] = df['adj_close'] / df['adj_close'].shift(1)
        df = df.drop(['adj_close'], axis=1)
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        df = df.dropna(axis=0)

        train_df = df[df['date'] <= train_end_date]
        val_df = df[(df['date'] > train_end_date) & (df['date'] <= val_end_date)]
        test_df = df[df['date'] > val_end_date]

        if len(df) > 0 and len(train_df) >= 251 and len(val_df) >= len(train_df) * 0.1 and len(test_df) - pred_len - seq_len >= batch_size:
            df.to_csv(f'{root_path}/{ticker.replace(".","_")}.csv', index=False)
            available_tickers.append(ticker)

    return available_tickers


