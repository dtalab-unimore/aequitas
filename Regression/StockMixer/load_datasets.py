import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.filter_constituents import filter_constituents_by_date


def find_max_values(df):
    max_values = {
        'adj_open': df['adj_open'].max(),
        'adj_close': df['adj_close'].max(),
        'adj_high': df['adj_high'].max(),
        'adj_low': df['adj_low'].max(),
        'volume': df['volume'].max()
    }
    return max_values

def dataset_normalization(dataset, max_values):
    dataset = dataset.copy()
    dataset['volume'] = dataset['volume'].astype('float32')
    dataset.loc[:, 'adj_open'] = dataset['adj_open'] / max_values['adj_open']
    dataset.loc[:, 'adj_close'] = dataset['adj_close'] / max_values['adj_close']
    dataset.loc[:, 'adj_high'] = dataset['adj_high'] / max_values['adj_high']
    dataset.loc[:, 'adj_low'] = dataset['adj_low'] / max_values['adj_low']
    dataset.loc[:, 'volume'] = dataset['volume'] / max_values['volume']
    return dataset


def load_dataset(data_path, universe, start_date, end_train_date, end_valid_date, start_test_date, end_date, pred_len, lookback_length):

    dataset = pd.read_csv(f'{data_path}/{universe}/{universe}.csv')
    constituents = pd.read_csv(f'{data_path}/constituents/eodhd/{universe}.csv')
    tickers = filter_constituents_by_date(constituents, start_test_date)['EODHD'].tolist()
    dataset = dataset[dataset['instrument'].isin(tickers)]

    dataset = dataset[['date', 'instrument', 'adj_open', 'adj_close', 'adj_high', 'adj_low', 'volume']]
    dataset = dataset.sort_values(by=['date', 'instrument'])
    dataset = dataset[dataset['volume'] > 0]

    tickers = dataset['instrument'].drop_duplicates().reset_index(drop=True).tolist()
    dates = dataset['date'].drop_duplicates().reset_index(drop=True)

    all_data = []
    gt_data = []
    price_data = []
    available_tickers = []

    for ticker in tqdm(tickers):
        df = dataset[dataset['instrument'] == ticker].copy()
        df['label'] = (df['adj_close'].shift(-pred_len) - df['adj_close']) / df['adj_close']

        if ticker == 'SBNY.US':
            continue

        if df['date'].min() <= start_date and df['date'].max() >= end_date:
            df = pd.merge(dates, df, on='date', how='left').ffill().bfill().infer_objects()
            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            df = df[mask]
            gt_data.append(df['label'])
            max_values = find_max_values(df[(df['date'] >= start_date) & (df['date'] <= end_train_date)])
            df = dataset_normalization(df, max_values)
            df = df.drop(columns=['instrument', 'date', 'label'])

            all_data.append(df)
            available_tickers.append(ticker)
            price_data.append(df['adj_close'])

    dates = dates[(dates >= start_date) & (dates <= end_date)].reset_index(drop=True)
    valid_index = pd.to_datetime(dates[dates > end_train_date]).idxmin() - lookback_length - pred_len + 1
    test_index = pd.to_datetime(dates[dates > end_valid_date]).idxmin() - lookback_length - pred_len + 1

    return np.array(all_data), np.array(gt_data), np.array(price_data), valid_index, test_index, dates, available_tickers

