import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.filter_constituents import filter_constituents_by_date
from utils.training import first_test_day


def find_max_values(df):
    max_values = {
        'adj_close': df['adj_close'].max(),
    }
    return max_values

def dataset_normalization(dataset, max_values):
    dataset = dataset.copy()
    dataset.loc[:, 'adj_close'] = dataset['adj_close'] / max_values['adj_close']
    dataset.loc[:, 'ma5'] = dataset['ma5'] / max_values['adj_close']
    dataset.loc[:, 'ma10'] = dataset['ma10'] / max_values['adj_close']
    dataset.loc[:, 'ma20'] = dataset['ma20'] / max_values['adj_close']
    dataset.loc[:, 'ma30'] = dataset['ma30'] / max_values['adj_close']
    return dataset


def load_dataset(data_path, universe, start_date, end_train_date, end_valid_date, start_test_date, end_date, pred_len, lookback_length):
    """
    start_test_date / end_date are None in training: the data then ends with
    the validation period and test_index marks the end of validation. The
    labels are computed before the date filter, so the last validation days
    still have them.
    """
    end_date = end_date or end_valid_date

    dataset = pd.read_csv(f'{data_path}/{universe}/{universe}.csv')
    constituents = pd.read_csv(f'{data_path}/constituents/eodhd/{universe}.csv')
    tickers = filter_constituents_by_date(constituents, start_test_date or first_test_day(end_valid_date))['EODHD'].tolist()
    dataset = dataset[dataset['instrument'].isin(tickers)]

    dataset = dataset[['date', 'instrument', 'adj_close']]
    dataset = dataset.sort_values(by=['date', 'instrument'])

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
            df = pd.merge(dates, df, on='date', how='left').ffill().infer_objects()

            df['ma5'] = df['adj_close'].rolling(5).mean()
            df['ma10'] = df['adj_close'].rolling(10).mean()
            df['ma20'] = df['adj_close'].rolling(20).mean()
            df['ma30'] = df['adj_close'].rolling(30).mean()

            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            df = df[mask]

            df = df.bfill().infer_objects()

            gt_data.append(df['label'])
            max_values = find_max_values(df[(df['date'] >= start_date) & (df['date'] <= end_train_date)])
            df = dataset_normalization(df, max_values)
            df = df.drop(columns=['instrument', 'date', 'label'])
            all_data.append(df)
            available_tickers.append(ticker)
            price_data.append(df['adj_close'])

    dates = dates[(dates >= start_date) & (dates <= end_date)].reset_index(drop=True)
    lookback = lookback_length + pred_len - 1
    # Offset of the first sample whose target falls after the given date
    # (counting the dates also works when the data stops right there)
    valid_index = int((dates <= end_train_date).sum()) - lookback
    if start_test_date:
        # The test starts at start_test_date, not necessarily right after validation
        test_index = int((dates < start_test_date).sum()) - lookback
    else:
        test_index = int((dates <= end_valid_date).sum()) - lookback

    return np.array(all_data), np.array(gt_data), np.array(price_data), valid_index, test_index, dates, available_tickers


def save_dates(dates, tickers, start_index, pred_len, seq_len):
    last_seq_date = dates[start_index + seq_len - 1: - pred_len]
    dates_gt = dates[start_index + seq_len + pred_len - 1:]
    tickers = [tickers] * len(dates_gt)
    return dates_gt, last_seq_date, tickers


def load_incidence_matrix(tickers, args):
    inci_matrix = np.load(f'{args.data_path}/{args.universe}/{args.universe}_inc_matrix.npz')
    matr = inci_matrix['inc_matrix']
    matr_tickers = inci_matrix['tickers']

    matr_tickers = [mt in tickers for mt in matr_tickers]
    matr = matr[matr_tickers]

    return matr