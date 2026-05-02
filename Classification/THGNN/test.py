import json
import os
from argparse import ArgumentParser

import pandas as pd

from utils.filter_constituents import filter_constituents_by_date
from utils.load_configuation import _log, load_configuration

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))


def run_test(
        task_type,
        test_model,
        universe,
        config_name,
        sl_value,
        pl_value,
        start_test_date,
        end_date,
        log_fn=None,
):
    args, model_save_path, metrics_path = load_configuration(
        task_type, test_model, universe, config_name, sl_value, pl_value, start_test_date, end_date, PROJECT_ROOT, log_fn
    )

    dataset = pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}.csv')
    dataset = dataset[['date', 'instrument', 'open', 'high', 'low', 'close', 'volume', 'adj_close']]
    constituents = pd.read_csv(f'{args.data_path}/constituents/eodhd/{args.universe}.csv')
    tickers = filter_constituents_by_date(constituents, args.start_test_date)['EODHD'].tolist()
    dataset = dataset[dataset['instrument'].isin(tickers)]



if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--task_type", type=str, default='Classification')
    parser.add_argument("--model_name", type=str, default='THGNN')
    parser.add_argument("--universe", type=str, default='sx5e')
    parser.add_argument("--config_name", type=str, default='THGNN')
    parser.add_argument("--sl_value", type=int, default=20)
    parser.add_argument("--pl_value", type=int, default=5)
    parser.add_argument("--start_test_date", type=str, default='2023-01-01')
    parser.add_argument("--end_date", type=str, default='2023-01-01')
    args = parser.parse_args()

    out = run_test(
        task_type=args.task_type,
        test_model=args.model_name,
        universe=args.universe,
        config_name=args.config_name,
        sl_value=args.sl_value,
        pl_value=args.pl_value,
        start_test_date=args.start_test_date,
        end_date=args.end_date,
    )
    print(json.dumps(out, indent=4))