import os
import pickle
from argparse import ArgumentParser
from glob import glob
from types import SimpleNamespace

import quantstats as qs
from convert_predictions import convert_classification_preds, convert_daily_to_cumulative_returns
from utils.evaluation import download_asset_prices
from portfolio.transforms import create_long_short_portfolio_history
from portfolio.returns import portfolio_daily_returns

from paths import (
    get_price_path,
    get_model_results_dir,
    get_evaluation_output_dir,
)


def filter_valid_dates(pred_paths: list[str], args):
    for elem in pred_paths:
        with open(elem, 'rb') as f:
            results = pickle.load(f)

        tickers = results['tickers']

        # case 1: string list -> each list element is a ticker
        if isinstance(tickers[0], str):
            common = set(tickers)

        # case 2: list of list -> each list element is a list of tickers for a date
        else:
            max_len = max(len(t) for t in tickers)
            k = max(1, int(0.1 * max_len))

            mask = [len(i) > k for i in tickers]

            results['preds'] = [x for x, m in zip(results['preds'], mask) if m]
            results['labels'] = [x for x, m in zip(results['labels'], mask) if m]
            results['last_date'] = [x for x, m in zip(results['last_date'], mask) if m]
            results['pred_date'] = [x for x, m in zip(results['pred_date'], mask) if m]
            results['tickers'] = [x for x, m in zip(results['tickers'], mask) if m]

            if not results['tickers']:
                raise ValueError(f"No valid dates left after filtering in {elem}")

            common = set(results['tickers'][0])
            for t in results['tickers'][1:]:
                common &= set(t)

        common_len = len(common)
        limit = args.top_k + args.short_k

        if limit <= 0:
            raise ValueError("At least one between top_k and short_k must be > 0")

        if common_len < limit:
            raise ValueError(f"Not enough valid tickers in {elem}: {common_len} < {limit}")
        else:
            print(f"{elem}: {common_len} valid tickers")

        with open(elem, 'wb') as f:
            pickle.dump(results, f)


def discover_prediction_files(args):
    """
    Find all prediction files matching:
    <root>/<type>/<model>/results/<universe>/<config>/<seed>/y<start>_y<end>/results_sl<sl>_pl<pl>.pkl
    """

    base_results_dir = get_model_results_dir(
        prediction_type=args.type,
        model=args.model,
        universe=args.universe,
        results_root=args.results_root,
    )

    if args.seed is None:
        pattern = os.path.join(
            base_results_dir,
            "*",
            "*",
            f"y{args.initial_year}_y{args.end_year}",
            f"results_sl{args.sl}_pl{args.pl}.pkl"
        )
    else:
        pattern = os.path.join(
            base_results_dir,
            "*",
            str(args.seed),
            f"y{args.initial_year}_y{args.end_year}",
            f"results_sl{args.sl}_pl{args.pl}.pkl"
        )

    pred_paths = sorted(glob(pattern))

    if not pred_paths:
        raise FileNotFoundError(
            "No prediction files found.\n"
            f"Search pattern was:\n{pattern}"
        )

    runs = []

    for pred_path in pred_paths:
        # Expected structure:
        # .../<universe>/<config>/<seed>/y<start>_y<end>/results_sl<x>_pl<y>.pkl
        year_folder = os.path.basename(os.path.dirname(pred_path))
        seed_folder = os.path.basename(os.path.dirname(os.path.dirname(pred_path)))
        config_folder = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(pred_path))))

        runs.append({
            "config": config_folder,
            "seed": int(seed_folder),
            "year_folder": year_folder,
            "pred_path": pred_path,
        })

    return runs


def demo_portfolio_evaluation(args):
    # Config params
    price_path = get_price_path(args.universe)

    out_dir = get_evaluation_output_dir(
        universe=args.universe,
        prediction_type=args.type,
        model=args.model,
        config=args.config,
        seed=args.seed,
        initial_year=args.initial_year,
        end_year=args.end_year,
        top_k=args.top_k,
        short_k=args.short_k,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_paths = [args.pred_path]

    print("Preds_paths:", pred_paths)

    # Preprocessing: filter dates with enough tickers
    filter_valid_dates(pred_paths, args)

    # Preprocessing: convert classification predictions
    if args.type == 'Classification':
        pred_paths = convert_classification_preds(pred_paths, args.model)

    if args.type == 'Regression' and args.model == 'D-Va':
        pred_paths, args.pl = convert_daily_to_cumulative_returns(pred_paths, args.sl, args.pl)

    # Create portfolio history
    print("Creating portfolio history...")
    pf_history = create_long_short_portfolio_history(
        pred_paths,
        top_k=args.top_k,
        short_k=args.short_k,
        start_date=f"{args.initial_year}-01-01",
        end_date=f"{args.end_year}-12-31",
        freq=f'{args.pl}B',
        verbose=True
    )

    with open(out_dir / f"pf_history_sl{args.sl}_pl{args.pl}_seed{args.seed}.pkl", "wb") as f:
        pickle.dump(pf_history, f)

    # Download/load asset prices
    print("Loading asset prices...")
    asset_prices = download_asset_prices(pf_history, price_path)

    # Compute portfolio daily returns
    print("Computing portfolio returns...")
    pf_returns = portfolio_daily_returns(pf_history, asset_prices)
    pf_cum_rets = (1 + pf_returns).cumprod() * 100000

    # Compute strategy metrics
    metrics = qs.reports.metrics(pf_returns, mode='full', rf=0.0, compounded=True, display=False)
    metrics.to_csv(out_dir / f"metrics_sl{args.sl}_pl{args.pl}_seed{args.seed}.csv")

    # Compute EOY returns
    eoy_returns = pf_returns.resample("YE").apply(qs.stats.comp)
    eoy_returns.to_csv(out_dir / f"eoy_returns_sl{args.sl}_pl{args.pl}_seed{args.seed}.csv")

    print("Save results in:", out_dir)


def build_runtime_args(args, run):
    return SimpleNamespace(
        universe=args.universe,
        model=args.model,
        type=args.type,
        seed=run["seed"],
        sl=args.sl,
        pl=args.pl,
        top_k=args.top_k,
        short_k=args.short_k,
        initial_year=args.initial_year,
        end_year=args.end_year,
        results_root=args.results_root,
        config=run["config"],
        pred_path=run["pred_path"],
    )


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--universe', type=str, default='sx5e', help='Universe', required=True)
    parser.add_argument('--model', type=str, default='MASTER', help='Model name', required=True)
    parser.add_argument('--initial_year', type=int, default=2021, required=True)
    parser.add_argument('--end_year', type=int, default=2024, required=True)
    parser.add_argument('--seed', type=int, default=42, help='Random seed. Use -1 to evaluate all seeds.')
    parser.add_argument('--type', type=str, default='Regression', choices=['Regression', 'Classification', 'Ranking'],
                        help='Prediction type')
    parser.add_argument('--sl', type=int, default=5, help='sequence length')
    parser.add_argument('--pl', type=int, default=1, help='pred length')
    parser.add_argument('--top_k', type=int, default=10, help='top_k')
    parser.add_argument('--short_k', type=int, default=0, help='short_k')

    parser.add_argument(
        '--results_root',
        type=str,
        default='/hdd10t/n284480/FINBENCH_DEMO',
        help='Root directory containing <type>/<model>/results/...'
    )

    args = parser.parse_args()

    if args.end_year < args.initial_year:
        raise ValueError("end_year must be >= initial_year")

    if args.seed == -1:
        args.seed = None

    print(f'\n==== {args.model} ====')
    print(f"Evaluation period: y{args.initial_year}_y{args.end_year}")

    runs = discover_prediction_files(args)

    print(f"Found {len(runs)} prediction file(s).")

    for run in runs:
        print(
            f"[RUN] {args.model} | {args.universe} | config={run['config']} | "
            f"seed={run['seed']} | sl={args.sl} pl={args.pl} | "
            f"period=y{args.initial_year}_y{args.end_year}"
        )

        runtime_args = build_runtime_args(args, run)
        demo_portfolio_evaluation(runtime_args)

    # evaluate_benchmarks(
    #     benchmarks=['SXXP.INDX', 'GSPC.INDX', 'DJI.INDX', 'SX5E.INDX', 'NDX.INDX'],
    #     start_date='2020-01-01',
    #     end_date='2024-12-31',
    #     api_token=load_config('config.json').get('eodhd')
    # )
