import pandas as pd
from paths import get_config_path
from .eodhd import download_multi
from .storage import load_config, read_stock_dataset

import quantstats as qs


def evaluate_benchmarks(benchmarks: list[str], start_date: str, end_date: str, api_token: str) -> pd.DataFrame:
    """ Evaluate performance metrics for multiple benchmarks over a specified period """

    mkeys = ['Cumulative Return', 'CAGR﹪', 'Sharpe', 'Sortino', 'Volatility (ann.)', 'Max Drawdown', 'Avg. Drawdown']

    # Download data
    asset_returns, errors = download_multi(benchmarks, api_token=api_token, data_type='returns', p=4)
    if errors:
        raise ValueError(f"Failed to download {len(errors)} benchmarks: {errors}")

    results = {}

    for benchmark, returns in asset_returns.items():
        # returns = returns[(returns.index >= start_date) & (returns.index <= end_date)]
        # Filter by date range
        returns = returns.loc[start_date:end_date]

        if returns.empty:
            print(f"Warning: No data for {benchmark} in specified date range")
            continue

        # Extract core metrics
        metrics = qs.reports.metrics(returns, mode='full', rf=0.0, compounded=True, display=False)['Strategy'].to_dict()

        benchmark_results = {metric: metrics.get(metric, None) for metric in mkeys}

        # Calculate end-of-year returns
        eoy_returns = returns.resample('YE').apply(qs.stats.comp)
        eoy_returns.index = eoy_returns.index.year
        benchmark_results.update(eoy_returns.to_dict())

        results[benchmark] = benchmark_results

    # Convert to DataFrame and transpose
    df = pd.DataFrame(results).T

    return df


def download_asset_prices(pf_history: list[dict], price_path: str = None) -> dict[str, pd.DataFrame]:
    userdata = load_config(str(get_config_path("config.json")))
    asset_prices = {}

    # Try to load cached prices if path provided
    if price_path:
        prices = read_stock_dataset(price_path)
        max_date = prices.index.get_level_values('date').max()

        # Only use cached data if it covers the full date range
        if max_date > pf_history[-1]['test_end']:
            asset_prices = {k: df.droplevel(0) for k, df in prices.groupby('instrument')}
            print(f"Loaded {len(asset_prices)} assets from cache")

    # Identify missing tickers
    pf_tickers = {ticker for snp in pf_history for ticker in snp['tickers']}
    missing_tickers = list(pf_tickers - set(asset_prices.keys()))

    # Download missing prices
    if missing_tickers:
        print(f"Downloading {len(missing_tickers)} missing tickers...")
        missing_prices, errors = download_multi(
            missing_tickers,
            api_token=userdata.get('eodhd'),
            data_type='price',
            verbose=True,
            p=4
        )
        if errors:
            raise ValueError(f"{len(errors)} tickers failed to download: {errors}")

        asset_prices.update(missing_prices)

    return asset_prices
