import os
import sys
import pickle
from argparse import ArgumentParser
from glob import glob
from pathlib import Path
from types import SimpleNamespace
import quantstats as qs
from convert_predictions import convert_classification_preds, convert_daily_to_cumulative_returns
from portfolio.returns import portfolio_daily_returns
from portfolio.transforms import create_quintile_portfolios_history
from utils.evaluation import download_asset_prices
from paths import (
    get_price_path,
    get_quintile_output_dir,
)


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CURRENT_DIR))


def discover_prediction_files(args):
    """
    Find all prediction files matching:
    <results_root>/<type>/<model>/results/<universe>/<config>/<seed>/y<start>_y<end>/results_sl<sl>_pl<pl>.pkl
    """

    base_results_dir = os.path.join(
        args.results_root,
        args.type,
        args.model,
        "results",
        args.universe,
    )

    if args.seed is None:
        pattern = os.path.join(
            base_results_dir,
            "*",
            "*",
            f"y{args.initial_year}_y{args.end_year}",
            f"results_sl{args.sl}_pl{args.pl}.pkl",
        )
    else:
        pattern = os.path.join(
            base_results_dir,
            "*",
            str(args.seed),
            f"y{args.initial_year}_y{args.end_year}",
            f"results_sl{args.sl}_pl{args.pl}.pkl",
        )

    pred_paths = sorted(glob(pattern))

    if not pred_paths:
        raise FileNotFoundError(
            "No prediction files found.\n"
            f"Search pattern was:\n{pattern}"
        )

    runs = []

    for pred_path in pred_paths:
        seed_folder = os.path.basename(os.path.dirname(os.path.dirname(pred_path)))
        config_folder = os.path.basename(
            os.path.dirname(os.path.dirname(os.path.dirname(pred_path)))
        )

        runs.append(
            {
                "config": config_folder,
                "seed": int(seed_folder),
                "pred_path": pred_path,
            }
        )

    return runs


def demo_portfolio_evaluation_quintile(args):
    price_path = get_price_path(args.universe)

    out_dir = get_quintile_output_dir(
        universe=args.universe,
        prediction_type=args.type,
        model=args.model,
        config=args.config,
        seed=args.seed,
        initial_year=args.initial_year,
        end_year=args.end_year,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    pred_paths = [args.pred_path]

    print("Preds_paths:", pred_paths)

    if args.type == "Classification":
        pred_paths = convert_classification_preds(pred_paths, args.model)

    if args.type == "Regression" and args.model == "D-Va":
        pred_paths, args.pl = convert_daily_to_cumulative_returns(
            pred_paths,
            args.sl,
            args.pl,
        )

    print("Creating portfolio history...")

    quintile_histories = create_quintile_portfolios_history(
        pred_paths,
        start_date=f"{args.initial_year}-01-01",
        end_date=f"{args.end_year}-12-31",
        freq=f"{args.pl}B",
        verbose=True,
    )

    for q, pf_history in quintile_histories.items():
        print(f"Processing quintile {q}...")

        with open(
                os.path.join(
                    out_dir,
                    f"pf_history_q{q}_sl{args.sl}_pl{args.pl}_seed{args.seed}.pkl",
                ),
                "wb",
        ) as f:
            pickle.dump(pf_history, f)

        print("Loading asset prices...")
        asset_prices = download_asset_prices(pf_history, price_path)

        print("Computing portfolio returns...")
        pf_returns = portfolio_daily_returns(pf_history, asset_prices)

        metrics = qs.reports.metrics(
            pf_returns,
            mode="full",
            rf=0.0,
            compounded=True,
            display=False,
        )

        metrics.to_csv(
            os.path.join(
                out_dir,
                f"metrics_q{q}_sl{args.sl}_pl{args.pl}_seed{args.seed}.csv",
            )
        )

        eoy_returns = pf_returns.resample("YE").apply(qs.stats.comp)

        eoy_returns.to_csv(
            os.path.join(
                out_dir,
                f"eoy_returns_q{q}_sl{args.sl}_pl{args.pl}_seed{args.seed}.csv",
            )
        )

    print("Saved results in:", out_dir)


def build_runtime_args(args, run):
    return SimpleNamespace(
        universe=args.universe,
        model=args.model,
        type=args.type,
        seed=run["seed"],
        sl=args.sl,
        pl=args.pl,
        initial_year=args.initial_year,
        end_year=args.end_year,
        results_root=args.results_root,
        config=run["config"],
        pred_path=run["pred_path"],
    )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--universe", type=str, default="sx5e", help="Universe")
    parser.add_argument("--model", type=str, default="MASTER", help="Model name")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Use -1 to evaluate all seeds.",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="Regression",
        choices=["Regression", "Classification", "Ranking"],
        help="Type",
    )
    parser.add_argument("--sl", type=int, default=5, help="sequence length")
    parser.add_argument("--pl", type=int, default=1, help="pred length")
    parser.add_argument("--initial_year", type=int, default=2021, help="start year")
    parser.add_argument("--end_year", type=int, default=2024, help="end year")

    parser.add_argument(
        "--results_root",
        type=str,
        default="/hdd10t/n284480/FINBENCH_DEMO",
        help="Root directory containing <type>/<model>/results/...",
    )

    args = parser.parse_args()

    if args.end_year < args.initial_year:
        raise ValueError("end_year must be >= initial_year")

    if args.seed == -1:
        args.seed = None

    print(f"\n==== {args.model} ====")
    print(f"Evaluation period: y{args.initial_year}_y{args.end_year}")

    runs = discover_prediction_files(args)

    print(f"Found {len(runs)} prediction file(s).")

    for run in runs:
        print(
            f"[RUN] {args.model} | {args.universe} | "
            f"config={run['config']} | seed={run['seed']} | "
            f"sl={args.sl} pl={args.pl} | "
            f"period=y{args.initial_year}_y{args.end_year}"
        )

        runtime_args = build_runtime_args(args, run)
        demo_portfolio_evaluation_quintile(runtime_args)
