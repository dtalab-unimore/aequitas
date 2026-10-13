import os
import shutil
import tempfile
from argparse import ArgumentParser

from .train import clear_data_cache, preprocess, run_estimate, save_estimate_results
from utils.load_configuation import load_configuration
from utils.logger import _log, emit_result

ESTIMATE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ESTIMATE_DIR, "..", ".."))


def run_test(task_type, test_model, universe, config_name, sl_value, pl_value,
             start_test_date, end_date, log_fn=None):
    _log(log_fn, "Preparing test...")
    args, model_save_path, metrics_path = load_configuration(
        task_type, test_model, universe, config_name, sl_value, pl_value,
        start_test_date, end_date, PROJECT_ROOT, log_fn
    )

    work_dir = tempfile.mkdtemp(prefix="estimate_test_", dir=model_save_path)
    try:
        # The model is built for the trained set of stocks: reuse it, in order.
        clear_data_cache(model_save_path)
        preprocess(args, work_dir, tickers=args.trained_tickers)
        results_file = run_estimate(args, work_dir, model_save_path, train=False)
        metrics, results_pkl, metrics_json = save_estimate_results(args, results_file, metrics_path, log_fn)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        clear_data_cache(model_save_path)

    return {"metrics": metrics, "results_path": results_pkl, "metrics_path": metrics_json}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task_type", type=str, default='Regression')
    parser.add_argument("--model_name", type=str, default='ESTIMATE')
    parser.add_argument("--universe", type=str, default='sp500')
    parser.add_argument("--config_name", type=str, default='ESTIMATE')
    parser.add_argument("--sl_value", type=int, default=20)
    parser.add_argument("--pl_value", type=int, default=5)
    parser.add_argument("--start_test_date", type=str, default='2023-01-01')
    parser.add_argument("--end_date", type=str, default='2023-12-31')
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
    emit_result(out)
