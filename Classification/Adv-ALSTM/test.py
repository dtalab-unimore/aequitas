import os
from argparse import ArgumentParser

from .evaluator import evaluate
from .train import WEIGHTS_FILE, build_model, load_data
from utils.load_configuation import load_configuration
from utils.logger import _log, emit_result
from utils.save_results import save_results

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))


def run_test(task_type, test_model, universe, config_name, sl_value, pl_value,
             start_test_date, end_date, log_fn=None):
    _log(log_fn, "Preparing test...")
    args, model_save_path, metrics_path = load_configuration(
        task_type, test_model, universe, config_name, sl_value, pl_value,
        start_test_date, end_date, PROJECT_ROOT, log_fn
    )

    _, _, _, _, tes_pv, tes_gt, test_tickers, test_last_dates, test_dates = load_data(args)

    model = build_model(args, tes_pv.shape[2])
    model.load(os.path.join(model_save_path, WEIGHTS_FILE))
    tes_pre = model.predict(tes_pv).numpy()

    metrics, results_pkl, metrics_json = save_results(
        args, evaluate(tes_pre, tes_gt, bool(args.hinge_lose)), tes_pre, tes_gt,
        test_dates, test_last_dates, test_tickers, metrics_path, log_fn=log_fn
    )
    return {"metrics": metrics, "results_path": results_pkl, "metrics_path": metrics_json}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task_type", type=str, default='Classification')
    parser.add_argument("--model_name", type=str, default='Adv-ALSTM')
    parser.add_argument("--universe", type=str, default='sp500')
    parser.add_argument("--config_name", type=str, default='Adv-ALSTM')
    parser.add_argument("--sl_value", type=int, default=15)
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
