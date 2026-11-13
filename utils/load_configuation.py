import json
import os
from argparse import Namespace

from .logger import _log
from .training import check_sequential_test


def create_saving_path(args, task_type, test_model, project_root):
    """
    Return (model_save_path, metrics_path), creating the folders.

    metrics_path holds the test results, so it exists only when the args carry
    a test period (start_test_date and end_date, set by load_configuration):
    in train.py, which has no test period, it is None.
    """
    args.seq_len = getattr(args, "seq_len", 1)
    model_save_path = f"model_params/{args.universe}/{args.model_name}/sl_{args.seq_len}/pl_{args.pred_len}"
    model_save_path = os.path.join(project_root, task_type, test_model, model_save_path)
    os.makedirs(model_save_path, exist_ok=True)

    start_test_date = getattr(args, "start_test_date", None)
    end_date = getattr(args, "end_date", None)
    if not start_test_date or not end_date:
        return model_save_path, None

    metrics_path = f"results/{args.universe}/{args.model_name}/{args.seed}/y{start_test_date.split('-')[0]}_y{end_date.split('-')[0]}"
    metrics_path = os.path.join(project_root, task_type, test_model, metrics_path)
    os.makedirs(metrics_path, exist_ok=True)

    return model_save_path, metrics_path

def load_configuration(
        task_type,
        test_model,
        universe,
        config_name,
        sl_value,
        pl_value,
        start_test_date,
        end_date,
        root_dir,
        log_fn=None,
):
    config_dir = os.path.join(
        root_dir,
        task_type,
        test_model,
        "model_params",
        universe,
        config_name,
        f"sl_{sl_value}",
        f"pl_{pl_value}",
    )

    config_path = os.path.join(config_dir, "config_params.json")
    _log(log_fn, f"Reading configuration from: {config_path}")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config_params.json not found: {config_path}")

    with open(config_path, "r", encoding="utf-8-sig") as f:
        saved_args = json.load(f)

    # The test follows validation directly (same tickers as in training)
    check_sequential_test(start_test_date, saved_args["end_valid_date"])

    args_dict = saved_args.copy()
    args_dict["start_test_date"] = start_test_date
    args_dict["end_date"] = end_date
    args_dict["model_name"] = config_name
    args_dict["universe"] = universe
    args_dict["seq_len"] = sl_value
    args_dict["pred_len"] = pl_value
    args_dict["data_path"] = "./Evaluation/data"

    args = Namespace(**args_dict)

    _log(log_fn, "Creating output folders...")
    model_save_path, metrics_path = create_saving_path(args, task_type, test_model, root_dir)

    return args, model_save_path, metrics_path