import json
import os
from argparse import Namespace


def _log(log_fn, msg):
    if log_fn is not None:
        log_fn(msg)

def create_saving_path(args, task_type, test_model, project_root):
    model_save_path = f"model_params/{args.universe}/{args.model_name}/sl_{args.seq_len}/pl_{args.pred_len}"
    metrics_path = f"results/{args.universe}/{args.model_name}/{args.seed}/y{args.start_test_date.split('-')[0]}_y{args.end_date.split('-')[0]}"

    model_save_path = os.path.join(project_root, task_type, test_model, model_save_path)
    metrics_path = os.path.join(project_root, task_type, test_model, metrics_path)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)

    if not os.path.exists(metrics_path):
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
    _log(log_fn, f"Lettura configurazione da: {config_path}")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config_params.json non trovato: {config_path}")

    with open(config_path, "r", encoding="utf-8-sig") as f:
        saved_args = json.load(f)

    args_dict = saved_args.copy()
    args_dict["start_test_date"] = start_test_date
    args_dict["end_date"] = end_date
    args_dict["model_name"] = config_name
    args_dict["universe"] = universe
    args_dict["seq_len"] = sl_value
    args_dict["pred_len"] = pl_value
    args_dict["data_path"] = "./Evaluation/data"

    args = Namespace(**args_dict)

    _log(log_fn, "Creazione dei path di salvataggio...")
    model_save_path, metrics_path = create_saving_path(args, task_type, test_model, root_dir)

    return args, model_save_path, metrics_path