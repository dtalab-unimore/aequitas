from Aequitas.script_runner import run_model_script


def dispatch_train(
    universe,
    task_type,
    model_name,
    config_name,
    sl_value,
    pl_value,
    start_train_date,
    end_train_date,
    start_valid_date,
    end_valid_date,
    log_fn,
):
    """
    `sl_value` is None for models without a configurable sequence length
    (see MODELS_WITHOUT_SEQUENCE_LENGTH in train_models.py).
    """
    args = [
        "--universe", universe,
        "--model_name", config_name,
        "--pred_len", str(pl_value),
        "--start_date", start_train_date,
        "--end_train_date", end_train_date,
        "--start_valid_date", start_valid_date,
        "--end_valid_date", end_valid_date,
    ]

    if sl_value is not None:
        args.extend(["--seq_len", str(sl_value)])

    if model_name.lower() == "hist":
        args.extend(["--model_name", "HIST", "--run_name", config_name])

    return run_model_script(f"{task_type}.{model_name}.train", args, log_fn)
