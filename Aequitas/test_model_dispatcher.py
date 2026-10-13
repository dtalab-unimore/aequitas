from Aequitas.script_runner import run_model_script


def dispatch_test(
    universe,
    task_type,
    test_model,
    config_name,
    sl_value,
    pl_value,
    start_test_date,
    end_date,
    log_fn,
):
    args = [
        "--universe", universe,
        "--config_name", config_name,
        "--sl_value", str(sl_value),
        "--pl_value", str(pl_value),
        "--start_test_date", start_test_date,
        "--end_date", end_date,
        "--task_type", task_type,
        "--model_name", test_model,
    ]

    return run_model_script(f"{task_type}.{test_model}.test", args, log_fn)
