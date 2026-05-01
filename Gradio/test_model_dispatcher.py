import importlib

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
    module_name = f"{task_type}.{test_model}.test"

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ValueError(
            f"Impossibile importare il runner del modello: {module_name}"
        ) from e

    if not hasattr(module, "run_test"):
        raise ValueError(f"Il modulo {module_name} non contiene run_test(...)")

    return module.run_test(
        task_type=task_type,
        test_model=test_model,
        universe=universe,
        config_name=config_name,
        sl_value=sl_value,
        pl_value=pl_value,
        start_test_date=start_test_date,
        end_date=end_date,
        log_fn=log_fn,
    )