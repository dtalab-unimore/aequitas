import os
import json
import gradio as gr

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

ALL_TASKS = ["Classification", "Regression", "Ranking"]


def get_universe_options():
    universes = set()

    for task in ALL_TASKS:
        task_dir = os.path.join(MODEL_ROOT, task)
        if not os.path.isdir(task_dir):
            continue

        for model_name in os.listdir(task_dir):
            model_dir = os.path.join(task_dir, model_name)
            params_dir = os.path.join(model_dir, "model_params")
            if not os.path.isdir(params_dir):
                continue

            for universe in os.listdir(params_dir):
                universe_dir = os.path.join(params_dir, universe)
                if os.path.isdir(universe_dir):
                    universes.add(universe)

    return sorted(universes)


def get_model_params_base_dir(universe: str, task_type: str, model_name: str) -> str:
    return os.path.join(
        MODEL_ROOT,
        task_type,
        model_name,
        "model_params",
        universe,
    )


def get_available_tasks(universe: str):
    if not universe:
        return []

    available = []
    for task in ALL_TASKS:
        task_dir = os.path.join(MODEL_ROOT, task)
        if not os.path.isdir(task_dir):
            continue

        for model_name in os.listdir(task_dir):
            universe_dir = os.path.join(task_dir, model_name, "model_params", universe)
            if os.path.isdir(universe_dir):
                available.append(task)
                break

    return sorted(available)


def get_available_models(universe: str, task_type: str):
    if not universe or not task_type:
        return []

    task_dir = os.path.join(MODEL_ROOT, task_type)
    if not os.path.isdir(task_dir):
        return []

    models = []
    for model_name in os.listdir(task_dir):
        universe_dir = os.path.join(task_dir, model_name, "model_params", universe)
        if os.path.isdir(universe_dir):
            models.append(model_name)

    return sorted(models)


def get_available_configs(universe: str, task_type: str, model_name: str):
    if not universe or not task_type or not model_name:
        return []

    base_dir = get_model_params_base_dir(universe, task_type, model_name)
    if not os.path.isdir(base_dir):
        return []

    configs = []
    for name in os.listdir(base_dir):
        config_dir = os.path.join(base_dir, name)
        config_file = os.path.join(config_dir, "config_params.json")
        if os.path.isdir(config_dir) and os.path.isfile(config_file):
            configs.append(name)

    return sorted(configs)


def load_config_params(universe: str, task_type: str, model_name: str, config_name: str):
    if not universe or not task_type or not model_name or not config_name:
        return None, ""

    config_path = os.path.join(
        get_model_params_base_dir(universe, task_type, model_name),
        config_name,
        "config_params.json",
    )

    if not os.path.isfile(config_path):
        return None, f"config_params.json non trovato: {config_path}"

    try:
        with open(config_path, "r", encoding="utf-8-sig") as f:
            raw_text = f.read()

        if not raw_text.strip():
            return None, f"config_params.json vuoto: {config_path}"

        params = json.loads(raw_text)
        return params, json.dumps(params, indent=2, ensure_ascii=False)

    except Exception as e:
        preview = raw_text[:200] if "raw_text" in locals() else ""
        return None, (
            f"Errore nella lettura di {config_path}: {e}\n\n"
            f"Prime 200 chars lette:\n{repr(preview)}"
        )


def on_universe_change(universe: str):
    tasks = get_available_tasks(universe)
    first_task = None

    return (
        gr.update(choices=tasks, value=first_task, interactive=bool(tasks))
    )


def on_task_change(universe: str, task_type: str):
    models = get_available_models(universe, task_type)

    return (
        gr.update(choices=models, value=None, interactive=bool(models))
    )


def on_model_change(universe: str, task_type: str, model_name: str):
    configs = get_available_configs(universe, task_type, model_name)

    return (
        gr.update(choices=configs, value=None, interactive=bool(configs))
    )


def on_config_change(universe: str, task_type: str, model_name: str, config_name: str):
    params, config_text = load_config_params(universe, task_type, model_name, config_name)

    if not params:
        return (
            gr.update(value=config_text, visible=True),
            gr.update(visible=False),
            "",
            "",
        )

    sl = params.get("sl", "")
    pl = params.get("pl", "")

    return (
        gr.update(value=config_text, visible=True),
        gr.update(visible=True),
        str(sl),
        str(pl),
    )

def update_end_year_choices(start_year):
    all_years = [2020, 2021, 2022, 2023, 2024]

    if start_year is None or start_year == "":
        return gr.update(
            choices=[],
            value=None,
            interactive=False,
        )

    valid_end_years = [y for y in all_years if y > int(start_year)]

    return gr.update(
        choices=valid_end_years,
        value=None,
        interactive=bool(valid_end_years),
    )