import os
import json
import traceback
from datetime import datetime
import queue
import threading
import time
import gradio as gr
import pandas as pd

from Gradio.test_model_dispatcher import dispatch_test
from utils.filter_constituents import filter_constituents_by_date

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

ALL_TASKS = ["Classification", "Regression", "Ranking"]
TEST_YEARS = [2020, 2021, 2022, 2023, 2024]

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
        if os.path.isdir(config_dir):
            configs.append(name)

    return sorted(configs)


def load_config_params(universe: str, task_type: str, model_name: str, config_name: str, sl_value, pl_value):
    if (not universe or not task_type or not model_name or not config_name or sl_value is None or pl_value is None):
        return None, ""

    config_path = os.path.join(
        get_model_params_base_dir(universe, task_type, model_name),
        config_name,
        f"sl_{sl_value}",
        f"pl_{pl_value}",
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

    return (
        gr.update(choices=tasks, value=None, interactive=bool(tasks)),  # task
        gr.update(choices=[], value=None, interactive=False),           # model
        gr.update(choices=[], value=None, interactive=False),           # config
        gr.update(choices=[], value=None, interactive=False),           # sl
        gr.update(choices=[], value=None, interactive=False),           # pl
        gr.update(choices=[], value=None, interactive=False),           # start year
        gr.update(choices=[], value=None, interactive=False),           # end year
        gr.update(value=""),                                           # config info
        gr.update(interactive=False),                                  # test button
    )


def on_task_change(universe: str, task_type: str):
    models = get_available_models(universe, task_type)

    return (
        gr.update(choices=models, value=None, interactive=bool(models)),  # model
        gr.update(choices=[], value=None, interactive=False),             # config
        gr.update(choices=[], value=None, interactive=False),             # sl
        gr.update(choices=[], value=None, interactive=False),             # pl
        gr.update(choices=[], value=None, interactive=False),             # start year
        gr.update(choices=[], value=None, interactive=False),             # end year
        gr.update(value=""),                                             # config info
        gr.update(interactive=False),                                    # test button
    )

def on_model_change(universe: str, task_type: str, model_name: str):
    configs = get_available_configs(universe, task_type, model_name)

    return (
        gr.update(choices=configs, value=None, interactive=bool(configs)),  # config
        gr.update(choices=[], value=None, interactive=False),               # sl
        gr.update(choices=[], value=None, interactive=False),               # pl
        gr.update(choices=[], value=None, interactive=False),               # start year
        gr.update(choices=[], value=None, interactive=False),               # end year
        gr.update(value=""),                                               # config info
        gr.update(interactive=False),                                      # test button
    )


def on_config_change(universe: str, task_type: str, model_name: str, config_name: str):
    sl_values = get_available_sl(universe, task_type, model_name, config_name)

    return (
        gr.update(choices=sl_values, value=None, interactive=bool(sl_values)),  # sl
        gr.update(choices=[], value=None, interactive=False),                   # pl
        gr.update(choices=[], value=None, interactive=False),                   # start year
        gr.update(choices=[], value=None, interactive=False),                   # end year
        gr.update(value=""),                                                   # config info
        gr.update(interactive=False),                                          # test button
    )

def on_sl_change(universe: str, task_type: str, model_name: str, config_name: str, sl_value):
    pl_values = get_available_pl(universe, task_type, model_name, config_name, sl_value)

    return (
        gr.update(choices=pl_values, value=None, interactive=bool(pl_values)),  # pl
        gr.update(choices=[], value=None, interactive=False),                   # start year
        gr.update(choices=[], value=None, interactive=False),                   # end year
        gr.update(value=""),                                                   # config info
        gr.update(interactive=False),                                          # test button
    )


def on_pl_change(universe: str, task_type: str, model_name: str, config_name: str, sl_value, pl_value):
    params, config_text = load_config_params(
        universe,
        task_type,
        model_name,
        config_name,
        sl_value,
        pl_value,
    )

    if not params:
        return (
            gr.update(value=config_text, visible=True),          # config info
            gr.update(choices=[], value=None, interactive=False), # start year
            gr.update(choices=[], value=None, interactive=False), # end year
            gr.update(interactive=False),                        # test button
        )

    return (
        gr.update(value=config_text, visible=True),              # config info
        gr.update(choices=TEST_YEARS, value=None, interactive=True),  # start year
        gr.update(choices=[], value=None, interactive=False),    # end year
        gr.update(interactive=False),                            # test button
    )

def extract_numeric_suffix(folder_name: str, prefix: str):
    if not folder_name.startswith(prefix):
        return None
    try:
        return int(folder_name[len(prefix):])
    except ValueError:
        return None


def get_available_sl(universe: str, task_type: str, model_name: str, config_name: str):
    if not universe or not task_type or not model_name or not config_name:
        return []

    base_dir = os.path.join(
        get_model_params_base_dir(universe, task_type, model_name),
        config_name
    )

    if not os.path.isdir(base_dir):
        return []

    sl_values = []
    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        value = extract_numeric_suffix(name, "sl_")
        if os.path.isdir(full_path):
            sl_values.append(value)

    return sorted(sl_values)


def get_available_pl(universe: str, task_type: str, model_name: str, config_name: str, sl_value):
    if not universe or not task_type or not model_name or not config_name or sl_value is None:
        return []

    sl_dir = os.path.join(
        get_model_params_base_dir(universe, task_type, model_name),
        config_name,
        f"sl_{sl_value}"
    )

    if not os.path.isdir(sl_dir):
        return []

    pl_values = []
    for name in os.listdir(sl_dir):
        full_path = os.path.join(sl_dir, name)
        value = extract_numeric_suffix(name, "pl_")
        if os.path.isdir(full_path) and value is not None:
            pl_values.append(value)

    return sorted(pl_values)


def update_end_year_choices(start_year):
    all_years = [2020, 2021, 2022, 2023, 2024]

    if start_year is None or start_year == "":
        return gr.update(
            choices=[],
            value=None,
            interactive=False,
        )

    valid_end_years = [y for y in all_years if y >= int(start_year)]

    return gr.update(
        choices=valid_end_years,
        value=None,
        interactive=bool(valid_end_years),
    )


def get_tickers(universe: str, start_test_year: str):
    if not universe or not start_test_year:
        return gr.update(
            visible=True,
            value="""
            <div class="ticker-card ticker-warning">
                Please select both universe and start year before running the test.
            </div>
            """
        )

    start_test_date = f"{int(start_test_year)}-01-01"

    constituents = pd.read_csv(
        f"./Evaluation/data/constituents/eodhd/{universe}.csv"
    )

    tickers = filter_constituents_by_date(constituents, start_test_date)
    ticker_list = tickers["EODHD"].tolist()

    preview_tickers = ticker_list[:40]
    remaining = max(0, len(ticker_list) - len(preview_tickers))

    chips = "".join(
        f'<span class="ticker-chip">{ticker}</span>'
        for ticker in preview_tickers
    )

    more_text = (
        f'<span class="ticker-more">+{remaining} more</span>'
        if remaining > 0
        else ""
    )

    html = f"""
    <div class="ticker-card">
        <div class="ticker-title">Selected preprocessed tickers</div>
        <div class="ticker-subtitle">
            The test will be run on the tickers already preprocessed for
            <b>{universe}</b>, filtered from <b>{start_test_date}</b>.
        </div>
        <div class="ticker-count">
            {len(ticker_list)} tickers selected
        </div>
        <div class="ticker-list">
            {chips}
            {more_text}
        </div>
    </div>
    """

    return gr.update(
        visible=True,
        value=html,
    )


def run_test_with_logs(universe, task_type, test_model, config_name, sl_value, pl_value, start_year, end_year):
    logs = []
    log_queue = queue.Queue()
    done_queue = queue.Queue()

    def push_log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        log_queue.put(f"[{ts}] {msg}")

    def current_logs():
        return "\n".join(logs)

    if None in [universe, task_type, test_model, config_name, sl_value, pl_value, start_year, end_year]:
        yield gr.update(value="Missing parameters."), ""
        return

    start_test_date = f"{int(start_year)}-01-01"
    end_date = f"{int(end_year)}-12-31"

    def worker():
        try:
            push_log("Start test...")

            output = dispatch_test(
                universe=universe,
                task_type=task_type,
                test_model=test_model,
                config_name=config_name,
                sl_value=int(sl_value),
                pl_value=int(pl_value),
                start_test_date=start_test_date,
                end_date=end_date,
                log_fn=push_log,
            )

            done_queue.put(("ok", output))

        except Exception:
            push_log("Error during the test.")
            done_queue.put(("error", traceback.format_exc()))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    result_text = ""

    while True:

        if not log_queue.empty():
            logs.append(log_queue.get())

        yield gr.update(value=current_logs()), result_text

        if not done_queue.empty() and log_queue.empty():
            status, payload = done_queue.get()

            if status == "ok":
                logs.append("[OK] Test completato.")
                result_text = json.dumps(payload, indent=2, ensure_ascii=False)
            else:
                result_text = payload

            yield gr.update(value=current_logs()), result_text
            break

        time.sleep(0.10)