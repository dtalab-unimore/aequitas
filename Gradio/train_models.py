import queue
import threading
import traceback
import time
from datetime import datetime

import gradio as gr
import json

from Gradio.train_model_dispatcher import dispatch_train


def get_models():
    with open('./Gradio/configs/model_to_data.json', 'r') as f:
        config = json.load(f)

    return config.keys()

def _has_value(value):
    if value is None:
        return False

    if isinstance(value, str):
        return value.strip() != ""

    return True


def toggle_train_button(
    train_universe,
    train_task,
    train_model,
    train_config,
    sl,
    pl,
    start_train,
    end_train,
    start_valid,
    end_valid,
):
    required_values = [
        train_universe,
        train_task,
        train_model,
        train_config,
        sl,
        pl,
        start_train,
        end_train,
        start_valid,
        end_valid,
    ]

    all_filled = all(_has_value(value) for value in required_values)

    return gr.update(interactive=all_filled)

def train_model_with_logs(
    universe,
    task_type,
    model_name,
    config_name,
    sl_value,
    pl_value,
    start_train_year,
    end_train_year,
    start_valid_year,
    end_valid_year,
):
    logs = []
    log_queue = queue.Queue()
    done_queue = queue.Queue()

    def push_log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        log_queue.put(f"[{ts}] {msg}")

    def current_logs():
        return "\n".join(logs)

    if None in [universe, task_type, model_name, config_name, sl_value, pl_value, start_train_year, end_train_year, start_valid_year, end_valid_year]:
        yield gr.update(value="Missing parameters."), ""
        return

    start_train_date = f"{int(start_train_year)}-01-01"
    end_train_date = f"{int(end_train_year)}-12-31"
    start_valid_date = f"{int(start_valid_year)}-01-01"
    end_valid_date = f"{int(end_valid_year)}-12-31"

    def worker():
        try:
            push_log("Start training...")

            output = dispatch_train(
                universe=universe,
                task_type=task_type,
                model_name=model_name,
                config_name=config_name,
                sl_value=sl_value,
                pl_value=pl_value,
                start_train_date=start_train_date,
                end_train_date=end_train_date,
                start_valid_date=start_valid_date,
                end_valid_date=end_valid_date,
                log_fn=push_log
            )

            done_queue.put(("ok", output))

        except Exception:
            push_log("Error during the training.")
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
                logs.append("[OK] Train completed.")
                result_text = json.dumps(payload, indent=2, ensure_ascii=False)
            else:
                result_text = payload

            yield gr.update(value=current_logs()), result_text
            break

        time.sleep(0.10)