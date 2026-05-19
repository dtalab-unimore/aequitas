import queue
import re
import threading
import traceback
import time
from datetime import datetime

import gradio as gr
import json

from Gradio.train_model_dispatcher import dispatch_train

YEAR_RE = re.compile(r"^\d{4}$")

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

def _is_empty(value):
    return value is None or str(value).strip() == ""


def _is_positive_int(value):
    value = str(value or "").strip()
    return value.isdigit() and int(value) > 0


def _is_year(value):
    value = str(value or "").strip()
    return bool(YEAR_RE.match(value))


def validate_train_inputs_soft(
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
    errors = []

    sl_update = gr.skip()
    pl_update = gr.skip()
    start_train_update = gr.skip()
    end_train_update = gr.skip()
    start_valid_update = gr.skip()
    end_valid_update = gr.skip()

    required_fields = {
        "Universe": train_universe,
        "Task Type": train_task,
        "Model": train_model,
        "Configuration": train_config,
    }

    for field_name, value in required_fields.items():
        if _is_empty(value):
            errors.append(f"{field_name} is required.")

    if not _is_positive_int(sl):
        errors.append("Sequence length must be a positive integer.")
        sl_update = gr.update(value="")

    if not _is_positive_int(pl):
        errors.append("Prediction length must be a positive integer.")
        pl_update = gr.update(value="")

    if not _is_year(start_train):
        errors.append("Start Train Year must be a 4-digit year.")
        start_train_update = gr.update(value="")

    if not _is_year(end_train):
        errors.append("End Train Year must be a 4-digit year.")
        end_train_update = gr.update(value="")

    if not _is_year(start_valid):
        errors.append("Start Validation Year must be a 4-digit year.")
        start_valid_update = gr.update(value="")

    if not _is_year(end_valid):
        errors.append("End Validation Year must be a 4-digit year.")
        end_valid_update = gr.update(value="")

    if not errors:
        start_train_y = int(str(start_train).strip())
        end_train_y = int(str(end_train).strip())
        start_valid_y = int(str(start_valid).strip())
        end_valid_y = int(str(end_valid).strip())

        if start_train_y > end_train_y:
            errors.append("Start Train Year must be less than or equal to End Train Year.")
            start_train_update = gr.update(value="")
            end_train_update = gr.update(value="")

        if start_valid_y > end_valid_y:
            errors.append("Start Validation Year must be less than or equal to End Validation Year.")
            start_valid_update = gr.update(value="")
            end_valid_update = gr.update(value="")

        if end_train_y > start_valid_y:
            errors.append("End Train Year must be before Start Validation Year.")
            end_train_update = gr.update(value="")
            start_valid_update = gr.update(value="")

    if errors:
        msg = """
        <div style="
            width: 100%;
            display: flex;
            justify-content: center;
            margin: 10px 0;
        ">
            <div style="
                display: inline-block;
                width: fit-content;
                max-width: 100%;
                padding: 10px 14px;
                border: 1px solid #fecaca;
                background: #fef2f2;
                color: #991b1b;
                border-radius: 10px;
                font-size: 14px;
                line-height: 1.4;
                box-sizing: border-box;
                text-align: left;
            ">
                <b>Invalid training parameters.</b>
                <ul style="margin: 6px 0 0 18px; padding: 0;">
                    {}
                </ul>
            </div>
        </div>
        """.format("".join(f"<li>{e}</li>" for e in errors))

        return (
            False,                          # train_validation_state
            gr.update(visible=False),       # train_output_section
            gr.update(value=""),            # train_logs
            gr.update(value=None),          # train_result
            sl_update,
            pl_update,
            start_train_update,
            end_train_update,
            start_valid_update,
            end_valid_update,
            gr.update(value=msg, visible=True),  # train_validation_msg
            gr.update(interactive=False),        # train_button
        )

    return (
        True,                           # train_validation_state
        gr.update(visible=True),        # train_output_section
        gr.update(value=""),            # train_logs
        gr.update(value=None),          # train_result
        gr.skip(),                      # sl
        gr.skip(),                      # pl
        gr.skip(),                      # start_train
        gr.skip(),                      # end_train
        gr.skip(),                      # start_valid
        gr.skip(),                      # end_valid
        gr.update(value="", visible=False),  # train_validation_msg
        gr.update(interactive=True),         # train_button
    )


def train_model_with_logs_if_valid(
    is_valid,
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
    if not is_valid:
        yield gr.skip(), gr.skip()
        return

    result = train_model_with_logs(
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
    )

    if hasattr(result, "__iter__") and not isinstance(result, (tuple, list, dict, str)):
        yield from result
    else:
        yield result