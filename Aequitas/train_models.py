import re
import time

import gradio as gr

from Aequitas.data_load import read_configs
from Aequitas.script_runner import start_job, follow_job, get_job, cancel_job
from Aequitas.test_models import get_tickers_split
from Aequitas.train_model_dispatcher import dispatch_train
from Aequitas.ui import error_alert

YEAR_RE = re.compile(r"^\d{4}$")

MODELS_WITHOUT_SEQUENCE_LENGTH = {"hist", "finformer", "discoverplf"}


def _normalize_model_name(model_name):
    return str(model_name or "").strip().lower()


def model_uses_sequence_length(model_name) -> bool:
    return _normalize_model_name(model_name) not in MODELS_WITHOUT_SEQUENCE_LENGTH


def normalize_sequence_length_for_model(model_name, sl_value):
    if not model_uses_sequence_length(model_name):
        return None

    return sl_value

def get_models_train(task):
    config = read_configs(task)
    models = list(config.keys())

    return (
        gr.Dropdown(
            choices=models,
            value=None,
            interactive=bool(models),
        ),
        gr.update(
            value=None,
            interactive=False,
            placeholder="Select a model first",
        ),
        gr.update(interactive=False),
    )

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
        pl,
        start_train,
        end_train,
        start_valid,
        end_valid,
    ]

    if model_uses_sequence_length(train_model):
        required_values.append(sl)

    all_filled = all(_has_value(value) for value in required_values)

    return gr.update(interactive=all_filled)


def on_train_model_change(
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
    if _is_empty(train_model):
        sl_update = gr.update(
            value=None,
            interactive=False,
            placeholder="Select a model first",
        )

    elif not model_uses_sequence_length(train_model):
        sl_update = gr.update(
            value=None,
            interactive=False,
            placeholder="Not used by this model",
        )

    else:
        sl_update = gr.update(
            value=None,
            interactive=True,
            placeholder="Insert the sequence length",
        )

    # Dopo il cambio modello, il valore effettivo di sl è None.
    effective_sl = None

    button_update = toggle_train_button(
        train_universe,
        train_task,
        train_model,
        train_config,
        effective_sl,
        pl,
        start_train,
        end_train,
        start_valid,
        end_valid,
    )

    return sl_update, button_update


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
    required_values = [
        universe,
        task_type,
        model_name,
        config_name,
        pl_value,
        start_train_year,
        end_train_year,
        start_valid_year,
        end_valid_year,
    ]

    if model_uses_sequence_length(model_name):
        required_values.append(sl_value)

    if any(_is_empty(value) for value in required_values):
        yield (
            gr.update(value="Missing parameters."),
            "",
            gr.update(visible=False),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )
        return

    start_train_date = f"{int(start_train_year)}-01-01"
    end_train_date = f"{int(end_train_year)}-12-31"
    start_valid_date = f"{int(start_valid_year)}-01-01"
    end_valid_date = f"{int(end_valid_year)}-12-31"

    def job(log_fn):
        return dispatch_train(
            universe=universe,
            task_type=task_type,
            model_name=model_name,
            config_name=config_name,
            sl_value=normalize_sequence_length_for_model(model_name, sl_value),
            pl_value=pl_value,
            start_train_date=start_train_date,
            end_train_date=end_train_date,
            start_valid_date=start_valid_date,
            end_valid_date=end_valid_date,
            log_fn=log_fn,
        )

    form_values = {
        "universe": universe,
        "task_type": task_type,
        "model_name": model_name,
        "config_name": config_name,
        "sl_value": sl_value,
        "pl_value": pl_value,
        "start_train_year": start_train_year,
        "end_train_year": end_train_year,
        "start_valid_year": start_valid_year,
        "end_valid_year": end_valid_year,
    }

    job_id = start_job(job, "Training", meta=form_values)
    yield from follow_training(job_id)


# -------------------------------------------------
# Re-attaching to a run after a page reload
# -------------------------------------------------
# The job id of the latest run is kept in the browser (gr.BrowserState), the
# job itself in the server-side registry of script_runner.

def follow_training(job_id):
    """
    Stream a training job into the Train tab.

    Yields: train_logs, train_result, train_loading, train_job_state,
    train_button, stop_button.
    While the job is running the Train button is disabled and the Stop
    button is shown.
    """
    job = get_job(job_id)
    first = True
    for logs_text, result_text, running in follow_job(job_id):
        stopping = job.cancel_requested
        yield (
            gr.update(value=logs_text),
            result_text,
            gr.update(visible=running),
            job_id if first else gr.skip(),
            gr.update(interactive=not running),
            gr.update(
                visible=running,
                interactive=not stopping,
                value="Stopping..." if stopping else "Stop training",
            ),
        )
        first = False


def stop_training(job_id):
    """Interrupt the browser's running training job."""
    cancel_job(job_id)
    return gr.update(interactive=False, value="Stopping...")


def restore_training_view(job_id, run_shown):
    """
    When the Train tab is opened after a page reload, put the form, the
    tickers and the output section back as they were for the browser's last
    training run. Does nothing if this page already shows a run.

    Outputs: universe, task, model, config, sl, pl, start_train, end_train,
    start_valid, end_valid, train_output_section, tickers_info, tickers_count,
    tickers_download, tickers_list, resume_token, run_shown.

    Following the logs is not done here but by the `resume_token` change
    event, so this event returns immediately.
    """
    job = get_job(job_id)
    if job is None or run_shown:
        return (*(gr.update(),) * 15, gr.update(), run_shown)

    m = job.meta
    uses_sl = model_uses_sequence_length(m["model_name"])
    models = list(read_configs(m["task_type"]).keys())
    tickers_info, tickers_count, tickers_download, tickers_list = get_tickers_split(
        m["universe"], m["end_valid_year"]
    )

    return (
        gr.update(value=m["universe"]),
        gr.update(value=m["task_type"]),
        gr.update(choices=models, value=m["model_name"], interactive=True),
        gr.update(value=m["config_name"]),
        gr.update(
            value=m["sl_value"] if uses_sl else None,
            interactive=uses_sl,
            placeholder="Insert the sequence length" if uses_sl else "Not used by this model",
        ),
        gr.update(value=m["pl_value"]),
        gr.update(value=m["start_train_year"]),
        gr.update(value=m["end_train_year"]),
        gr.update(value=m["start_valid_year"]),
        gr.update(value=m["end_valid_year"]),
        gr.update(visible=True),
        tickers_info,
        tickers_count,
        tickers_download,
        tickers_list,
        f"{job_id}:{time.time()}",
        True,
    )


def resume_training(job_id):
    """After a page reload, re-attach the log box to the browser's last training run."""
    if get_job(job_id) is None:
        yield (gr.skip(),) * 6
        return

    yield from follow_training(job_id)


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

    if model_uses_sequence_length(train_model):
        if not _is_positive_int(sl):
            errors.append("Sequence length must be a positive integer.")
            sl_update = gr.update(value="")
    else:
        sl = None
        sl_update = gr.update(
            value=sl,
            interactive=False,
            placeholder="Not used by this model",
        )

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

        if end_train_y >= start_valid_y:
            errors.append("End Train Year must be before Start Validation Year.")
            end_train_update = gr.update(value="")
            start_valid_update = gr.update(value="")

    if errors:
        msg = error_alert("Invalid training parameters.", errors)

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
        yield (
            gr.skip(),
            gr.skip(),
            gr.update(visible=False),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )
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

    yield from result