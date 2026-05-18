import gradio as gr
import json


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