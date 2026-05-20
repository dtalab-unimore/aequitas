"""
Cascade-dropdown update logic for the Evaluation tab form.
"""
import gradio as gr

from Gradio.eval_config import EVAL_MODES, TOP_SHORT_PAIRS_BY_UNIVERSE
from Gradio.eval_scan import (
    scan_available_pkls,
    filter_records,
    unique_values,
    unique_length_pair_choices,
    value_if_valid,
    values_if_valid,
    is_not_empty,
    as_list,
)

DATASET_LABELS = {
    "sp500": "[sp500] Standard & Poor's 500",
    "dji": "[dji] Dow Jones Industrial Average",
    "nasdaq100": "[nasdaq100] NASDAQ-100",
    "sxxp": "[sxxp] STOXX Europe 600",
    "sx5e": "[sx5e] EURO STOXX 50",
}


def universe_dropdown_choices(universes):
    return [
        (DATASET_LABELS.get(str(u).lower(), str(u)), u)
        for u in universes
    ]

# -------------------------------------------------
# Field metadata
# -------------------------------------------------

MULTISELECT_FIELDS = {"model", "configuration", "length_pair"}

FIELD_ORDER = [
    "task_type",
    "model",
    "universe",
    "configuration",
    "year_start",
    "year_end",
    "length_pair",
    "top_k",
    "short_k",
]


def empty_value_for_field(field):
    return [] if field in MULTISELECT_FIELDS else None


# -------------------------------------------------
# Top / short K helpers
# -------------------------------------------------

def get_top_short_choices(universe, selected_top_k=None):
    pairs = TOP_SHORT_PAIRS_BY_UNIVERSE.get(str(universe).lower(), [])
    top_k_choices = sorted({top for top, _ in pairs})
    if selected_top_k is None:
        short_k_choices = []
    else:
        short_k_choices = sorted(
            {short for top, short in pairs if int(top) == int(selected_top_k)}
        )
    return top_k_choices, short_k_choices


def selected_top_short_pairs_for_universe(universe, top_k, short_k):
    selected_pairs = {
        (int(top), int(short))
        for top in as_list(top_k)
        for short in as_list(short_k)
    }
    available_pairs = set(TOP_SHORT_PAIRS_BY_UNIVERSE.get(str(universe).lower(), []))
    return sorted(selected_pairs & available_pairs)


# -------------------------------------------------
# Downstream reset
# -------------------------------------------------

def reset_downstream_values(changed_field: str, values: dict) -> dict:
    """Reset every field that comes after *changed_field* in FIELD_ORDER."""
    if changed_field == "eval_mode":
        for field in FIELD_ORDER:
            values[field] = empty_value_for_field(field)
        return values

    if changed_field not in FIELD_ORDER:
        return values

    changed_idx = FIELD_ORDER.index(changed_field)
    for field in FIELD_ORDER[changed_idx + 1:]:
        values[field] = empty_value_for_field(field)

    return values


# -------------------------------------------------
# Main cascade update function (bound to every form input)
# -------------------------------------------------

def update_eval_form(
    eval_mode,
    task_type,
    model,
    universe,
    configuration,
    year_start,
    year_end,
    length_pair,
    top_k,
    short_k,
    changed_field,
):
    records = scan_available_pkls()

    values = {
        "task_type": task_type,
        "model": model,
        "universe": universe,
        "configuration": configuration,
        "year_start": year_start,
        "year_end": year_end,
        "length_pair": length_pair,
        "top_k": top_k,
        "short_k": short_k,
    }

    values = reset_downstream_values(changed_field, values)

    form_visible = eval_mode in EVAL_MODES
    top_short_visible = eval_mode in ["EOY", "METRICS"]

    if not form_visible:
        return [
            gr.update(visible=False),
            gr.update(choices=[], value=None, interactive=False),   # task_type
            gr.update(choices=[], value=[], interactive=False),     # model
            gr.update(choices=[], value=None, interactive=False),   # universe
            gr.update(choices=[], value=[], interactive=False),     # configuration
            gr.update(choices=[], value=None, interactive=False),   # year_start
            gr.update(choices=[], value=None, interactive=False),   # year_end
            gr.update(choices=[], value=[], interactive=False),     # length_pair
            gr.update(visible=False),
            gr.update(choices=[], value=None, interactive=False),   # top_k
            gr.update(choices=[], value=None, interactive=False),   # short_k
            gr.update(interactive=False),
        ]

    # 1. Task type
    task_type_choices = unique_values(records, "task_type")
    values["task_type"] = value_if_valid(values["task_type"], task_type_choices)

    # 2. Model
    model_records = filter_records(records, task_type=values["task_type"])
    model_choices = (
        unique_values(model_records, "model")
        if is_not_empty(values["task_type"]) else []
    )
    values["model"] = values_if_valid(values["model"], model_choices)

    # 3. Universe
    universe_records = filter_records(records,
                                      task_type=values["task_type"],
                                      model=values["model"])
    universe_choices = (
        unique_values(universe_records, "universe")
        if is_not_empty(values["model"]) else []
    )

    values["universe"] = value_if_valid(values["universe"], universe_choices)

    universe_choices_display = universe_dropdown_choices(universe_choices)

    # 4. Configuration
    configuration_records = filter_records(records,
                                           task_type=values["task_type"],
                                           model=values["model"],
                                           universe=values["universe"])
    configuration_choices = (
        unique_values(configuration_records, "configuration")
        if is_not_empty(values["universe"]) else []
    )
    values["configuration"] = values_if_valid(values["configuration"],
                                              configuration_choices)

    # 5. Year start
    year_start_records = filter_records(records,
                                        task_type=values["task_type"],
                                        model=values["model"],
                                        universe=values["universe"],
                                        configuration=values["configuration"])
    year_start_choices = (
        unique_values(year_start_records, "year_start")
        if is_not_empty(values["configuration"]) else []
    )
    values["year_start"] = value_if_valid(values["year_start"], year_start_choices)

    # 6. Year end
    year_end_records = filter_records(records,
                                      task_type=values["task_type"],
                                      model=values["model"],
                                      universe=values["universe"],
                                      configuration=values["configuration"],
                                      year_start=values["year_start"])
    year_end_choices = (
        unique_values(year_end_records, "year_end")
        if is_not_empty(values["year_start"]) else []
    )
    values["year_end"] = value_if_valid(values["year_end"], year_end_choices)

    # 7. Length pairs
    length_pair_records = filter_records(records,
                                         task_type=values["task_type"],
                                         model=values["model"],
                                         universe=values["universe"],
                                         configuration=values["configuration"],
                                         year_start=values["year_start"],
                                         year_end=values["year_end"])
    length_pair_choices = (
        unique_length_pair_choices(length_pair_records)
        if is_not_empty(values["year_end"]) else []
    )
    values["length_pair"] = values_if_valid(values["length_pair"], length_pair_choices)

    # 8. Top K / Short K
    top_k_choices = []
    short_k_choices = []

    if top_short_visible and is_not_empty(values["length_pair"]):
        top_k_choices, _ = get_top_short_choices(values["universe"], values["top_k"])
        values["top_k"] = value_if_valid(values["top_k"], top_k_choices)

        if is_not_empty(values["top_k"]):
            _, short_k_choices = get_top_short_choices(values["universe"],
                                                       values["top_k"])
            values["short_k"] = value_if_valid(values["short_k"], short_k_choices)
        else:
            values["short_k"] = None
            short_k_choices = []
    else:
        values["top_k"] = None
        values["short_k"] = None

    # Completeness check
    base_complete = all(
        is_not_empty(values[f])
        for f in ["task_type", "model", "universe", "configuration",
                  "year_start", "year_end", "length_pair"]
    )
    if eval_mode in ["EOY", "METRICS"]:
        complete = (base_complete
                    and is_not_empty(values["top_k"])
                    and is_not_empty(values["short_k"]))
    else:
        complete = base_complete

    return [
        gr.update(visible=True),

        gr.update(choices=task_type_choices,
                  value=values["task_type"],
                  interactive=bool(task_type_choices)),
        gr.update(choices=model_choices,
                  value=values["model"],
                  interactive=is_not_empty(values["task_type"]) and bool(model_choices)),
        gr.update(
            choices=universe_choices_display,
            value=values["universe"],
            interactive=is_not_empty(values["model"]) and bool(universe_choices),
        ),
        gr.update(choices=configuration_choices,
                  value=values["configuration"],
                  interactive=is_not_empty(values["universe"]) and bool(configuration_choices)),
        gr.update(choices=year_start_choices,
                  value=values["year_start"],
                  interactive=is_not_empty(values["configuration"]) and bool(year_start_choices)),
        gr.update(choices=year_end_choices,
                  value=values["year_end"],
                  interactive=is_not_empty(values["year_start"]) and bool(year_end_choices)),
        gr.update(choices=length_pair_choices,
                  value=values["length_pair"],
                  interactive=is_not_empty(values["year_end"]) and bool(length_pair_choices)),

        gr.update(visible=top_short_visible),

        gr.update(choices=top_k_choices,
                  value=values["top_k"],
                  interactive=(top_short_visible
                               and is_not_empty(values["length_pair"])
                               and bool(top_k_choices))),
        gr.update(choices=short_k_choices,
                  value=values["short_k"],
                  interactive=(top_short_visible
                               and is_not_empty(values["top_k"])
                               and bool(short_k_choices))),

        gr.update(interactive=complete),
    ]