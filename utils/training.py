"""
Shared pieces of the train.py contract.

train.py only sees the training and validation periods (the UI passes
--start_date, --end_train_date, --start_valid_date, --end_valid_date): it
trains, picks the best checkpoint on validation and saves the model, its
config_params.json and the validation metrics. The test is run separately by
test.py, on the period chosen in the Test tab.
"""
import json
import os

import pandas as pd

from .filter_constituents import filter_constituents_by_date


def first_test_day(end_valid_date: str) -> str:
    """The day after the validation period: the earliest date a test can start."""
    return (pd.Timestamp(end_valid_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def first_test_year(end_valid_date: str) -> int | None:
    """
    Year the Test tab must start from: the test follows validation directly,
    so it starts the day after end_valid_date. None when that day is not a
    1 January (the Test tab works in whole years).
    """
    day = first_test_day(end_valid_date)
    return int(day[:4]) if day.endswith("-01-01") else None


def check_sequential_test(start_test_date: str, end_valid_date: str) -> None:
    """
    The test must start right after validation: the tickers are the index
    constituents on that day, the same set the model was trained on.
    """
    expected = first_test_day(end_valid_date)
    if pd.Timestamp(start_test_date) != pd.Timestamp(expected):
        raise ValueError(
            f"The test must start right after validation: validation ends on {end_valid_date}, "
            f"so the test must start on {expected} (got {start_test_date})."
        )


def training_constituents(constituents: pd.DataFrame, end_valid_date: str) -> pd.DataFrame:
    """
    Constituents a model is trained on: those in the index right after the
    validation period, i.e. the ones a test right after validation would use.
    """
    return filter_constituents_by_date(constituents, first_test_day(end_valid_date))


def ticker_selection_date(args) -> str:
    """
    Date the universe constituents are taken at: the test start in test.py
    (load_configuration sets it), the day after validation in train.py.
    """
    return getattr(args, "start_test_date", None) or first_test_day(args.end_valid_date)


def save_valid_metrics(metrics: dict, model_save_path: str, seq_len, pred_len) -> str:
    """Write the validation metrics next to the model and return the file path."""
    path = os.path.join(model_save_path, f"val_metrics_sl{seq_len}_pl{pred_len}.json")
    with open(path, "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=4)
    return path


def training_result(model_path: str, valid_metrics: dict | None, valid_metrics_path: str | None) -> dict:
    """The JSON a train.py emits (see utils.logger.emit_result)."""
    return {
        "model_path": model_path,
        "valid_metrics": {k: float(v) for k, v in (valid_metrics or {}).items()},
        "valid_metrics_path": valid_metrics_path,
    }
