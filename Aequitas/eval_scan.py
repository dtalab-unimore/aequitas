"""
Scan available .pkl result files and provide filtering / lookup utilities
used by both the form cascade logic and the evaluation runner.
"""
import re

from Aequitas.eval_config import (
    FINBENCH_RESULTS_ROOT,
    SEED,
    YEAR_DIR_RE,
    PKL_FILE_RE,
)

# -------------------------------------------------
# Generic column-name helpers (shared with eval_run / eval_output)
# -------------------------------------------------

def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_column(df, aliases: list):
    """Return the first df column whose normalised name matches any alias."""
    normalized = {
        normalize_column_name(col): col
        for col in df.columns
    }
    for alias in aliases:
        key = normalize_column_name(alias)
        if key in normalized:
            return normalized[key]
    return None


# -------------------------------------------------
# Generic list helpers
# -------------------------------------------------

def as_list(value) -> list:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple, set)):
        return [v for v in value if v is not None and v != ""]
    return [value]


def is_not_empty(value) -> bool:
    return len(as_list(value)) > 0


# -------------------------------------------------
# Length-pair formatting / parsing
# -------------------------------------------------

def format_length_pair(sequence_length, prediction_length) -> str:
    return f"SL={int(sequence_length)} | PL={int(prediction_length)}"


def parse_length_pair(value):
    """Parse 'SL=8 | PL=10' → (8, 10), or None on failure."""
    if value is None or value == "":
        return None
    match = re.search(
        r"SL\s*=\s*(?P<sl>\d+)\s*\|\s*PL\s*=\s*(?P<pl>\d+)",
        str(value),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return int(match.group("sl")), int(match.group("pl"))


def unique_length_pair_choices(records: list) -> list:
    pairs = sorted(
        {(int(r["sequence_length"]), int(r["prediction_length"])) for r in records}
    )
    return [format_length_pair(sl, pl) for sl, pl in pairs]


def filter_records_by_length_pairs(records: list, length_pairs) -> list:
    selected_pairs = set()
    for value in as_list(length_pairs):
        parsed = parse_length_pair(value)
        if parsed is not None:
            selected_pairs.add(parsed)
    if not selected_pairs:
        return records
    return [
        r for r in records
        if (int(r["sequence_length"]), int(r["prediction_length"])) in selected_pairs
    ]


# -------------------------------------------------
# Record filtering helpers
# -------------------------------------------------

def unique_values(records: list, key: str) -> list:
    return sorted({r[key] for r in records})


def value_if_valid(value, choices):
    if value in choices:
        return value
    if value is not None and str(value) in [str(c) for c in choices]:
        for c in choices:
            if str(c) == str(value):
                return c
    return None


def values_if_valid(values, choices) -> list:
    selected = as_list(values)
    by_string = {str(choice): choice for choice in choices}
    return [by_string[str(v)] for v in selected if str(v) in by_string]


def filter_records(records: list, **filters) -> list:
    filtered = records
    for key, value in filters.items():
        values = as_list(value)
        if not values:
            continue
        allowed = {str(v) for v in values}
        filtered = [r for r in filtered if str(r[key]) in allowed]
    return filtered


# -------------------------------------------------
# PKL scanner
# -------------------------------------------------

def scan_available_pkls() -> list:
    """
    Walk the results directory tree and return a list of record dicts for
    every valid .pkl file found:

        <type>/<model>/results/<universe>/<configuration>/<SEED>/
        y<start>_y<end>/results_sl<sl>_pl<pl>.pkl
    """
    records = []

    if not FINBENCH_RESULTS_ROOT.exists():
        return records

    pattern = FINBENCH_RESULTS_ROOT.glob(
        f"*/*/results/*/*/{SEED}/y*_y*/results_sl*_pl*.pkl"
    )

    for pkl_path in pattern:
        try:
            rel = pkl_path.relative_to(FINBENCH_RESULTS_ROOT)
            parts = rel.parts

            # expected depth: type / model / results / universe /
            #                 configuration / seed / year_dir / file
            if len(parts) != 8:
                continue

            type_, model, results_literal, universe, configuration, \
                seed, year_dir, filename = parts

            if results_literal != "results":
                continue
            if seed != str(SEED):
                continue

            year_match = YEAR_DIR_RE.match(year_dir)
            file_match = PKL_FILE_RE.match(filename)

            if not year_match or not file_match:
                continue

            records.append(
                {
                    "task_type": type_,
                    "model": model,
                    "universe": universe,
                    "configuration": configuration,
                    "year_start": int(year_match.group("year_start")),
                    "year_end": int(year_match.group("year_end")),
                    "sequence_length": int(file_match.group("sl")),
                    "prediction_length": int(file_match.group("pl")),
                    "pkl_path": str(pkl_path),
                }
            )

        except Exception:
            continue

    records.sort(
        key=lambda r: (
            r["task_type"], r["model"], r["universe"], r["configuration"],
            r["year_start"], r["year_end"],
            r["sequence_length"], r["prediction_length"],
        )
    )

    return records