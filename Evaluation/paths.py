from pathlib import Path
import os


def get_project_root() -> Path:
    env_root = os.environ.get("FINBENCH_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    # Evaluation/paths.py -> Evaluation -> FINBENCH_DEMO
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = get_project_root()

EVALUATION_DIR = PROJECT_ROOT / "Evaluation"
DATA_DIR = EVALUATION_DIR / "data"

EVALUATION_RESULT_DIR = EVALUATION_DIR / "evaluation_result"
EVALUATION_RESULT_QUINTILE_DIR = EVALUATION_DIR / "evaluation_result_quintile"

DEFAULT_RESULTS_ROOT = PROJECT_ROOT


def get_price_path(universe: str) -> Path:
    return DATA_DIR / universe / f"{universe}.csv"


def get_config_path(filename: str = "config.json") -> Path:
    return EVALUATION_DIR / filename


def get_model_results_dir(
    prediction_type: str,
    model: str,
    universe: str,
    results_root: str | Path | None = None,
) -> Path:
    root = Path(results_root).expanduser().resolve() if results_root else DEFAULT_RESULTS_ROOT
    return root / prediction_type / model / "results" / universe


def get_evaluation_output_dir(
    universe: str,
    prediction_type: str,
    model: str,
    config: str,
    seed: int,
    initial_year: int,
    end_year: int,
    top_k: int,
    short_k: int,
    root_dir: str | Path | None = None,
) -> Path:
    base = Path(root_dir).expanduser().resolve() if root_dir else EVALUATION_RESULT_DIR

    return (
        base
        / universe
        / f"top{top_k}_short{short_k}"
        / prediction_type
        / model
        / config
        / str(seed)
        / f"y{initial_year}_y{end_year}"
    )


def get_quintile_output_dir(
    universe: str,
    prediction_type: str,
    model: str,
    config: str,
    seed: int,
    initial_year: int,
    end_year: int,
    root_dir: str | Path | None = None,
) -> Path:
    base = Path(root_dir).expanduser().resolve() if root_dir else EVALUATION_RESULT_QUINTILE_DIR

    return (
        base
        / universe
        / prediction_type
        / model
        / config
        / str(seed)
        / f"y{initial_year}_y{end_year}"
    )