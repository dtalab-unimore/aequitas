import re
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------------------
# Plot layout
# -------------------------------------------------
MAX_PLOT_COLUMNS = 3
FIXED_SUBPLOT_WIDTH = 4.2
FIXED_SUBPLOT_HEIGHT_EOY = 2.55
FIXED_SUBPLOT_HEIGHT_QUINTILE = 2.65

# -------------------------------------------------
# OpenAI
# -------------------------------------------------
OPENAI_EXPLANATION_MODEL = os.getenv("OPENAI_EXPLANATION_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(f"Using OpenAI model: {OPENAI_EXPLANATION_MODEL}")

# -------------------------------------------------
# Paths
# -------------------------------------------------
# eval_config.py lives in:
#   FINBENCH_DEMO/Gradio/eval_config.py
# so parents[0] = Gradio, parents[1] = FINBENCH_DEMO
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[1]

FINBENCH_RESULTS_ROOT = PROJECT_ROOT
EVALUATION_DIR = PROJECT_ROOT / "Evaluation"
METRICS_EXTRACTION_SCRIPT = EVALUATION_DIR / "metrics_extraction.py"

# -------------------------------------------------
# Evaluation
# -------------------------------------------------
SEED = 42

EVAL_MODES = ["EOY", "METRICS", "QUINTILE"]

EVAL_TO_CMD = {
    "EOY": "eoy",
    "METRICS": "metrics",
    "QUINTILE": "quintile",
}

# -------------------------------------------------
# Regex
# -------------------------------------------------
YEAR_DIR_RE = re.compile(
    r"^y(?P<year_start>\d{4})_y(?P<year_end>\d{4})$"
)

PKL_FILE_RE = re.compile(
    r"^results_sl(?P<sl>\d+)_pl(?P<pl>\d+)\.pkl$"
)

# -------------------------------------------------
# Top/short K pairs per universe
# -------------------------------------------------
TOP_SHORT_PAIRS_BY_UNIVERSE = {
    "dji": [(5, 0), (5, 5)],
    "nasdaq100": [(5, 0), (5, 5)],
    "sx5e": [(5, 0), (5, 5)],
    "sxxp": [(10, 0), (10, 10)],
    "sp500": [(10, 0), (10, 10)],
}