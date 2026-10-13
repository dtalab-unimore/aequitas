import os
import sys
from pathlib import Path

GRADIO_TMP_DIR = Path("./.gradio_tmp").resolve()
GRADIO_TMP_DIR.mkdir(parents=True, exist_ok=True)

# Must be set before importing gradio
os.environ["GRADIO_TEMP_DIR"] = str(GRADIO_TMP_DIR)

import gradio as gr
from Aequitas.Tabs.tab_download import build_tab as build_download_tab
from Aequitas.Tabs.tab_train import build_train_tab
from Aequitas.Tabs.tab_test import build_tab as build_test_tab
from Aequitas.Tabs.tab_evaluation import build_tab as build_evaluation_tab
from Aequitas.Tabs.tab_metrics import build_tab as build_metrics_tab

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EVALUATION_DIR = os.path.join(PROJECT_ROOT, "Evaluation")
CSS_PATH = os.path.join(PROJECT_ROOT, "Aequitas", "style.css")

os.environ["FINBENCH_ROOT"] = PROJECT_ROOT

for p in [PROJECT_ROOT, EVALUATION_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)


# -------------------------------------------------
# Theme
# -------------------------------------------------
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size="md",
    font=[gr.themes.GoogleFont("Plus Jakarta Sans")],
)

# -------------------------------------------------
# Layout
# -------------------------------------------------
with gr.Blocks(title="Aequitas") as demo:
    gr.HTML('<div class="hero"><h1>Aequitas</h1></div>')

    build_download_tab()
    build_train_tab()
    build_test_tab()
    build_evaluation_tab()
    build_metrics_tab()

# -------------------------------------------------
# Launch
# -------------------------------------------------
with open(CSS_PATH, "r", encoding="utf-8") as f:
    custom_css = f.read()

demo.queue().launch(
    server_port=int(os.getenv("INTERFACE_PORT")),
    theme=theme,
    css=custom_css,
)
