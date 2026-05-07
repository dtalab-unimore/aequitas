import os
import sys
import gradio as gr
from Gradio.Tabs.tab_download import build_tab as build_download_tab
from Gradio.Tabs.tab_test import build_tab as build_test_tab
from Gradio.Tabs.tab_evaluation import build_tab as build_evaluation_tab
from Gradio.Tabs.tab_metrics import build_tab as build_metrics_tab

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ""))
EVALUATION_DIR = os.path.join(PROJECT_ROOT, "Evaluation")

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
with gr.Blocks() as demo:
    with gr.Column(elem_classes="main-shell"):
        gr.HTML("""
        <div class="hero">
            <h1>FinBench Demo</h1>
        </div>
        """)

    build_download_tab()
    build_test_tab()
    build_evaluation_tab()
    build_metrics_tab()

# -------------------------------------------------
# Launch
# -------------------------------------------------
CSS_PATH = os.path.join(PROJECT_ROOT, "Gradio", "style.css")

with open(CSS_PATH, "r", encoding="utf-8") as f:
    custom_css = f.read()

demo.queue().launch(
    server_port=7860,
    theme=theme,
    css=custom_css,
)
