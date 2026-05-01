import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
EVALUATION_DIR = os.path.join(PROJECT_ROOT, "Evaluation")

for p in [PROJECT_ROOT, EVALUATION_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import gradio as gr
import pandas as pd

from data_load import (
    run_data_pipeline,
    use_existing_dataset,
    update_existing_button,
    update_preview,
    refresh_preview_controls
)

from test_models import (
    get_universe_options,
    on_universe_change,
    on_task_change,
    on_model_change,
    on_config_change,
    update_end_year_choices,
    on_sl_change,
    on_pl_change,
    run_test_with_logs
)

# -------------------------------------------------
# TAB 1 / 2 / 3 HELPERS
# -------------------------------------------------
def update_datasets(market):
    if market == "EU":
        datasets = ["sx5e", "sxxp"]
    else:
        datasets = ["sp500", "nasdaq100", "dji"]
    return gr.update(choices=datasets, value=datasets[0] if datasets else None)


def on_dataset_change(dataset_value):
    btn_update = update_existing_button(dataset_value)
    return (
        btn_update,
        "",
        pd.DataFrame(),
    )


def on_market_change(market_value):
    new_dataset_update = update_datasets(market_value)
    return (
        new_dataset_update,
        "",
        pd.DataFrame(),
    )

# -------------------------------------------------
# UI
# -------------------------------------------------
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    radius_size="md",
    font=[gr.themes.GoogleFont("Plus Jakarta Sans")],
)

with gr.Blocks() as demo:
    with gr.Column(elem_classes="main-shell"):
        gr.HTML("""
        <div class="hero">
            <h1>📊 FinBench</h1>
            <p>Benchmarking framework for stock market prediction with reproducible evaluation of financial ML models.</p>
            <p style="margin-top:8px; font-size:0.92rem; opacity:0.88;">CIKM Demo Track</p>
        </div>
        """)

    # ── TAB 1: DOWNLOAD DATA ──────────────────────────────────────────────
    with gr.Tab("Download Data"):
        with gr.Row():
            market  = gr.Dropdown(["EU", "USA"], label="Market", value=None)
            dataset = gr.Dropdown(label="Dataset")
            status  = gr.Textbox(label="Status")

        with gr.Row():
            with gr.Column(scale=2): gr.Markdown("")
            with gr.Column(scale=3):
                btn_download = gr.Button("Fetch Dataset", variant="primary", min_width=140)
            with gr.Column(scale=3):
                btn_existing = gr.Button(
                    "Dataset Not Available", interactive=False, variant="stop", min_width=140
                )
            with gr.Column(scale=2): gr.Markdown("")

        gr.Markdown("---")

        with gr.Row():
            preview_selector = gr.Dropdown(
                label="Preview file", choices=[], value=None, scale=1
            )
            preview = gr.Dataframe(label="Data Preview", scale=4)

        # ── wiring ─────────────────────────────────────────────────────────
        market.change(
            fn=on_market_change,
            inputs=market,
            outputs=[dataset, status, preview]
        )

        dataset.change(
            fn=on_dataset_change,
            inputs=dataset,
            outputs=[btn_existing, status, preview]
        )

        download_evt = btn_download.click(
            fn=run_data_pipeline,
            inputs=[dataset],
            outputs=[status, btn_existing],
        )

        download_evt.then(
            fn=refresh_preview_controls,
            inputs=[dataset],
            outputs=[preview_selector, preview],
        )

        btn_existing.click(fn=use_existing_dataset, inputs=[dataset],
                           outputs=[status, preview_selector, preview, btn_existing])
        preview_selector.change(fn=update_preview, inputs=[dataset, preview_selector], outputs=preview)

    # ── TAB 2: TEST ───────────────────────────────────────────────
    with gr.Tab("Test Model"):
        with gr.Column():
            gr.Markdown("### 1️⃣ Select configuration to test")

            universe_options = get_universe_options()

            with gr.Row():
                with gr.Column():
                    test_universe = gr.Dropdown(
                        choices=universe_options,
                        value=None,
                        label="Universe",
                    )

                    test_task = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Task Type",
                        interactive=False,
                    )

                    test_model = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Model",
                        interactive=False,
                    )

                    test_config = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Configuration",
                        interactive=False,
                    )

                    sl = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Sequence length",
                        interactive=False,
                    )

                    pl = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Prediction length",
                        interactive=False,
                    )
                with gr.Column():
                    config_info = gr.Code(
                        label="Configuration Parameters",
                        language="json",
                        value="",
                        interactive=False,
                    )



            gr.Markdown("---")

            with gr.Column(visible=False, variant="panel") as test_dates_section:
                gr.Markdown("### 2️⃣ Choose dates to test")

                with gr.Row():
                    with gr.Column():
                        start_test = gr.Dropdown(
                            choices=[2020, 2021, 2022, 2023, 2024],
                            value="",
                            label="Start Year",
                        )

                        end_test = gr.Dropdown(
                            choices=[],
                            value="",
                            label="End Year",
                            interactive=False,
                        )

                    with gr.Column():
                        test_logs = gr.Code(
                            label="Test Progress",
                            language="shell",
                            value="",
                            interactive=False,
                        )

                with gr.Row():
                    with gr.Column(scale=5):
                        gr.Markdown("")
                    with gr.Column(scale=1):
                        test_button = gr.Button(
                            "TEST",
                            variant="primary",
                            min_width=50,
                            interactive=False,
                        )
                    with gr.Column(scale=5):
                        gr.Markdown("")

                with gr.Row():
                    test_result = gr.Code(
                        label="Test Result",
                        language="json",
                        value="",
                        interactive=False,
                    )

            test_universe.input(
                fn=on_universe_change,
                inputs=[test_universe],
                outputs=[test_task, test_model, test_config, sl, pl, config_info, test_dates_section]
            )

            test_task.input(
                fn=on_task_change,
                inputs=[test_universe, test_task],
                outputs=[test_model, test_config, sl, pl, config_info, test_dates_section]
            )

            test_model.input(
                fn=on_model_change,
                inputs=[test_universe, test_task, test_model],
                outputs=[test_config, sl, pl, config_info, test_dates_section],
            )

            test_config.input(
                fn=on_config_change,
                inputs=[test_universe, test_task, test_model, test_config],
                outputs=[sl, pl, config_info, test_dates_section],
            )

            start_test.change(
                fn=update_end_year_choices,
                inputs=[start_test],
                outputs=[end_test],
            )

            sl.input(
                fn=on_sl_change,
                inputs=[test_universe, test_task, test_model, test_config, sl],
                outputs=[pl, config_info, test_dates_section],
            )

            pl.input(
                fn=on_pl_change,
                inputs=[test_universe, test_task, test_model, test_config, sl, pl],
                outputs=[config_info, test_dates_section],
            )

            end_test.change(
                fn=lambda start, end: gr.update(interactive=(start is not None and end is not None)),
                inputs=[start_test, end_test],
                outputs=test_button,
            )

            test_button.click(
                fn=run_test_with_logs,
                inputs=[test_universe, test_task, test_model, test_config, sl, pl, start_test, end_test],
                outputs=[test_logs, test_result],
            )


    # ── TAB 3: EVALUATION ───────────────────────────────────────────────────────
    with gr.Tab("Evaluation"):
        gr.Markdown("Evaluation interface coming soon...")

demo.launch(theme=theme, css="style.css")