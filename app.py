import os
import sys
import gradio as gr


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ""))
EVALUATION_DIR = os.path.join(PROJECT_ROOT, "Evaluation")

for p in [PROJECT_ROOT, EVALUATION_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from Gradio.data_load import (
    run_data_pipeline,
    use_existing_dataset,
    refresh_preview_controls,
    reset_preview_page,
    next_preview_page,
    prev_preview_page,
    on_dataset_change,
    on_market_change
)

from Gradio.test_models import (
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

from Gradio.metrics_view import (
    on_metrics_task_change,
    on_metrics_year_range_change,
    on_metrics_model_change,
    on_metrics_config_pl_change,
    update_metrics_plot,
    on_metrics_datasets_change,
)

# -------------------------------------------------
# UI
# -------------------------------------------------
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size="md",
    font=[gr.themes.GoogleFont("Plus Jakarta Sans")],
)

with gr.Blocks() as demo:
    with gr.Column(elem_classes="main-shell"):
        gr.HTML("""
        <div class="hero">
            <h1>FinBench Demo</h1>
        </div>
        """)

    # ── TAB 1: DOWNLOAD DATA ──────────────────────────────────────────────
    with gr.Tab("Download Data"):
        with gr.Row():
            market = gr.Dropdown(["EU", "USA"], label="Market", value=None)
            dataset = gr.Dropdown(label="Dataset")
            status = gr.Textbox(label="Status")

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

        preview_page_state = gr.State(0)

        with gr.Row():
            with gr.Column(scale=1):
                preview_selector = gr.Dropdown(
                    label="Preview file",
                    choices=[],
                    value=None,
                )

            with gr.Column(scale=4):
                preview = gr.Dataframe(
                    label="Data Preview",
                    interactive=False,
                    max_height=420,
                    buttons=["fullscreen", "copy"],
                )

                with gr.Row(elem_classes="preview-pagination-outer"):
                    with gr.Column(scale=1):
                        gr.Markdown("")

                    with gr.Column(scale=0, min_width=200, elem_classes="preview-pagination-box"):
                        with gr.Row(elem_classes="preview-pagination-inner"):
                            btn_prev_preview = gr.Button(
                                "‹",
                                interactive=False,
                                elem_classes="preview-round-btn",
                                scale=0,
                                min_width=36,
                            )

                            preview_page_label = gr.HTML(
                                "",
                                elem_classes="preview-page-label",
                            )

                            btn_next_preview = gr.Button(
                                "›",
                                interactive=False,
                                elem_classes="preview-round-btn",
                                scale=0,
                                min_width=36,
                            )
        # ── wiring ─────────────────────────────────────────────────────────
        market.change(
            fn=on_market_change,
            inputs=market,
            outputs=[
                dataset,
                status,
                preview_selector,
                preview,
                preview_page_label,
                btn_prev_preview,
                btn_next_preview,
                preview_page_state,
            ],
        )

        dataset.change(
            fn=on_dataset_change,
            inputs=dataset,
            outputs=[
                btn_existing,
                status,
                preview_selector,
                preview,
                preview_page_label,
                btn_prev_preview,
                btn_next_preview,
                preview_page_state,
            ],
        )

        download_evt = btn_download.click(
            fn=run_data_pipeline,
            inputs=[dataset],
            outputs=[status, btn_existing],
        )

        download_evt.then(
            fn=refresh_preview_controls,
            inputs=[dataset],
            outputs=[preview_selector],
        ).then(
            fn=reset_preview_page,
            inputs=[dataset, preview_selector],
            outputs=[
                preview,
                preview_page_label,
                btn_prev_preview,
                btn_next_preview,
                preview_page_state,
            ],
        )

        btn_existing.click(
            fn=use_existing_dataset,
            inputs=[dataset],
            outputs=[
                status,
                preview_selector,
                preview,
                preview_page_label,
                btn_prev_preview,
                btn_next_preview,
                preview_page_state,
                btn_existing,
            ],
        )

        preview_selector.change(
            fn=reset_preview_page,
            inputs=[dataset, preview_selector],
            outputs=[
                preview,
                preview_page_label,
                btn_prev_preview,
                btn_next_preview,
                preview_page_state,
            ],
        )

        btn_next_preview.click(
            fn=next_preview_page,
            inputs=[dataset, preview_selector, preview_page_state],
            outputs=[
                preview,
                preview_page_label,
                btn_prev_preview,
                btn_next_preview,
                preview_page_state,
            ],
        )

        btn_prev_preview.click(
            fn=prev_preview_page,
            inputs=[dataset, preview_selector, preview_page_state],
            outputs=[
                preview,
                preview_page_label,
                btn_prev_preview,
                btn_next_preview,
                preview_page_state,
            ],
        )

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
                    with gr.Row():
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
                        lines=31,
                        max_lines=31,
                        value=None,
                        interactive=False,
                    )

            gr.Markdown("---")

            with gr.Column(visible=False, variant="panel") as test_dates_section:
                gr.Markdown("### 2️⃣ Choose dates to test")

                with gr.Row():
                    with gr.Column():
                        start_test = gr.Dropdown(
                            choices=[2020, 2021, 2022, 2023, 2024],
                            value=None,
                            label="Start Year",
                        )

                        end_test = gr.Dropdown(
                            choices=[],
                            value=None,
                            label="End Year",
                            interactive=False,
                        )

                    with gr.Column():
                        test_logs = gr.Textbox(
                            label="Test Progress",
                            value="",
                            lines=6,
                            max_lines=6,
                            interactive=False,
                            autoscroll=True,
                            elem_classes="test-logs-box",
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
                        value=None,
                        interactive=False,
                        elem_id="test-result-code"
                    )

            test_universe.input(
                fn=on_universe_change,
                inputs=[test_universe],
                outputs=[test_task, test_model, test_config, sl, pl, config_info, test_dates_section],
            )

            test_task.input(
                fn=on_task_change,
                inputs=[test_universe, test_task],
                outputs=[test_model, test_config, sl, pl, config_info, test_dates_section],
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

            start_test.change(
                fn=update_end_year_choices,
                inputs=[start_test],
                outputs=[end_test],
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
        gr.Markdown("Evaluation content goes here...")

    # ── TAB 4: VISUALIZATION ───────────────────────────────────────────────────────
    with gr.Tab("Metrics"):
        gr.Markdown("### Model Metrics Visualization")

        with gr.Row():
            with gr.Column(scale=1):
                metrics_task = gr.Dropdown(
                    choices=["Regression", "Classification", "Ranking"],
                    value=None,
                    label="Task Type",
                )
                metrics_year_range = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Year Range",
                    interactive=False,
                )
                metrics_model = gr.Dropdown(
                    choices=[],
                    value=[],
                    label="Model",
                    interactive=False,
                    multiselect=True,
                )
                metrics_config_pl = gr.CheckboxGroup(
                    choices=[],
                    value=[],
                    label="Configuration / Prediction Length",
                    interactive=False,
                )
                metrics_datasets = gr.Dropdown(
                    choices=[],
                    value=[],
                    label="Dataset",
                    multiselect=True,
                    interactive=False,
                )

            with gr.Column(scale=4):
                with gr.Row():
                    metrics_table = gr.HTML(
                        value="",
                        label="Metrics Table",
                        elem_classes="metrics-table-container",
                    )

                with gr.Accordion("Plot", open=False, elem_classes="metrics-accordion", visible=False) as metrics_plot_accordion:
                    with gr.Row(elem_classes="metrics-plot-panel"):
                        with gr.Column(scale=1, min_width=210, elem_classes="metrics-plot-controls"):
                            metrics_metric_selector = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="Metric",
                                interactive=False,
                                visible=True,
                            )

                            metrics_dataset_radio = gr.Radio(
                                choices=[],
                                value=None,
                                label="Dataset",
                                interactive=False,
                                visible=True,
                            )

                        with gr.Column(scale=5, elem_classes="metrics-plot-card"):
                            metrics_plot = gr.Plot(
                                value=None,
                                label="",
                                visible=True,
                                elem_classes="metrics-plot-box",
                            )

        # ── wiring ──────────────────────────────────────────────────────────
        metrics_task.change(
            fn=on_metrics_task_change,
            inputs=[metrics_task],
            outputs=[
                metrics_year_range, metrics_model, metrics_config_pl,
                metrics_datasets, metrics_table,
                metrics_metric_selector, metrics_plot, metrics_dataset_radio,
            ],
        )

        metrics_year_range.change(
            fn=on_metrics_year_range_change,
            inputs=[metrics_task, metrics_year_range],
            outputs=[
                metrics_model, metrics_config_pl, metrics_datasets,
                metrics_table,
                metrics_metric_selector, metrics_plot, metrics_dataset_radio,
            ],
        )

        metrics_model.change(
            fn=on_metrics_model_change,
            inputs=[metrics_task, metrics_year_range, metrics_model],
            outputs=[
                metrics_config_pl, metrics_datasets, metrics_table,
                metrics_metric_selector, metrics_plot, metrics_dataset_radio,
            ],
        )

        metrics_config_pl.change(
            fn=on_metrics_config_pl_change,
            inputs=[metrics_task, metrics_year_range, metrics_model, metrics_config_pl],
            outputs=[
                metrics_datasets, metrics_table,
                metrics_metric_selector, metrics_plot, metrics_dataset_radio,
            ],
        )

        metrics_datasets.change(
            fn=on_metrics_datasets_change,
            inputs=[
                metrics_task,
                metrics_year_range,
                metrics_model,
                metrics_config_pl,
                metrics_datasets,
                metrics_metric_selector,
                metrics_dataset_radio,
            ],
            outputs=[
                metrics_table,
                metrics_metric_selector,
                metrics_plot,
                metrics_dataset_radio,
                metrics_plot_accordion,
            ],
        )

        metrics_metric_selector.change(
            fn=update_metrics_plot,
            inputs=[
                metrics_task, metrics_year_range, metrics_model,
                metrics_config_pl, metrics_datasets,
                metrics_metric_selector, metrics_dataset_radio,
            ],
            outputs=[metrics_plot],
        )

        metrics_dataset_radio.change(
            fn=update_metrics_plot,
            inputs=[
                metrics_task, metrics_year_range, metrics_model,
                metrics_config_pl, metrics_datasets,
                metrics_metric_selector, metrics_dataset_radio,
            ],
            outputs=[metrics_plot],
        )

CSS_PATH = os.path.join(PROJECT_ROOT, "Gradio", "style.css")

with open(CSS_PATH, "r", encoding="utf-8") as f:
    custom_css = f.read()

demo.queue().launch(
    server_port=7860,
    theme=theme,
    css=custom_css,
)
