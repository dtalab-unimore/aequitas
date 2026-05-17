import gradio as gr

from Gradio.test_models import (
    get_universe_options,
    on_universe_change,
    on_task_change,
    on_model_change,
    on_config_change,
    on_sl_change,
    on_pl_change,
    run_test_with_logs,
    TEST_YEARS, get_tickers,
)

def build_tab():
    UI_EVENT = dict(
        show_progress="hidden",
        queue=False,
    )

    def _is_empty(value):
        return value is None or value == ""

    def _update_end_years(start):
        if _is_empty(start):
            return (
                gr.update(
                    choices=[],
                    value=None,
                    interactive=False,
                ),
                gr.update(interactive=False),
            )

        start = int(start)

        return (
            gr.update(
                choices=[year for year in TEST_YEARS if int(year) >= start],
                value=None,
                interactive=True,
            ),
            gr.update(interactive=False),
        )

    def _update_test_button(start, end):
        return gr.update(
            interactive=not _is_empty(start) and not _is_empty(end)
        )

    with gr.Tab("Test Model", elem_id="main-tabs"):
        gr.Markdown("## Test Model")
        gr.Markdown(
            "Select the desidered model and the test period."
        )

        universe_options = get_universe_options()

        # ─────────────────────────────────────────────────────────────
        # Main configuration area
        # ─────────────────────────────────────────────────────────────
        with gr.Row(equal_height=False):

            # Left side: controls
            with gr.Column(scale=3, min_width=480):

                with gr.Group():
                    gr.Markdown("### 1. Model configuration")

                    with gr.Row():
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

                    with gr.Row():
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

                with gr.Group():
                    gr.Markdown("### 2. Sequence settings")

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

                with gr.Group():
                    gr.Markdown("### 3. Test period")

                    with gr.Row():
                        start_test = gr.Dropdown(
                            choices=[],
                            value=None,
                            label="Start Year",
                            interactive=False,
                        )
                        end_test = gr.Dropdown(
                            choices=[],
                            value=None,
                            label="End Year",
                            interactive=False,
                        )

            # Right side: config preview
            with gr.Column(scale=2, min_width=420):
                config_info = gr.Code(
                    label="Configuration Parameters",
                    language="json",
                    value=None,
                    interactive=False,
                    elem_id="config-code-box",
                )

        # ─────────────────────────────────────────────────────────────
        # TEST button
        # ─────────────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML("")

            with gr.Column(scale=1, min_width=180):
                test_button = gr.Button(
                    "TEST",
                    variant="primary",
                    min_width=120,
                    interactive=False,
                    elem_id="test-button",
                )

            with gr.Column(scale=2):
                gr.HTML("")

        # ─────────────────────────────────────────────────────────────
        # Hidden outputs, shown after TEST click
        # ─────────────────────────────────────────────────────────────
        with gr.Column(visible=False) as test_output_section:
            # Riga tickers + download CSV
            with gr.Row(equal_height=False, elem_classes="tickers-row"):
                with gr.Column(elem_classes="tickers-card-wrap"):
                    tickers_info = gr.HTML(
                        value="",
                        visible=False,
                        elem_id="tickers-info",
                    )

                    tickers_download = gr.DownloadButton(
                        label="Download tickers info ↓",
                        value=None,
                        visible=False,
                        variant="secondary",
                        elem_id="download-tickers-button",
                    )

            # Riga logs + results affiancati
            with gr.Row(equal_height=True, elem_classes="test-results-row"):
                with gr.Column(scale=1, elem_classes="test-output-card"):
                    gr.HTML("""
                    <div class="test-output-header">
                        <span class="test-output-icon">▶</span>
                        <span>Test Progress</span>
                    </div>
                    """)

                    test_logs = gr.Textbox(
                        label=None,
                        value="",
                        lines=10,
                        max_lines=10,
                        interactive=False,
                        autoscroll=True,
                        show_label=False,
                        elem_id="test-logs-box",
                    )

                with gr.Column(scale=1, elem_classes="test-output-card"):
                    gr.HTML("""
                    <div class="test-output-header">
                        <span class="test-output-icon">{ }</span>
                        <span>Test Result</span>
                    </div>
                    """)

                    test_result = gr.Code(
                        label=None,
                        language="json",
                        value=None,
                        interactive=False,
                        show_label=False,
                        elem_id="test-result-code",
                        buttons=['download'],
                    )

        # ─────────────────────────────────────────────────────────────
        # Wiring
        # IMPORTANT:
        # - queue=False sulle callback leggere evita il lag.
        # - input() evita callback scatenate da update programmatici.
        # ─────────────────────────────────────────────────────────────

        test_universe.input(
            fn=on_universe_change,
            inputs=[test_universe],
            outputs=[
                test_task,
                test_model,
                test_config,
                sl,
                pl,
                start_test,
                end_test,
                config_info,
                test_button,
            ],
            **UI_EVENT,
        )

        test_task.input(
            fn=on_task_change,
            inputs=[test_universe, test_task],
            outputs=[
                test_model,
                test_config,
                sl,
                pl,
                start_test,
                end_test,
                config_info,
                test_button,
            ],
            **UI_EVENT,
        )

        test_model.input(
            fn=on_model_change,
            inputs=[test_universe, test_task, test_model],
            outputs=[
                test_config,
                sl,
                pl,
                start_test,
                end_test,
                config_info,
                test_button,
            ],
            **UI_EVENT,
        )

        test_config.input(
            fn=on_config_change,
            inputs=[test_universe, test_task, test_model, test_config],
            outputs=[
                sl,
                pl,
                start_test,
                end_test,
                config_info,
                test_button,
            ],
            **UI_EVENT,
        )

        sl.input(
            fn=on_sl_change,
            inputs=[test_universe, test_task, test_model, test_config, sl],
            outputs=[
                pl,
                start_test,
                end_test,
                config_info,
                test_button,
            ],
            **UI_EVENT,
        )

        pl.input(
            fn=on_pl_change,
            inputs=[
                test_universe,
                test_task,
                test_model,
                test_config,
                sl,
                pl,
            ],
            outputs=[
                config_info,
                start_test,
                end_test,
                test_button,
            ],
            **UI_EVENT,
        )

        start_test.input(
            fn=_update_end_years,
            inputs=[start_test],
            outputs=[end_test, test_button],
            **UI_EVENT,
        )

        end_test.input(
            fn=_update_test_button,
            inputs=[start_test, end_test],
            outputs=[test_button],
            **UI_EVENT,
        )

        show_tickers = test_button.click(
            fn=get_tickers,
            inputs=[test_universe, start_test],
            outputs=[tickers_info, tickers_download],
            show_progress="hidden",
        )

        show_outputs_evt = show_tickers.then(
            fn=lambda: (
                gr.update(visible=True),
                gr.update(value=""),
                gr.update(value=None),
            ),
            inputs=[],
            outputs=[
                test_output_section,
                test_logs,
                test_result,
            ],
            show_progress="hidden",
            queue=False,
        )

        show_outputs_evt.then(
            fn=run_test_with_logs,
            inputs=[
                test_universe,
                test_task,
                test_model,
                test_config,
                sl,
                pl,
                start_test,
                end_test,
            ],
            outputs=[
                test_logs,
                test_result,
            ],
        )