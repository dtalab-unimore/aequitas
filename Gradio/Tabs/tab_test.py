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
    with gr.Tab("Test Model"):
        gr.Markdown("## Test Model")
        gr.Markdown(
            "Select the model configuration, choose the test period, and run the evaluation."
        )

        universe_options = get_universe_options()

        # ─────────────────────────────────────────────────────────────
        # Configuration area: 3 columns
        # ─────────────────────────────────────────────────────────────
        with gr.Row(equal_height=True):

            # Column 1
            with gr.Column(scale=1, min_width=280):
                test_universe = gr.Dropdown(
                    choices=universe_options,
                    value=None,
                    label="Universe",
                )

                test_model = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Model",
                    interactive=False,
                )

                sl = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Sequence length",
                    interactive=False,
                )

                start_test = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Start Year",
                    interactive=False,
                )

            # Column 2
            with gr.Column(scale=1, min_width=280):
                test_task = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Task Type",
                    interactive=False,
                )

                test_config = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Configuration",
                    interactive=False,
                )

                pl = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Prediction length",
                    interactive=False,
                )

                end_test = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="End Year",
                    interactive=False,
                )

            # Column 3
            with gr.Column(scale=2, min_width=420):
                config_info = gr.Code(
                    label="Configuration Parameters",
                    language="json",
                    value=None,
                    interactive=False,
                    elem_id="config-code-box",
                )

        # ─────────────────────────────────────────────────────────────
        # Centered TEST button
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
            tickers_info = gr.HTML(
                value="",
                visible=False,
                elem_id="tickers-info",
            )

            test_logs = gr.Textbox(
                label="Test Progress",
                value="",
                lines=7,
                max_lines=7,
                interactive=False,
                autoscroll=True,
                elem_classes="test-logs-box",
            )

            test_result = gr.Code(
                label="Test Result",
                language="json",
                value=None,
                interactive=False,
                elem_id="test-result-code",
            )

        # ── wiring ─────────────────────────────────────────────────────

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
        )

        start_test.change(
            fn=lambda start: gr.update(
                choices=[] if start in [None, ""] else [
                    year for year in TEST_YEARS if year >= int(start)
                ],
                value=None,
                interactive=start not in [None, ""],
            ),
            inputs=[start_test],
            outputs=[end_test],
            show_progress="hidden",
            queue=False,
        )

        end_test.change(
            fn=lambda start, end: gr.update(
                interactive=(
                    start not in [None, ""]
                    and end not in [None, ""]
                )
            ),
            inputs=[start_test, end_test],
            outputs=[test_button],
            show_progress="hidden",
            queue=False,
        )

        show_tickers = test_button.click(
            fn=get_tickers,
            inputs=[test_universe, start_test],
            outputs=[tickers_info],
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