import gradio as gr

from Gradio.test_models import (
    get_universe_options,
    on_universe_change,
    on_task_change,
    on_model_change,
    on_config_change,
    update_end_year_choices,
    on_sl_change,
    on_pl_change,
    run_test_with_logs,
)


def build_tab():
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
                        elem_id="test-result-code",
                    )

        # ── wiring ─────────────────────────────────────────────────────────
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