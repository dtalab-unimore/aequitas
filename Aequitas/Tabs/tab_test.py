import gradio as gr

from Aequitas.test_models import (
    get_universe_options,
    on_universe_change,
    on_task_change,
    on_model_change,
    on_config_change,
    on_sl_change,
    on_pl_change,
    run_test_with_logs,
    TEST_YEARS,
    NO_PARAMS_TEXT,
    get_tickers,
)
from Aequitas.ui import section_header, AUTOSCROLL_LOG_JS, STREAM_EVENT

UI_EVENT = dict(show_progress="hidden", queue=False)


def build_tab():

    def _is_empty(value):
        return value is None or value == ""

    def _update_end_years(start):
        if _is_empty(start):
            return (
                gr.update(choices=[], value=None, interactive=False),
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
        return gr.update(interactive=not _is_empty(start) and not _is_empty(end))

    with gr.Tab("Test Model"):
        universe_options = get_universe_options()

        with gr.Column(elem_classes=["aq-tab"]):

            # ── Configuration cards ──────────────────────────────────────
            with gr.Row(equal_height=True, elem_classes=["aq-row"]):

                with gr.Column(scale=3, elem_classes=["aq-card"]):
                    gr.HTML(section_header("Model configuration"))

                    with gr.Row():
                        test_universe = gr.Dropdown(
                            choices=universe_options,
                            value=None,
                            label="Universe",
                            elem_classes=["dropdown-single-compact"],
                        )
                        test_task = gr.Dropdown(
                            choices=[],
                            value=None,
                            label="Task Type",
                            interactive=False,
                            elem_classes=["dropdown-single-compact"],
                        )

                    with gr.Row():
                        test_model = gr.Dropdown(
                            choices=[],
                            value=None,
                            label="Model",
                            interactive=False,
                            elem_classes=["dropdown-single-compact"],
                        )
                        test_config = gr.Dropdown(
                            choices=[],
                            value=None,
                            label="Configuration",
                            interactive=False,
                            elem_classes=["dropdown-single-compact"],
                        )

                with gr.Column(scale=2, elem_classes=["aq-card"]):
                    gr.HTML(section_header("Sequence settings"))
                    sl = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Sequence length",
                        interactive=False,
                        elem_classes=["dropdown-single-compact"],
                    )
                    pl = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Prediction length",
                        interactive=False,
                        elem_classes=["dropdown-single-compact"],
                    )

                with gr.Column(scale=2, elem_classes=["aq-card"]):
                    gr.HTML(section_header("Test period"))
                    start_test = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Start Year",
                        interactive=False,
                        elem_classes=["dropdown-single-compact"],
                    )
                    end_test = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="End Year",
                        interactive=False,
                        elem_classes=["dropdown-single-compact"],
                    )

                with gr.Column(scale=3, elem_classes=["aq-card"]):
                    gr.HTML(section_header("Train model parameters"))
                    config_info = gr.Textbox(
                        label=None,
                        show_label=False,
                        value=NO_PARAMS_TEXT,
                        lines=8,
                        max_lines=8,
                        interactive=False,
                        buttons=["copy"],
                        elem_classes=["aq-log"],
                    )

            # ── TEST button ──────────────────────────────────────────────
            with gr.Row(elem_classes=["aq-action-row"]):
                test_button = gr.Button(
                    "Test",
                    variant="primary",
                    interactive=False,
                    elem_classes=["aq-primary-btn"],
                )

            # ── Outputs, shown after TEST click ──────────────────────────
            with gr.Column(visible=False) as test_output_section:

                with gr.Row(elem_classes=["aq-row"]):
                    with gr.Column(elem_classes=["tickers-card-wrap"]):
                        tickers_info = gr.HTML(value="", visible=False)

                        tickers_download = gr.DownloadButton(
                            label="Download tickers info ↓",
                            value=None,
                            visible=False,
                            variant="secondary",
                            elem_classes=["aq-chip-btn", "ticker-download-btn"],
                        )

                with gr.Row(equal_height=True, elem_classes=["aq-row"]):
                    with gr.Column(scale=1, elem_classes=["aq-card"]):
                        gr.HTML(section_header("Test progress"))

                        test_logs = gr.Textbox(
                            label=None,
                            value="",
                            lines=10,
                            max_lines=10,
                            interactive=False,
                            autoscroll=True,
                            show_label=False,
                            elem_classes=["aq-log", "aq-log-fixed", "aq-log-autoscroll"],
                        )

                    with gr.Column(scale=1, elem_classes=["aq-card"]):
                        gr.HTML(section_header("Test result"))

                        test_result = gr.Code(
                            label=None,
                            language="json",
                            value=None,
                            interactive=False,
                            show_label=False,
                            elem_id="test-result-code",
                            buttons=["download"],
                        )

        # ── Wiring ───────────────────────────────────────────────────────
        # queue=False on read callbacks prevents lag; .input() avoids
        # re-triggering on programmatic updates.
        test_logs.change(fn=None, js=AUTOSCROLL_LOG_JS, **UI_EVENT)

        test_universe.input(
            fn=on_universe_change,
            inputs=[test_universe],
            outputs=[test_task, test_model, test_config, sl, pl,
                     start_test, end_test, config_info, test_button],
            **UI_EVENT,
        )

        test_task.input(
            fn=on_task_change,
            inputs=[test_universe, test_task],
            outputs=[test_model, test_config, sl, pl,
                     start_test, end_test, config_info, test_button],
            **UI_EVENT,
        )

        test_model.input(
            fn=on_model_change,
            inputs=[test_universe, test_task, test_model],
            outputs=[test_config, sl, pl, start_test, end_test, config_info, test_button],
            **UI_EVENT,
        )

        test_config.input(
            fn=on_config_change,
            inputs=[test_universe, test_task, test_model, test_config],
            outputs=[sl, pl, start_test, end_test, config_info, test_button],
            **UI_EVENT,
        )

        sl.input(
            fn=on_sl_change,
            inputs=[test_universe, test_task, test_model, test_config, sl],
            outputs=[pl, start_test, end_test, config_info, test_button],
            **UI_EVENT,
        )

        pl.input(
            fn=on_pl_change,
            inputs=[test_universe, test_task, test_model, test_config, sl, pl],
            outputs=[config_info, start_test, end_test, test_button],
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

        test_button.click(
            fn=get_tickers,
            inputs=[test_universe, start_test],
            outputs=[tickers_info, tickers_download],
            show_progress="hidden",
        ).then(
            fn=lambda: (gr.update(visible=True), gr.update(value=""), gr.update(value=None)),
            inputs=[],
            outputs=[test_output_section, test_logs, test_result],
            **UI_EVENT,
        ).then(
            fn=run_test_with_logs,
            inputs=[test_universe, test_task, test_model, test_config,
                    sl, pl, start_test, end_test],
            outputs=[test_logs, test_result],
            **STREAM_EVENT,
        )
