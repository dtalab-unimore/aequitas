import gradio as gr

from Aequitas.test_models import get_tickers
from Aequitas.train_models import (
    toggle_train_button,
    validate_train_inputs_soft,
    train_model_with_logs_if_valid,
    get_models_train,
    on_train_model_change,
    restore_training_view,
    resume_training,
)
from Aequitas.ui import section_header, loading_badge, AUTOSCROLL_LOG_JS

UI_EVENT = dict(show_progress="hidden", queue=False)


def build_train_tab():

    def get_train_tickers_if_valid(is_valid, universe, end_valid):
        if not is_valid:
            return (
                gr.update(value="", visible=False),
                gr.update(value=None, visible=False),
            )

        return get_tickers(universe, end_valid)

    def show_train_loading_if_valid(is_valid):
        return gr.update(visible=bool(is_valid))

    # Job id of this browser's latest run, kept in localStorage across page
    # reloads. Declared outside the tab: Gradio mounts a tab's components only
    # when the tab is opened, and BrowserState reads its value on mount.
    train_job_state = gr.BrowserState(None, storage_key="aequitas_train_job")

    with gr.Tab("Train Model") as train_tab:
        train_validation_state = gr.State(False)
        resume_token = gr.State(None)
        # True once this page has shown a run (restored or started here).
        run_shown_state = gr.State(False)

        with gr.Column(elem_classes=["aq-tab"]):

            # ── Configuration cards ──────────────────────────────────────
            with gr.Row(equal_height=True, elem_classes=["aq-row"]):

                with gr.Column(scale=2, elem_classes=["aq-card"]):
                    gr.HTML(section_header("Model configuration"))

                    with gr.Row():
                        train_universe = gr.Dropdown(
                            choices=["sp500", "sxxp", "sx5e", "dji", "nasdaq100"],
                            value=None,
                            label="Universe",
                            elem_classes=["dropdown-single-compact"],
                        )
                        train_task = gr.Dropdown(
                            choices=["Classification", "Regression", "Ranking"],
                            value=None,
                            label="Task Type",
                            interactive=True,
                            elem_classes=["dropdown-single-compact"],
                        )

                    with gr.Row():
                        train_model = gr.Dropdown(
                            choices=[],
                            value=None,
                            label="Model",
                            interactive=False,
                            elem_classes=["dropdown-single-compact"],
                        )
                        train_config = gr.Textbox(
                            value=None,
                            label="Configuration",
                            interactive=True,
                            placeholder="Insert the configuration name",
                        )

                with gr.Column(scale=1, elem_classes=["aq-card"]):
                    gr.HTML(section_header("Sequence settings"))

                    sl = gr.Textbox(
                        value=None,
                        label="Sequence length",
                        interactive=True,
                        placeholder="Insert the sequence length",
                    )
                    pl = gr.Textbox(
                        value=None,
                        label="Prediction length",
                        interactive=True,
                        placeholder="Insert the prediction length",
                    )

                with gr.Column(scale=2, elem_classes=["aq-card"]):
                    gr.HTML(section_header("Training period"))

                    with gr.Row():
                        start_train = gr.Textbox(
                            value=None,
                            label="Start Train Year",
                            interactive=True,
                            placeholder="Insert the start train year",
                        )
                        end_train = gr.Textbox(
                            value=None,
                            label="End Train Year",
                            interactive=True,
                            placeholder="Insert the end train year",
                        )

                    with gr.Row():
                        start_valid = gr.Textbox(
                            value=None,
                            label="Start Validation Year",
                            interactive=True,
                            placeholder="Insert the start validation year",
                        )
                        end_valid = gr.Textbox(
                            value=None,
                            label="End Validation Year",
                            interactive=True,
                            placeholder="Insert the end validation year",
                        )

            train_validation_msg = gr.HTML(value="", visible=False)

            # ── TRAIN button ─────────────────────────────────────────────
            with gr.Row(elem_classes=["aq-action-row"]):
                train_button = gr.Button(
                    "Train",
                    variant="primary",
                    interactive=False,
                    elem_classes=["aq-primary-btn"],
                )

            # ── Outputs, shown after TRAIN click ─────────────────────────
            with gr.Column(visible=False) as train_output_section:

                with gr.Row(equal_height=True, elem_classes=["aq-row"]):

                    with gr.Column(scale=1, elem_classes=["aq-card", "tickers-card-wrap"]):
                        gr.HTML(section_header("Tickers used"))

                        tickers_info = gr.HTML(value="", visible=False)

                        tickers_download = gr.DownloadButton(
                            label="Download tickers info ↓",
                            value=None,
                            visible=False,
                            variant="secondary",
                            elem_classes=["aq-chip-btn", "ticker-download-btn"],
                        )

                    with gr.Column(scale=1, elem_classes=["aq-card", "aq-badge-host"]):
                        gr.HTML(section_header("Train progress"))

                        train_loading = gr.HTML(
                            value=loading_badge("Training..."),
                            visible=False,
                            elem_classes=["aq-badge-slot"],
                        )

                        train_logs = gr.Textbox(
                            label=None,
                            value="",
                            lines=10,
                            max_lines=10,
                            interactive=False,
                            autoscroll=True,
                            show_label=False,
                            elem_classes=["aq-log", "aq-log-fixed", "aq-log-autoscroll"],
                        )

                train_result = gr.Code(
                    label=None,
                    language="json",
                    value=None,
                    interactive=False,
                    visible=False,
                    show_label=False,
                )

        # ── Wiring ───────────────────────────────────────────────────────
        required_train_inputs = [
            train_universe,
            train_task,
            train_model,
            train_config,
            sl,
            pl,
            start_train,
            end_train,
            start_valid,
            end_valid,
        ]

        train_logs.change(fn=None, js=AUTOSCROLL_LOG_JS, **UI_EVENT)

        # .input (user edits only): restoring a run after a reload sets these
        # values programmatically and must not trigger the reset cascade.
        train_task.input(
            fn=get_models_train,
            inputs=[train_task],
            outputs=[train_model, sl, train_button],
            **UI_EVENT,
        )

        train_model.input(
            fn=on_train_model_change,
            inputs=required_train_inputs,
            outputs=[sl, train_button],
            **UI_EVENT,
        )

        train_universe.change(
            fn=toggle_train_button,
            inputs=required_train_inputs,
            outputs=train_button,
            **UI_EVENT,
        )

        for component in [train_config, sl, pl, start_train, end_train, start_valid, end_valid]:
            for event in (component.input, component.change):
                event(
                    fn=toggle_train_button,
                    inputs=required_train_inputs,
                    outputs=train_button,
                    **UI_EVENT,
                )

        train_button.click(fn=lambda: True, outputs=[run_shown_state], **UI_EVENT)

        train_button.click(
            fn=validate_train_inputs_soft,
            inputs=required_train_inputs,
            outputs=[
                train_validation_state,
                train_output_section,
                train_logs,
                train_result,
                sl,
                pl,
                start_train,
                end_train,
                start_valid,
                end_valid,
                train_validation_msg,
                train_button,
            ],
            **UI_EVENT,
        ).then(
            fn=get_train_tickers_if_valid,
            inputs=[train_validation_state, train_universe, end_valid],
            outputs=[tickers_info, tickers_download],
            **UI_EVENT,
        ).then(
            fn=show_train_loading_if_valid,
            inputs=[train_validation_state],
            outputs=[train_loading],
            **UI_EVENT,
        ).then(
            fn=train_model_with_logs_if_valid,
            inputs=[train_validation_state, *required_train_inputs],
            outputs=[train_logs, train_result, train_loading, train_job_state, train_button],
            show_progress="hidden",
        )

        # After a page reload, opening the tab shows the run this browser
        # started (if any). This runs on tab select rather than page load:
        # Gradio 6 mounts a tab only when it is opened, and updating an
        # unmounted tab on load can leave the app-wide loading overlay stuck.
        # The log stream itself runs off resume_token's change event.
        train_tab.select(
            fn=restore_training_view,
            inputs=[train_job_state, run_shown_state],
            outputs=[*required_train_inputs, train_output_section, tickers_info, tickers_download,
                     resume_token, run_shown_state],
            show_progress="hidden",
        )

        resume_token.change(
            fn=resume_training,
            inputs=[train_job_state],
            outputs=[train_logs, train_result, train_loading, train_job_state, train_button],
            show_progress="hidden",
        )
