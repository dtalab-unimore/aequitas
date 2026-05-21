import gradio as gr
from Gradio.test_models import get_tickers
from Gradio.train_models import toggle_train_button, validate_train_inputs_soft, \
    train_model_with_logs_if_valid, get_models_train,on_train_model_change


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

    with gr.Tab("Train Model", elem_id="main-tabs"):
        train_validation_state = gr.State(False)

        gr.Markdown("## Train Model")
        gr.Markdown(
            "Select the model hyperparameters and choose the training period."
        )

        # ─────────────────────────────────────────────────────────────
        # Main configuration area
        # ─────────────────────────────────────────────────────────────
        with gr.Row(equal_height=False):

            # Left side: controls
            with gr.Column(scale=2):

                with gr.Group():
                    gr.Markdown("### 1. Model configuration")

                    with gr.Row():
                        train_universe = gr.Dropdown(
                            choices=['sp500', 'sxxp', 'sx5e', 'dji', 'nasdaq100'],
                            value=None,
                            label="Universe",
                            elem_classes=['dropdown-single-compact']
                        )
                        train_task = gr.Dropdown(
                            choices=['Classification', 'Regression', 'Ranking'],
                            value=None,
                            label="Task Type",
                            interactive=True,
                            elem_classes=['dropdown-single-compact']
                        )

                    with gr.Row():
                        train_model = gr.Dropdown(
                            choices=[],
                            value=None,
                            label="Model",
                            interactive=False,
                            elem_classes=['dropdown-single-compact']
                        )
                        train_config = gr.Textbox(
                            value=None,
                            label="Configuration",
                            interactive=True,
                            placeholder="Insert the configuration name",
                        )

            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 2. Sequence settings")

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

            with gr.Column(scale=2):
                with gr.Group():
                    gr.Markdown("### 3. Periods")

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

        with gr.Row():
            train_validation_msg = gr.HTML(
                value="",
                visible=False,
                elem_id="train-validation-msg",
            )

        # ─────────────────────────────────────────────────────────────
        # TRAIN button
        # ─────────────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML("")

            with gr.Column(scale=1, min_width=180):
                train_button = gr.Button(
                    "TRAIN",
                    variant="primary",
                    min_width=120,
                    interactive=False,
                    elem_id="test-button",
                )

            with gr.Column(scale=2):
                gr.HTML("")

        # ─────────────────────────────────────────────────────────────
        # Hidden outputs, shown after TRAIN click
        # ─────────────────────────────────────────────────────────────
        with gr.Column(visible=False) as train_output_section:

            with gr.Row(equal_height=True, elem_classes="test-results-row train-results-row"):

                with gr.Column(scale=1, elem_classes="test-output-card tickers-card-wrap"):
                    gr.HTML("""
                    <div class="test-output-header">
                        <span class="test-output-icon">▦</span>
                        <span>Tickers Used</span>
                    </div>
                    """)

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
                        elem_id="download-train-tickers-button",
                    )

                # Colonna destra: progress
                with gr.Column(
                        scale=1,
                        elem_classes="test-output-card",
                        elem_id="train-progress-card",
                ):
                    gr.HTML("""
                    <div class="test-output-header">
                        <span class="test-output-icon">▶</span>
                        <span>Train Progress</span>
                    </div>
                    """)

                    train_loading = gr.HTML(
                        value="""
                        <div class="train-header-loading">
                            <div class="train-header-spinner"></div>
                            <span>Training...</span>
                        </div>
                        """,
                        visible=False,
                        elem_id="train-header-loading-box",
                    )

                    train_logs = gr.Textbox(
                        label=None,
                        value="",
                        lines=10,
                        max_lines=10,
                        interactive=False,
                        autoscroll=True,
                        show_label=False,
                        elem_id="test-logs-box",
                    )

            train_result = gr.Code(
                label=None,
                language="json",
                value=None,
                interactive=False,
                visible=False,
                show_label=False,
                elem_id="test-result-code-hidden",
            )

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

        train_task.change(
            fn=get_models_train,
            inputs=[train_task],
            outputs=[train_model, sl, train_button],
            show_progress="hidden",
            queue=False,
        )

        train_model.change(
            fn=on_train_model_change,
            inputs=required_train_inputs,
            outputs=[sl, train_button],
            show_progress="hidden",
            queue=False,
        )

        dropdown_inputs = [
            train_universe,

        ]

        textbox_inputs = [
            train_config,
            sl,
            pl,
            start_train,
            end_train,
            start_valid,
            end_valid,
        ]

        for component in dropdown_inputs:
            component.change(
                fn=toggle_train_button,
                inputs=required_train_inputs,
                outputs=train_button,
                show_progress="hidden",
                queue=False,
            )

        for component in textbox_inputs:
            component.input(
                fn=toggle_train_button,
                inputs=required_train_inputs,
                outputs=train_button,
                show_progress="hidden",
                queue=False,
            )

            component.change(
                fn=toggle_train_button,
                inputs=required_train_inputs,
                outputs=train_button,
                show_progress="hidden",
                queue=False,
            )

        validate_train_evt = train_button.click(
            fn=validate_train_inputs_soft,
            inputs=[
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
            ],
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
            show_progress="hidden",
            queue=False,
        )

        show_train_tickers_evt = validate_train_evt.then(
            fn=get_train_tickers_if_valid,
            inputs=[
                train_validation_state,
                train_universe,
                end_valid,
            ],
            outputs=[
                tickers_info,
                tickers_download,
            ],
            show_progress="hidden",
            queue=False,
        )

        # show_train_tickers_evt.then(
        #     fn=train_model_with_logs_if_valid,
        #     inputs=[
        #         train_validation_state,
        #         train_universe,
        #         train_task,
        #         train_model,
        #         train_config,
        #         sl,
        #         pl,
        #         start_train,
        #         end_train,
        #         start_valid,
        #         end_valid,
        #     ],
        #     outputs=[
        #         train_logs,
        #         train_result,
        #     ],
        #     show_progress="hidden",
        # )
        show_train_loading_evt = show_train_tickers_evt.then(
            fn=show_train_loading_if_valid,
            inputs=[train_validation_state],
            outputs=[train_loading],
            show_progress="hidden",
            queue=False,
        )

        show_train_loading_evt.then(
            fn=train_model_with_logs_if_valid,
            inputs=[
                train_validation_state,
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
            ],
            outputs=[
                train_logs,
                train_result,
                train_loading,
            ],
            show_progress="hidden",
        )