import gradio as gr

from Gradio.data_load import get_models
from Gradio.train_models import toggle_train_button, train_model_with_logs, validate_train_inputs_soft, \
    train_model_with_logs_if_valid


def build_train_tab():

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
                        )
                        train_task = gr.Dropdown(
                            choices=['Classification', 'Regression', 'Ranking'],
                            value=None,
                            label="Task Type",
                            interactive=True,
                        )

                    with gr.Row():
                        train_model = gr.Dropdown(
                            choices=get_models(),
                            value=None,
                            label="Model",
                            interactive=True,
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
        # Hidden outputs, shown after TEST click
        # ─────────────────────────────────────────────────────────────
        with gr.Column(visible=False) as train_output_section:
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

                with gr.Column(scale=1, elem_classes="test-output-card"):
                    gr.HTML("""
                    <div class="test-output-header">
                        <span class="test-output-icon">{ }</span>
                        <span>Test Result</span>
                    </div>
                    """)

                    train_result = gr.Code(
                        label=None,
                        language="json",
                        value=None,
                        interactive=False,
                        show_label=False,
                        elem_id="test-result-code",
                        buttons=['download'],
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

        dropdown_inputs = [
            train_universe,
            train_task,
            train_model,
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
            component.change(
                fn=toggle_train_button,
                inputs=required_train_inputs,
                outputs=train_button,
                show_progress="hidden",
                queue=False,
            )

        train_button.click(
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
        ).then(
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
            ],
            show_progress="hidden",
        )
