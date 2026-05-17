import gradio as gr

from Gradio.test_models import (
    get_universe_options
)

def build_train_tab():

    with gr.Tab("Train Model", elem_id="main-tabs"):
        gr.Markdown("## Train Model")
        gr.Markdown(
            "Select the model hyperparameters and choose the training period."
        )

        # ─────────────────────────────────────────────────────────────
        # Main configuration area
        # ─────────────────────────────────────────────────────────────
        with gr.Row(equal_height=False):

            # Left side: controls
            with gr.Column(scale=3, min_width=480):

                with gr.Group():
                    gr.Markdown("### 1. Model configuration")

                    with gr.Row():
                        train_universe = gr.Dropdown(
                            choices=['sp500', 'sxxp', 'sx5e', 'dji', 'nasdaq100'],
                            value=None,
                            label="Universe",
                        )
                        train_task = gr.Dropdown(
                            choices=[],
                            value=None,
                            label="Task Type",
                            interactive=False,
                        )

                    with gr.Row():
                        train_model = gr.Dropdown(
                            choices=[],
                            value=None,
                            label="Model",
                            interactive=False,
                        )
                        train_config = gr.Dropdown(
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
                    gr.Markdown("### 3. Periods")

                    with gr.Row():
                        start_train = gr.Dropdown(
                            choices=[],
                            value=None,
                            label="Start Train Year",
                            interactive=False,
                        )
                        end_train = gr.Dropdown(
                            choices=[],
                            value=None,
                            label="End Train Year",
                            interactive=False,
                        )

                    with gr.Row():
                        start_valid = gr.Dropdown(
                            choices=[],
                            value=None,
                            label="Start Validation Year",
                            interactive=False,
                        )
                        end_valid = gr.Dropdown(
                            choices=[],
                            value=None,
                            label="End Validation Year",
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
