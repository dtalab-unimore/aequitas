import gradio as gr

from Gradio.data_load import (
    run_data_pipeline,
    use_existing_dataset,
    refresh_preview_controls,
    reset_preview_page,
    next_preview_page,
    prev_preview_page,
    on_dataset_change,
    on_market_change, get_models, get_files,
)


def build_tab():
    with gr.Tab("Download Data", elem_id="main-tabs"):
        with gr.Row():
            market = gr.Dropdown(["EU", "USA"], label="Market", value=None)
            dataset = gr.Dropdown(label="Dataset")
            status = gr.Textbox(label="Status")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("")
            with gr.Column(scale=3):
                btn_download = gr.Button("Fetch Dataset", variant="primary", min_width=140)
            with gr.Column(scale=3):
                btn_existing = gr.Button(
                    "Dataset Not Available", interactive=False, variant="stop", min_width=140
                )
            with gr.Column(scale=2):
                gr.Markdown("")

        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                universes = gr.Dropdown(
                    choices=['sp500', "dji", "nasdaq100", "sxxp", "sx5e"],
                    label="Select the dataset",
                    value=None
                )

                models = gr.Dropdown(
                    choices=get_models(),
                    label="Available Models",
                    value=None
                )

                btn_check_files = gr.Button(
                    "Check Files",
                    variant="primary",
                    min_width=140
                )

            with gr.Column():
                files = gr.Textbox(
                    label="Data Description",
                    interactive=False,
                    lines=10,
                    max_lines=10,
                    elem_id="files",
                )

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
                dataset, status, preview_selector, preview,
                preview_page_label, btn_prev_preview, btn_next_preview,
                preview_page_state,
            ],
        )

        dataset.change(
            fn=on_dataset_change,
            inputs=dataset,
            outputs=[
                btn_existing, status, preview_selector, preview,
                preview_page_label, btn_prev_preview, btn_next_preview,
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
                preview, preview_page_label, btn_prev_preview,
                btn_next_preview, preview_page_state,
            ],
        )

        btn_existing.click(
            fn=use_existing_dataset,
            inputs=[dataset],
            outputs=[
                status, preview_selector, preview, preview_page_label,
                btn_prev_preview, btn_next_preview, preview_page_state,
                btn_existing,
            ],
        )

        preview_selector.change(
            fn=reset_preview_page,
            inputs=[dataset, preview_selector],
            outputs=[
                preview, preview_page_label, btn_prev_preview,
                btn_next_preview, preview_page_state,
            ],
        )

        btn_next_preview.click(
            fn=next_preview_page,
            inputs=[dataset, preview_selector, preview_page_state],
            outputs=[
                preview, preview_page_label, btn_prev_preview,
                btn_next_preview, preview_page_state,
            ],
        )

        btn_prev_preview.click(
            fn=prev_preview_page,
            inputs=[dataset, preview_selector, preview_page_state],
            outputs=[
                preview, preview_page_label, btn_prev_preview,
                btn_next_preview, preview_page_state,
            ],
        )

        btn_check_files.click(
            fn=get_files,
            inputs=[universes, models],
            outputs=[files]
        )