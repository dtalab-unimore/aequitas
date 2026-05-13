import gradio as gr

from Gradio.data_load import (
    get_models,
    inspect_required_files,
    download_missing_files,
    refresh_preview_controls,
    reset_preview_page,
    next_preview_page,
    prev_preview_page,
)


DATASETS = ["sp500", "dji", "nasdaq100", "sxxp", "sx5e"]


def build_tab():
    with gr.Tab("Download Data", elem_id="main-tabs"):
        with gr.Column(elem_classes=["download-shell"]):

            # ── Top row: controls + required files ────────────────────────
            with gr.Row(elem_classes=["download-top-row"]):

                # LEFT – selezione + azione
                with gr.Column(scale=1, min_width=260, elem_classes=["download-card", "download-controls-card"]):
                    gr.Markdown(
                        """<div class="evaluation-output-header">
                            <span>Configuration</span>
                        </div>"""
                    )

                    dataset = gr.Dropdown(
                        choices=DATASETS,
                        label="Dataset",
                        value=None,
                    )
                    models = gr.Dropdown(
                        choices=get_models(),
                        label="Model",
                        value=None,
                    )
                    status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        elem_id="download-status-box",
                    )

                    # Stati interni
                    universe_targets_state = gr.State([])
                    market_targets_state = gr.State([])

                    btn_download_missing = gr.Button(
                        "Download Missing Files",
                        variant="primary",
                        interactive=False,
                        elem_id="download-missing-btn",
                    )

                # RIGHT – required files con spinner
                with gr.Column(
                    scale=2,
                    elem_classes=["download-card", "download-files-card", "evaluation-loading-host"],
                ):
                    gr.Markdown(
                        """<div class="evaluation-output-header">
                            <span>Required files</span>
                        </div>"""
                    )

                    files_loading = gr.HTML(
                        """
                        <div class="evaluation-loading-overlay">
                            <div class="evaluation-loading-box">
                                <div class="evaluation-loading-spinner"></div>
                                <div class="evaluation-loading-text">Downloading files...</div>
                            </div>
                        </div>
                        """,
                        visible=False,
                        elem_id="files-loading-overlay",
                    )

                    files = gr.Textbox(
                        label=None,
                        interactive=False,
                        lines=11,
                        max_lines=14,
                        show_label=False,
                        elem_id="files",
                    )

            # ── Bottom row: preview selector + dataframe ───────────────────
            with gr.Row(elem_classes=["download-top-row", "download-preview-row"]):

                # LEFT – selettore file
                with gr.Column(scale=1, min_width=200, elem_classes=["download-card", "download-selector-card"]):
                    gr.Markdown(
                        """<div class="evaluation-output-header">
                            <span>Preview file</span>
                        </div>"""
                    )
                    preview_selector = gr.Dropdown(
                        label=None,
                        choices=[],
                        value=None,
                        show_label=False,
                        elem_id="preview-selector",
                    )

                # RIGHT – dataframe + paginazione + spinner
                with gr.Column(
                    scale=5,
                    elem_classes=["download-card", "download-preview-card", "evaluation-loading-host"],
                ):
                    gr.Markdown(
                        """<div class="evaluation-output-header">
                            <span>Data preview</span>
                        </div>"""
                    )

                    preview_page_state = gr.State(0)

                    preview_loading = gr.HTML(
                        """
                        <div class="evaluation-loading-overlay">
                            <div class="evaluation-loading-box">
                                <div class="evaluation-loading-spinner"></div>
                                <div class="evaluation-loading-text">Loading preview...</div>
                            </div>
                        </div>
                        """,
                        visible=False,
                        elem_id="preview-loading-overlay",
                    )

                    preview = gr.Dataframe(
                        label=None,
                        interactive=False,
                        max_height=360,
                        show_label=False,
                        elem_id="preview-table",
                    )

                    with gr.Row(elem_classes=["preview-pagination-outer"]):
                        with gr.Column(scale=1):
                            gr.HTML("")
                        with gr.Column(
                            scale=0,
                            min_width=200,
                            elem_classes=["preview-pagination-box"],
                        ):
                            with gr.Row(elem_classes=["preview-pagination-inner"]):
                                btn_prev_preview = gr.Button(
                                    "‹",
                                    interactive=False,
                                    elem_classes=["preview-round-btn"],
                                    scale=0,
                                    min_width=36,
                                )
                                preview_page_label = gr.HTML(
                                    "",
                                    elem_classes=["preview-page-label"],
                                )
                                btn_next_preview = gr.Button(
                                    "›",
                                    interactive=False,
                                    elem_classes=["preview-round-btn"],
                                    scale=0,
                                    min_width=36,
                                )
                        with gr.Column(scale=1):
                            gr.HTML("")

        # ── Wiring ────────────────────────────────────────────────────────

        dataset.change(
            fn=inspect_required_files,
            inputs=[dataset, models],
            outputs=[
                status,
                files,
                btn_download_missing,
                universe_targets_state,
                market_targets_state,
            ],
            show_progress="hidden",
            queue=False,
        ).then(
            fn=refresh_preview_controls,
            inputs=[dataset, universe_targets_state, market_targets_state],
            outputs=[preview_selector],
            show_progress="hidden",
            queue=False,
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
            show_progress="hidden",
            queue=False,
        )

        models.change(
            fn=inspect_required_files,
            inputs=[dataset, models],
            outputs=[
                status,
                files,
                btn_download_missing,
                universe_targets_state,
                market_targets_state,
            ],
            show_progress="hidden",
            queue=False,
        ).then(
            fn=refresh_preview_controls,
            inputs=[dataset, universe_targets_state, market_targets_state],
            outputs=[preview_selector],
            show_progress="hidden",
            queue=False,
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
            show_progress="hidden",
            queue=False,
        )

        download_evt = btn_download_missing.click(
            fn=lambda: gr.update(visible=True),
            outputs=[files_loading],
            show_progress="hidden",
            queue=False,
        ).then(
            fn=download_missing_files,
            inputs=[
                dataset,
                models,
                universe_targets_state,
                market_targets_state,
            ],
            outputs=[
                status,
                files,
                btn_download_missing,
                universe_targets_state,
                market_targets_state,
            ],
            show_progress="full",
            queue=True,
        ).then(
            fn=lambda: gr.update(visible=False),
            outputs=[files_loading],
            show_progress="hidden",
            queue=False,
        )

        download_evt.then(
            fn=refresh_preview_controls,
            inputs=[dataset,  universe_targets_state, market_targets_state],
            outputs=[preview_selector],
            show_progress="hidden",
            queue=False,
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
            show_progress="hidden",
            queue=False,
        )

        preview_selector.change(
            fn=lambda: gr.update(visible=True),
            outputs=[preview_loading],
            show_progress="hidden",
            queue=False,
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
            show_progress="hidden",
            queue=False,
        ).then(
            fn=lambda: gr.update(visible=False),
            outputs=[preview_loading],
            show_progress="hidden",
            queue=False,
        )

        # Pagina successiva preview
        btn_next_preview.click(
            fn=lambda: gr.update(visible=True),
            outputs=[preview_loading],
            show_progress="hidden",
            queue=False,
        ).then(
            fn=next_preview_page,
            inputs=[
                dataset,
                preview_selector,
                preview_page_state,
            ],
            outputs=[
                preview,
                preview_page_label,
                btn_prev_preview,
                btn_next_preview,
                preview_page_state,
            ],
            show_progress="hidden",
            queue=False,
        ).then(
            fn=lambda: gr.update(visible=False),
            outputs=[preview_loading],
            show_progress="hidden",
            queue=False,
        )

        # Pagina precedente preview
        btn_prev_preview.click(
            fn=lambda: gr.update(visible=True),
            outputs=[preview_loading],
            show_progress="hidden",
            queue=False,
        ).then(
            fn=prev_preview_page,
            inputs=[
                dataset,
                preview_selector,
                preview_page_state,
            ],
            outputs=[
                preview,
                preview_page_label,
                btn_prev_preview,
                btn_next_preview,
                preview_page_state,
            ],
            show_progress="hidden",
            queue=False,
        ).then(
            fn=lambda: gr.update(visible=False),
            outputs=[preview_loading],
            show_progress="hidden",
            queue=False,
        )