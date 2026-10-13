import gradio as gr

from Aequitas.data_load import (
    get_models,
    inspect_required_files,
    download_missing_files,
    reset_preview_page,
    next_preview_page,
    prev_preview_page,
    empty_preview_table_html,
    refresh_preview_year_filter,
    prepare_ticker_preview,
    reset_preview_page_from_ticker_state,
    refresh_preview_selector_and_filters,
    get_preview_file_tooltip,
)
from Aequitas.ui import section_header, loading_overlay


DATASETS = [
    ("[sp500] Standard & Poor's 500", "sp500"),
    ("[dji] Dow Jones Industrial Average", "dji"),
    ("[nasdaq100] NASDAQ-100", "nasdaq100"),
    ("[sxxp] STOXX Europe 600", "sxxp"),
    ("[sx5e] EURO STOXX 50", "sx5e"),
]

UI_EVENT = dict(show_progress="hidden", queue=False)


def build_tab():
    with gr.Tab("Download Data"):
        with gr.Column(elem_classes=["aq-tab"]):

            # ── Top row: controls + required files ────────────────────────
            with gr.Row(elem_classes=["aq-row", "download-top-row"]):

                with gr.Column(scale=1, min_width=260, elem_classes=["aq-card", "download-fixed-card"]):
                    gr.HTML(section_header("Configuration"))
                    with gr.Row():
                        dataset = gr.Dropdown(
                            choices=DATASETS,
                            label="Universe",
                            value=None,
                            elem_classes=["dropdown-single-compact"],
                        )
                        models = gr.Dropdown(
                            choices=get_models(),
                            label="Models",
                            value=[],
                            multiselect=True,
                            elem_classes=["dropdown-multiselect-compact"],
                        )

                    universe_targets_state = gr.State([])
                    market_targets_state = gr.State([])

                    btn_download_missing = gr.Button(
                        "Download Missing Files",
                        variant="primary",
                        interactive=False,
                        elem_classes=["aq-primary-btn"],
                    )

                with gr.Column(
                    scale=2,
                    elem_classes=["aq-card", "aq-loading-host", "download-fixed-card"],
                ):
                    gr.HTML(section_header("Required files"))

                    files_loading = gr.HTML(
                        loading_overlay("Downloading files..."),
                        visible=False,
                    )

                    files = gr.Textbox(
                        label=None,
                        interactive=False,
                        lines=6,
                        max_lines=6,
                        show_label=False,
                        elem_classes=["aq-log"],
                    )

            # ── Bottom row: full-width preview ────────────────────────────
            with gr.Row(elem_classes=["aq-row"]):
                with gr.Column(elem_classes=["aq-card", "aq-loading-host"]):

                    with gr.Row(elem_classes=["preview-header-row"]):
                        gr.HTML(section_header("Data preview"))

                        with gr.Row(elem_classes=["preview-header-controls"]):

                            preview_selector = gr.Dropdown(
                                label=None,
                                choices=[],
                                value=None,
                                show_label=False,
                                elem_id="preview-selector",
                                elem_classes=["dropdown-single-compact", "preview-control"],
                            )

                            ticker_input = gr.Textbox(
                                label=None,
                                show_label=False,
                                visible=False,
                                interactive=True,
                                placeholder="Insert Ticker",
                                elem_id="ticker-input",
                                elem_classes=["preview-control"],
                            )

                            preview_start_year = gr.Dropdown(
                                label=None,
                                choices=[],
                                value=None,
                                show_label=False,
                                visible=False,
                                elem_id="preview-start-year",
                                elem_classes=["dropdown-single-compact", "preview-control"],
                            )

                            with gr.Column(scale=0, elem_classes=["preview-tooltip-slot"]):
                                preview_file_tooltip = gr.HTML(
                                    value="",
                                    visible=True,
                                    elem_id="preview-file-tooltip-inline",
                                )

                    preview_page_state = gr.State(0)
                    ticker_preview_state = gr.State({"run": True, "ticker": None})

                    preview_loading = gr.HTML(
                        loading_overlay("Loading preview..."),
                        visible=False,
                    )

                    preview = gr.HTML(
                        value=empty_preview_table_html(),
                        elem_id="preview-table",
                    )

                    with gr.Row(elem_classes=["preview-pagination-outer"]):
                        with gr.Column(scale=1):
                            gr.HTML("")
                        with gr.Column(
                            scale=0,
                            min_width=0,
                            elem_classes=["preview-pagination-box"],
                        ):
                            with gr.Row(elem_classes=["preview-pagination-inner"]):
                                btn_prev_preview = gr.Button(
                                    "‹",
                                    interactive=False,
                                    elem_classes=["preview-round-btn"],
                                    scale=0,
                                    min_width=32,
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
                                    min_width=32,
                                )
                        with gr.Column(scale=1):
                            gr.HTML("")

        # ── Wiring ────────────────────────────────────────────────────────
        preview_inputs = [dataset, preview_selector, preview_start_year, ticker_input]
        preview_outputs = [
            preview,
            preview_page_label,
            btn_prev_preview,
            btn_next_preview,
            preview_page_state,
        ]
        files_outputs = [
            files,
            btn_download_missing,
            universe_targets_state,
            market_targets_state,
        ]
        selector_inputs = [dataset, universe_targets_state, market_targets_state]
        selector_outputs = [preview_selector, preview_start_year, ticker_input]

        def show(component):
            return dict(fn=lambda: gr.update(visible=True), outputs=[component], **UI_EVENT)

        def hide(component):
            return dict(fn=lambda: gr.update(visible=False), outputs=[component], **UI_EVENT)

        for component in (dataset, models):
            component.change(
                fn=inspect_required_files,
                inputs=[dataset, models],
                outputs=files_outputs,
                **UI_EVENT,
            ).then(
                fn=refresh_preview_selector_and_filters,
                inputs=selector_inputs,
                outputs=selector_outputs,
                **UI_EVENT,
            ).then(
                fn=get_preview_file_tooltip,
                inputs=[preview_selector],
                outputs=[preview_file_tooltip],
                **UI_EVENT,
            ).then(
                fn=reset_preview_page,
                inputs=preview_inputs,
                outputs=preview_outputs,
                **UI_EVENT,
            )

        btn_download_missing.click(
            **show(files_loading),
        ).then(
            fn=download_missing_files,
            inputs=[dataset, models, universe_targets_state, market_targets_state],
            outputs=files_outputs,
            show_progress="full",
            queue=True,
        ).then(
            **hide(files_loading),
        ).then(
            fn=refresh_preview_selector_and_filters,
            inputs=selector_inputs,
            outputs=selector_outputs,
            **UI_EVENT,
        ).then(
            fn=reset_preview_page,
            inputs=preview_inputs,
            outputs=preview_outputs,
            **UI_EVENT,
        )

        preview_selector.change(
            **show(preview_loading),
        ).then(
            fn=refresh_preview_year_filter,
            inputs=[dataset, preview_selector],
            outputs=[preview_start_year, ticker_input],
            **UI_EVENT,
        ).then(
            fn=get_preview_file_tooltip,
            inputs=[preview_selector],
            outputs=[preview_file_tooltip],
            **UI_EVENT,
        ).then(
            fn=reset_preview_page,
            inputs=preview_inputs,
            outputs=preview_outputs,
            **UI_EVENT,
        ).then(
            **hide(preview_loading),
        )

        preview_start_year.change(
            **show(preview_loading),
        ).then(
            fn=reset_preview_page,
            inputs=preview_inputs,
            outputs=preview_outputs,
            **UI_EVENT,
        ).then(
            **hide(preview_loading),
        )

        ticker_input.change(
            fn=prepare_ticker_preview,
            inputs=[ticker_input],
            outputs=[ticker_preview_state, preview_loading],
            **UI_EVENT,
        ).then(
            fn=reset_preview_page_from_ticker_state,
            inputs=[ticker_preview_state, dataset, preview_selector, preview_start_year],
            outputs=[*preview_outputs, preview_loading],
            trigger_mode="always_last",
            **UI_EVENT,
        )

        page_inputs = [dataset, preview_selector, preview_page_state, preview_start_year, ticker_input]

        btn_next_preview.click(
            fn=next_preview_page,
            inputs=page_inputs,
            outputs=preview_outputs,
            **UI_EVENT,
        )

        btn_prev_preview.click(
            fn=prev_preview_page,
            inputs=page_inputs,
            outputs=preview_outputs,
            **UI_EVENT,
        )
