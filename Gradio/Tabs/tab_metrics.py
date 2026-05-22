import gradio as gr

from Gradio.metrics_view import (
    on_metrics_task_change,
    on_metrics_year_range_change,
    on_metrics_model_change,
    on_metrics_config_pl_change,
    update_metrics_plot,
    on_metrics_datasets_change,
)

def build_tab():
    with gr.Tab("Metrics", elem_id="main-tabs"):

        with gr.Row(equal_height=False):

            # ── LEFT: filters ────────────────────────────────────────────────
            with gr.Column(scale=1, elem_classes="metrics-controls-col"):

                metrics_task = gr.Dropdown(
                    choices=["Regression", "Classification", "Ranking"],
                    value=None,
                    label="Task Type",
                    elem_classes=["dropdown-single-compact"]
                )
                metrics_year_range = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Year Range",
                    interactive=False,
                    elem_classes=["dropdown-single-compact"]
                )
                metrics_model = gr.Dropdown(
                    choices=[],
                    value=[],
                    label="Model",
                    interactive=False,
                    multiselect=True,
                    elem_classes=["dropdown-multiselect-compact"]
                )
                metrics_config_pl = gr.CheckboxGroup(
                    choices=[],
                    value=[],
                    label="Configuration / Prediction Length",
                    interactive=False,
                )
                metrics_datasets = gr.Dropdown(
                    choices=[],
                    value=[],
                    label="Dataset",
                    multiselect=True,
                    interactive=False,
                    elem_classes=["dropdown-multiselect-compact"]
                )
                view_metrics = gr.Button(
                    value="View Metrics",
                    variant="primary",
                    interactive=False,
                )

            # ── RIGHT: results ───────────────────────────────────────────────
            with gr.Column(scale=4):

                # table
                with gr.Row():
                    with gr.Column(elem_classes="metrics-table-wrap"):
                        metrics_table = gr.HTML(
                            value=_empty_state_html(),
                            elem_classes="metrics-table-container",
                        )
                        with gr.Row(elem_classes="metrics-download-row"):
                            metrics_download = gr.DownloadButton(
                                label="Download CSV",
                                value=None,
                                visible=False,
                                elem_id="download-metrics-button",
                            )

                # plot panel — always visible, starts empty
                with gr.Row(elem_classes="metrics-plot-panel-wrap"):
                    with gr.Column(scale=1, min_width=200, elem_classes="metrics-plot-controls"):
                        metrics_metric_selector = gr.Dropdown(
                            choices=[],
                            value=None,
                            label="Metric",
                            interactive=False,
                            elem_classes=["dropdown-single-compact"]
                        )
                        metrics_dataset_radio = gr.CheckboxGroup(
                            choices=[],
                            value=[],
                            label="Dataset nel plot",
                            interactive=False,
                        )

                    with gr.Column(scale=5, elem_classes="metrics-plot-card"):
                        metrics_plot = gr.Plot(
                            value=None,
                            label="",
                            elem_classes="metrics-plot-box",
                        )

        # ── helper ──────────────────────────────────────────────────────────
        def _toggle_btn(datasets):
            return gr.update(interactive=bool(datasets))

        # ── wiring ──────────────────────────────────────────────────────────
        metrics_task.change(
            fn=on_metrics_task_change,
            inputs=[metrics_task],
            outputs=[
                metrics_year_range, metrics_model, metrics_config_pl,
                metrics_datasets, metrics_table,
                metrics_metric_selector, metrics_plot, metrics_dataset_radio,
                metrics_download
            ],
            show_progress="hidden",
            queue=False,
        )

        metrics_year_range.change(
            fn=on_metrics_year_range_change,
            inputs=[metrics_task, metrics_year_range],
            outputs=[
                metrics_model, metrics_config_pl, metrics_datasets,
                metrics_table,
                metrics_metric_selector, metrics_plot, metrics_dataset_radio,
                metrics_download
            ],
            show_progress="hidden",
            queue=False,
        )

        metrics_model.change(
            fn=on_metrics_model_change,
            inputs=[metrics_task, metrics_year_range, metrics_model],
            outputs=[
                metrics_config_pl, metrics_datasets, metrics_table,
                metrics_metric_selector, metrics_plot, metrics_dataset_radio,
                metrics_download
            ],
            show_progress="hidden",
            queue=False,
        )

        metrics_config_pl.input(
            fn=on_metrics_config_pl_change,
            inputs=[metrics_task, metrics_year_range, metrics_model, metrics_config_pl],
            outputs=[
                metrics_datasets, metrics_table,
                metrics_metric_selector, metrics_plot, metrics_dataset_radio,
                metrics_download
            ],
            show_progress="hidden",
            queue=False,
        )

        metrics_datasets.change(
            fn=_toggle_btn,
            inputs=[metrics_datasets],
            outputs=[view_metrics],
            show_progress="hidden",
            queue=False,
        )

        view_metrics.click(
            fn=on_metrics_datasets_change,
            inputs=[
                metrics_task, metrics_year_range,
                metrics_config_pl, metrics_datasets,
                metrics_metric_selector, metrics_dataset_radio,
            ],
            outputs=[
                metrics_table,
                metrics_metric_selector, metrics_plot, metrics_dataset_radio,
                metrics_download
            ],
            show_progress="hidden",
            queue=False,
        )

        metrics_metric_selector.input(
            fn=update_metrics_plot,
            inputs=[
                metrics_task, metrics_year_range, metrics_model,
                metrics_config_pl, metrics_datasets,
                metrics_metric_selector, metrics_dataset_radio,
            ],
            outputs=[metrics_plot],
            show_progress="hidden",
            queue=False,
        )

        metrics_dataset_radio.input(
            fn=update_metrics_plot,
            inputs=[
                metrics_task, metrics_year_range, metrics_model,
                metrics_config_pl, metrics_datasets,
                metrics_metric_selector, metrics_dataset_radio,
            ],
            outputs=[metrics_plot],
            show_progress="hidden",
            queue=False,
        )


def _empty_state_html(message="Select filters and click <b>View Metrics</b> to load results."):
    return f"""
    <div class="metrics-empty">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="1.4" stroke-linecap="round">
        <rect x="3" y="3" width="18" height="18" rx="2"/>
        <path d="M3 9h18M9 21V9"/>
      </svg>
      <span>{message}</span>
    </div>
    """
#
# import gradio as gr
#
# from Gradio.metrics_view import (
#     on_metrics_task_change,
#     on_metrics_year_range_change,
#     on_metrics_model_change,
#     on_metrics_config_pl_change,
#     update_metrics_plot,
#     on_metrics_datasets_change,
# )
#
# def build_tab():
#     with gr.Tab("Metrics", elem_id="main-tabs"):
#
#         with gr.Row(equal_height=False):
#
#             # ── LEFT: filters ────────────────────────────────────────────────
#             with gr.Column(scale=1, elem_classes="metrics-controls-col"):
#
#                 metrics_task = gr.Dropdown(
#                     choices=["Regression", "Classification", "Ranking"],
#                     value=None,
#                     label="Task Type",
#                     elem_classes=["dropdown-single-compact"]
#                 )
#                 metrics_year_range = gr.Dropdown(
#                     choices=[],
#                     value=None,
#                     label="Year Range",
#                     interactive=False,
#                     elem_classes=["dropdown-single-compact"]
#                 )
#                 metrics_model = gr.Dropdown(
#                     choices=[],
#                     value=[],
#                     label="Model",
#                     interactive=False,
#                     multiselect=True,
#                     elem_classes=["dropdown-multiselect-compact"]
#                 )
#                 metrics_config_pl = gr.CheckboxGroup(
#                     choices=[],
#                     value=[],
#                     label="Configuration / Prediction Length",
#                     interactive=False,
#                 )
#                 metrics_datasets = gr.Dropdown(
#                     choices=[],
#                     value=[],
#                     label="Dataset",
#                     multiselect=True,
#                     interactive=False,
#                     elem_classes=["dropdown-multiselect-compact"]
#                 )
#                 view_metrics = gr.Button(
#                     value="View Metrics",
#                     variant="primary",
#                     interactive=False,
#                 )
#
#             # ── RIGHT: results ───────────────────────────────────────────────
#             with gr.Column(scale=4):
#
#                 # table
#                 with gr.Row():
#                     with gr.Column(elem_classes="metrics-table-wrap"):
#                         metrics_table = gr.HTML(
#                             value=_empty_state_html(),
#                             elem_classes="metrics-table-container",
#                         )
#                         with gr.Row(elem_classes="metrics-download-row"):
#                             metrics_download = gr.DownloadButton(
#                                 label="Download CSV",
#                                 value=None,
#                                 visible=False,
#                                 elem_id="download-metrics-button",
#                             )
#
#                 # plot panel — always visible, starts empty
#                 with gr.Row(elem_classes="metrics-plot-panel-wrap"):
#                     with gr.Column(scale=1, min_width=200, elem_classes="metrics-plot-controls"):
#                         metrics_metric_selector = gr.Dropdown(
#                             choices=[],
#                             value=None,
#                             label="Metric",
#                             interactive=False,
#                             elem_classes=["dropdown-single-compact"]
#                         )
#
#                     with gr.Column(scale=5, elem_classes="metrics-plot-card"):
#                         metrics_plot = gr.Plot(
#                             value=None,
#                             label="",
#                             elem_classes="metrics-plot-box",
#                         )
#
#         # ── helper ──────────────────────────────────────────────────────────
#         def _toggle_btn(datasets):
#             return gr.update(interactive=bool(datasets))
#
#         # ── wiring ──────────────────────────────────────────────────────────
#         metrics_task.change(
#             fn=on_metrics_task_change,
#             inputs=[metrics_task],
#             outputs=[
#                 metrics_year_range, metrics_model, metrics_config_pl,
#                 metrics_datasets, metrics_table,
#                 metrics_metric_selector, metrics_plot,
#                 metrics_download
#             ],
#             show_progress="hidden",
#             queue=False,
#         )
#
#         metrics_year_range.change(
#             fn=on_metrics_year_range_change,
#             inputs=[metrics_task, metrics_year_range],
#             outputs=[
#                 metrics_model, metrics_config_pl, metrics_datasets,
#                 metrics_table,
#                 metrics_metric_selector, metrics_plot,
#                 metrics_download
#             ],
#             show_progress="hidden",
#             queue=False,
#         )
#
#         metrics_model.change(
#             fn=on_metrics_model_change,
#             inputs=[metrics_task, metrics_year_range, metrics_model],
#             outputs=[
#                 metrics_config_pl, metrics_datasets, metrics_table,
#                 metrics_metric_selector, metrics_plot,
#                 metrics_download
#             ],
#             show_progress="hidden",
#             queue=False,
#         )
#
#         metrics_config_pl.input(
#             fn=on_metrics_config_pl_change,
#             inputs=[metrics_task, metrics_year_range, metrics_model, metrics_config_pl],
#             outputs=[
#                 metrics_datasets, metrics_table,
#                 metrics_metric_selector, metrics_plot,
#                 metrics_download
#             ],
#             show_progress="hidden",
#             queue=False,
#         )
#
#         metrics_datasets.change(
#             fn=_toggle_btn,
#             inputs=[metrics_datasets],
#             outputs=[view_metrics],
#             show_progress="hidden",
#             queue=False,
#         )
#
#         view_metrics.click(
#             fn=on_metrics_datasets_change,
#             inputs=[
#                 metrics_task, metrics_year_range,
#                 metrics_config_pl, metrics_datasets,
#                 metrics_metric_selector,
#             ],
#             outputs=[
#                 metrics_table,
#                 metrics_metric_selector, metrics_plot,
#                 metrics_download
#             ],
#             show_progress="hidden",
#             queue=False,
#         )
#
#         metrics_metric_selector.input(
#             fn=update_metrics_plot,
#             inputs=[
#                 metrics_task, metrics_year_range, metrics_model,
#                 metrics_config_pl, metrics_datasets,
#                 metrics_metric_selector,
#             ],
#             outputs=[metrics_plot],
#             show_progress="hidden",
#             queue=False,
#         )
#
#
#
#
# def _empty_state_html(message="Select filters and click <b>View Metrics</b> to load results."):
#     return f"""
#     <div class="metrics-empty">
#       <svg width="48" height="48" viewBox="0 0 24 24" fill="none"
#            stroke="currentColor" stroke-width="1.4" stroke-linecap="round">
#         <rect x="3" y="3" width="18" height="18" rx="2"/>
#         <path d="M3 9h18M9 21V9"/>
#       </svg>
#       <span>{message}</span>
#     </div>
#     """