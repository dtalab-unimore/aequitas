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
    with gr.Tab("Metrics"):
        gr.Markdown("### Model Metrics Visualization")

        with gr.Row():
            with gr.Column(scale=1):
                metrics_task = gr.Dropdown(
                    choices=["Regression", "Classification", "Ranking"],
                    value=None,
                    label="Task Type",
                )
                metrics_year_range = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Year Range",
                    interactive=False,
                )
                metrics_model = gr.Dropdown(
                    choices=[],
                    value=[],
                    label="Model",
                    interactive=False,
                    multiselect=True,
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
                )
                view_metrics = gr.Button(
                    value="View Metrics",
                    variant="primary",
                    interactive=False,
                )

            with gr.Column(scale=4):
                with gr.Row():
                    metrics_table = gr.HTML(
                        value="",
                        label="Metrics Table",
                        elem_classes="metrics-table-container",
                    )

                with gr.Row(elem_classes="metrics-plot-panel", visible=False) as metrics_plot_panel:

                    with gr.Column(scale=1, min_width=210, elem_classes="metrics-plot-controls"):
                        metrics_metric_selector = gr.Dropdown(
                            choices=[],
                            value=None,
                            label="Metric",
                            interactive=False,
                        )
                        metrics_dataset_radio = gr.Radio(
                            choices=[],
                            value=None,
                            label="Dataset",
                            interactive=False,
                        )

                    with gr.Column(scale=5, elem_classes="metrics-plot-card"):
                        metrics_plot = gr.Plot(
                            value=None,
                            label="",
                            #visible=False,
                            elem_classes="metrics-plot-box",
                        )

        # ── helper────────────
        def _toggle_view_btn(datasets):
            return gr.update(interactive=bool(datasets))

        # ── wiring ──────────────────────────────────────────────────────────
        metrics_task.change(
            fn=on_metrics_task_change,
            inputs=[metrics_task],
            outputs=[
                metrics_year_range, metrics_model, metrics_config_pl,
                metrics_datasets,
                metrics_table,
                metrics_plot_panel,
                metrics_metric_selector, metrics_plot, metrics_dataset_radio,
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
                metrics_plot_panel,
                metrics_metric_selector, metrics_plot, metrics_dataset_radio
            ],
            show_progress="hidden",
            queue=False,
        )

        metrics_model.change(
            fn=on_metrics_model_change,
            inputs=[metrics_task, metrics_year_range, metrics_model],
            outputs=[
                metrics_config_pl, metrics_datasets,
                metrics_table,
                metrics_plot_panel,
                metrics_metric_selector, metrics_plot, metrics_dataset_radio,
            ],
            show_progress="hidden",
            queue=False,
        )

        metrics_config_pl.input(
            fn=on_metrics_config_pl_change,
            inputs=[metrics_task, metrics_year_range, metrics_model, metrics_config_pl],
            outputs=[
                metrics_datasets, metrics_table,
                metrics_plot_panel,
                metrics_metric_selector, metrics_plot, metrics_dataset_radio,
            ],
            show_progress="hidden",
            queue=False,
        )

        # ── enable/disable button─────────
        metrics_datasets.change(
            fn=_toggle_view_btn,
            inputs=[metrics_datasets],
            outputs=[view_metrics],
            show_progress="hidden", queue=False,
        )

        view_metrics.click(
            fn=on_metrics_datasets_change,
            inputs=[
                metrics_task, metrics_year_range,
                metrics_config_pl, metrics_datasets,
                metrics_metric_selector, metrics_dataset_radio,
            ],
            outputs=[
                metrics_table, metrics_plot_panel, metrics_metric_selector, metrics_plot,
                metrics_dataset_radio,
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