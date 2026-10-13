import gradio as gr

from Aequitas.metrics_view import (
    on_metrics_task_change,
    on_metrics_year_range_change,
    on_metrics_model_change,
    on_metrics_config_pl_change,
    update_metrics_plot,
    on_metrics_datasets_change,
    _empty_state_html,
)
from Aequitas.ui import section_header

UI_EVENT = dict(show_progress="hidden", queue=False)


def build_tab():
    with gr.Tab("Metrics"):
        with gr.Column(elem_classes=["aq-tab"]):

            with gr.Row(equal_height=False, elem_classes=["aq-row"]):

                # ── LEFT: filters ────────────────────────────────────────────
                with gr.Column(scale=1, elem_classes=["aq-card"]):
                    gr.HTML(section_header("Filters"))

                    metrics_task = gr.Dropdown(
                        choices=["Regression", "Classification", "Ranking"],
                        value=None,
                        label="Task Type",
                        elem_classes=["dropdown-single-compact"],
                    )
                    metrics_year_range = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Year Range",
                        interactive=False,
                        elem_classes=["dropdown-single-compact"],
                    )
                    metrics_model = gr.Dropdown(
                        choices=[],
                        value=[],
                        label="Model",
                        interactive=False,
                        multiselect=True,
                        elem_classes=["dropdown-multiselect-compact"],
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
                        elem_classes=["dropdown-multiselect-compact"],
                    )
                    view_metrics = gr.Button(
                        value="View Metrics",
                        variant="primary",
                        interactive=False,
                        elem_classes=["aq-primary-btn"],
                    )

                # ── RIGHT: results ───────────────────────────────────────────
                with gr.Column(scale=4, elem_classes=["aq-stack"]):

                    with gr.Column(elem_classes=["aq-card"]):
                        gr.HTML(section_header("Metrics table"))
                        metrics_table = gr.HTML(value=_empty_state_html())
                        with gr.Row(elem_classes=["aq-chip-row"]):
                            metrics_download = gr.DownloadButton(
                                label="Download CSV ↓",
                                value=None,
                                visible=False,
                                elem_classes=["aq-chip-btn"],
                            )

                    # plot panel — always visible, starts empty
                    with gr.Column(elem_classes=["aq-card"]):
                        gr.HTML(section_header("Metrics plot"))
                        with gr.Row():
                            with gr.Column(scale=1, min_width=200):
                                metrics_metric_selector = gr.Dropdown(
                                    choices=[],
                                    value=None,
                                    label="Metric",
                                    interactive=False,
                                    elem_classes=["dropdown-single-compact"],
                                )
                                metrics_dataset_radio = gr.CheckboxGroup(
                                    choices=[],
                                    value=[],
                                    label="Datasets in plot",
                                    interactive=False,
                                )

                            with gr.Column(scale=5):
                                metrics_plot = gr.Plot(
                                    value=None,
                                    label="",
                                    show_label=False,
                                    elem_classes=["metrics-plot-box"],
                                )

        # ── wiring ──────────────────────────────────────────────────────────
        plot_outputs = [metrics_metric_selector, metrics_plot, metrics_dataset_radio, metrics_download]
        plot_inputs = [
            metrics_task, metrics_year_range, metrics_model,
            metrics_config_pl, metrics_datasets,
            metrics_metric_selector, metrics_dataset_radio,
        ]

        metrics_task.change(
            fn=on_metrics_task_change,
            inputs=[metrics_task],
            outputs=[metrics_year_range, metrics_model, metrics_config_pl,
                     metrics_datasets, metrics_table, *plot_outputs],
            **UI_EVENT,
        )

        metrics_year_range.change(
            fn=on_metrics_year_range_change,
            inputs=[metrics_task, metrics_year_range],
            outputs=[metrics_model, metrics_config_pl, metrics_datasets,
                     metrics_table, *plot_outputs],
            **UI_EVENT,
        )

        metrics_model.change(
            fn=on_metrics_model_change,
            inputs=[metrics_task, metrics_year_range, metrics_model],
            outputs=[metrics_config_pl, metrics_datasets, metrics_table, *plot_outputs],
            **UI_EVENT,
        )

        metrics_config_pl.input(
            fn=on_metrics_config_pl_change,
            inputs=[metrics_task, metrics_year_range, metrics_model, metrics_config_pl],
            outputs=[metrics_datasets, metrics_table, *plot_outputs],
            **UI_EVENT,
        )

        metrics_datasets.change(
            fn=lambda datasets: gr.update(interactive=bool(datasets)),
            inputs=[metrics_datasets],
            outputs=[view_metrics],
            **UI_EVENT,
        )

        view_metrics.click(
            fn=on_metrics_datasets_change,
            inputs=[
                metrics_task, metrics_year_range,
                metrics_config_pl, metrics_datasets,
                metrics_metric_selector, metrics_dataset_radio,
            ],
            outputs=[metrics_table, *plot_outputs],
            **UI_EVENT,
        )

        for component in (metrics_metric_selector, metrics_dataset_radio):
            component.input(
                fn=update_metrics_plot,
                inputs=plot_inputs,
                outputs=[metrics_plot],
                **UI_EVENT,
            )
