"""
Evaluation tab – UI only.

All business logic lives in:
    eval_config.py   – constants and paths
    eval_scan.py     – PKL scanning and record utilities
    eval_form.py     – cascade-dropdown update logic
    eval_plot.py     – matplotlib plot builders
    eval_output.py   – HTML/CSV formatting and LLM explanation
    eval_run.py      – evaluation runner + Gradio loading helpers
"""
from functools import partial

import gradio as gr
import pandas as pd

from Gradio.eval_config import EVAL_MODES
from Gradio.eval_form import update_eval_form
from Gradio.eval_output import empty_evaluation_table_html
from Gradio.eval_run import (
    run_evaluation,
    show_evaluation_loading,
    hide_evaluation_loading,
    show_chatgpt_explanation_loading,
    hide_chatgpt_explanation_loading,
    explain_evaluation_with_chatgpt,
)


def build_tab():
    with gr.Tab("Evaluation", elem_id="main-tabs"):
        with gr.Column():

            # ── main two-column row ──────────────────────────────────────────
            with gr.Row(elem_classes=["evaluation-mode-row"]):

                # LEFT – assessment type
                with gr.Column(scale=1, elem_classes=["evaluation-mode-col"]):
                    with gr.Group(elem_classes=["evaluation-mode-card"]):
                        gr.Markdown(
                            """
                            <div class="evaluation-mode-title-wrap">
                                <div class="evaluation-section-title evaluation-mode-title">
                                    <span class="evaluation-section-dot"></span>
                                    Assessment type
                                </div>
                            </div>
                            """
                        )
                        eval_mode = gr.Radio(
                            choices=EVAL_MODES,
                            value=None,
                            label=None,
                            interactive=True,
                            show_label=False,
                            container=False,
                            elem_classes=["evaluation-radio"],
                        )
                    run_btn = gr.Button(
                        "Run evaluation",
                        variant="primary",
                        interactive=False,
                        elem_id="evaluation-run-button",
                    )

                # RIGHT – params group
                with gr.Column(scale=4, elem_classes=["evaluation-params-col"]):
                    with gr.Group(
                        visible=False,
                        elem_classes=["evaluation-panel"],
                    ) as params_group:

                        # ── Result selection ─────────────────────────────────
                        gr.Markdown(
                            """
                            <div class="evaluation-section-title">
                                <span class="evaluation-section-dot"></span>
                                Result selection
                            </div>
                            """
                        )

                        with gr.Row(elem_classes=["evaluation-form-row"]):
                            task_type = gr.Dropdown(
                                choices=[], value=None,
                                label="Task type", interactive=False,
                            )
                            model = gr.Dropdown(
                                choices=[], value=[],
                                label="Model", interactive=False,
                                multiselect=True,
                                elem_classes=["dropdown-multiselect-compact"],
                            )
                            universe = gr.Dropdown(
                                choices=[], value=None,
                                label="Universe", interactive=False,
                            )
                            configuration = gr.Dropdown(
                                choices=[], value=[],
                                label="Configuration", interactive=False,
                                multiselect=True,
                                elem_classes=["dropdown-multiselect-compact"],
                            )

                        # ── Evaluation window ────────────────────────────────
                        gr.Markdown(
                            """
                            <div class="evaluation-section-title evaluation-section-spaced">
                                <span class="evaluation-section-dot"></span>
                                Evaluation window
                            </div>
                            """
                        )

                        with gr.Row(elem_classes=["evaluation-form-row"]):
                            year_start = gr.Dropdown(
                                choices=[], value=None,
                                label="Year start", interactive=False,
                            )
                            year_end = gr.Dropdown(
                                choices=[], value=None,
                                label="Year end", interactive=False,
                            )
                            length_pair = gr.CheckboxGroup(
                                choices=[], value=[],
                                label="Sequence / prediction length",
                                interactive=False,
                                elem_classes=["evaluation-length-checkbox"],
                            )

                        # ── Portfolio construction (conditional) ─────────────
                        with gr.Group(
                            visible=False,
                            elem_classes=["evaluation-portfolio-card"],
                        ) as top_short_row:
                            gr.Markdown(
                                """
                                <div class="evaluation-section-title">
                                    <span class="evaluation-section-dot"></span>
                                    Portfolio construction
                                </div>
                                """
                            )
                            with gr.Row(elem_classes=["evaluation-form-row",
                                                      "evaluation-portfolio-compact-row"]):
                                with gr.Column(scale=0, min_width=180,
                                               elem_classes=["evaluation-portfolio-small-col"]):
                                    top_k = gr.Dropdown(
                                        choices=[], value=None,
                                        label="Top K", interactive=False,
                                    )
                                with gr.Column(scale=0, min_width=180,
                                               elem_classes=["evaluation-portfolio-small-col"]):
                                    short_k = gr.Dropdown(
                                        choices=[], value=None,
                                        label="Short K", interactive=False,
                                    )

            # ── Output area ──────────────────────────────────────────────────
            with gr.Row(elem_classes=["evaluation-output-row"]):
                with gr.Column(
                    scale=1,
                    elem_classes=["evaluation-output-card", "evaluation-log-card"],
                ):
                    gr.Markdown(
                        """
                        <div class="evaluation-output-header">
                            <span>Execution log</span>
                        </div>
                        """
                    )
                    with gr.Accordion("Show command details", open=False,
                                      elem_classes=["evaluation-log-accordion"]):
                        log_box = gr.Textbox(
                            label=None, interactive=False,
                            lines=12, max_lines=12,
                            show_label=False, elem_id="execution-log",
                        )

                with gr.Column(
                    scale=4,
                    elem_classes=["evaluation-output-card", "evaluation-table-card",
                                  "evaluation-loading-host"],
                ):
                    gr.Markdown(
                        """
                        <div class="evaluation-output-header">
                            <span>Evaluation output</span>
                        </div>
                        """
                    )
                    table_loading = gr.HTML(
                        """
                        <div class="evaluation-loading-overlay">
                            <div class="evaluation-loading-box">
                                <div class="evaluation-loading-spinner"></div>
                                <div class="evaluation-loading-text">Running evaluation...</div>
                            </div>
                        </div>
                        """,
                        visible=False,
                    )
                    output_table = gr.HTML(
                        value=empty_evaluation_table_html(),
                        elem_id="evaluation-output-table",
                    )
                    with gr.Row(elem_classes=["metrics-download-row",
                                              "evaluation-download-row"]):
                        download_evaluation_button = gr.DownloadButton(
                            "Download CSV ⭣",
                            value=None, visible=False,
                            elem_id="download-evaluation-button",
                        )
                        explain_evaluation_button = gr.Button(
                            "Explain with ChatGPT",
                            visible=False, interactive=False,
                            elem_id="explain-evaluation-button",
                        )

                    chatgpt_loading = gr.HTML(
                        """
                        <div class="evaluation-chatgpt-inline-loading">
                            <div class="evaluation-loading-spinner"></div>
                            <div class="evaluation-loading-text">Generating explanation...</div>
                        </div>
                        """,
                        visible=False,
                    )
                    with gr.Group(
                        visible=False,
                        elem_classes=["evaluation-chatgpt-explanation-card"],
                    ) as chatgpt_explanation_group:
                        chatgpt_explanation = gr.Markdown(
                            value="",
                            elem_id="evaluation-chatgpt-explanation",
                        )

            # ── Plot ─────────────────────────────────────────────────────────
            with gr.Group(
                visible=True,
                elem_classes=["evaluation-output-card", "evaluation-plot-card",
                               "evaluation-loading-host"],
            ):
                gr.Markdown(
                    """
                    <div class="evaluation-output-header">
                        <span>Evaluation plot</span>
                    </div>
                    """
                )
                plot_loading = gr.HTML(
                    """
                    <div class="evaluation-loading-overlay">
                        <div class="evaluation-loading-box">
                            <div class="evaluation-loading-spinner"></div>
                            <div class="evaluation-loading-text">Building plot...</div>
                        </div>
                    </div>
                    """,
                    visible=False,
                )
                plot_output = gr.Plot(
                    label="", show_label=False,
                    visible=False, container=False,
                    elem_id="evaluation-plot",
                )

        # ── State ────────────────────────────────────────────────────────────
        evaluation_output_state = gr.State(value=pd.DataFrame())

        # ── Input / output lists ─────────────────────────────────────────────
        form_inputs = [
            eval_mode, task_type, model, universe, configuration,
            year_start, year_end, length_pair, top_k, short_k,
        ]
        form_outputs = [
            params_group,
            task_type, model, universe, configuration,
            year_start, year_end, length_pair,
            top_short_row,
            top_k, short_k,
            run_btn,
        ]

        # ── Bind form inputs ─────────────────────────────────────────────────
        def bind_form_input(component, changed_field):
            component.input(
                fn=partial(update_eval_form, changed_field=changed_field),
                inputs=form_inputs,
                outputs=form_outputs,
                show_progress="hidden",
                queue=False,
            )

        bind_form_input(eval_mode,     "eval_mode")
        bind_form_input(task_type,     "task_type")
        bind_form_input(model,         "model")
        bind_form_input(universe,      "universe")
        bind_form_input(configuration, "configuration")
        bind_form_input(year_start,    "year_start")
        bind_form_input(year_end,      "year_end")
        bind_form_input(length_pair,   "length_pair")
        bind_form_input(top_k,         "top_k")
        bind_form_input(short_k,       "short_k")

        # ── Run button chain ─────────────────────────────────────────────────
        run_event = run_btn.click(
            fn=show_evaluation_loading,
            inputs=[],
            outputs=[
                table_loading, plot_loading, run_btn,
                download_evaluation_button, explain_evaluation_button,
                chatgpt_explanation_group, chatgpt_explanation,
            ],
            show_progress="hidden",
            queue=False,
        )

        run_event = run_event.then(
            fn=run_evaluation,
            inputs=form_inputs,
            outputs=[
                log_box, output_table, download_evaluation_button,
                explain_evaluation_button, plot_output,
                evaluation_output_state,
            ],
            show_progress="hidden",
        )

        run_event.then(
            fn=hide_evaluation_loading,
            inputs=[],
            outputs=[table_loading, plot_loading, run_btn,
                     explain_evaluation_button],
            show_progress="hidden",
            queue=False,
        )

        # ── Explain button chain ─────────────────────────────────────────────
        explain_event = explain_evaluation_button.click(
            fn=show_chatgpt_explanation_loading,
            inputs=[],
            outputs=[
                chatgpt_loading, explain_evaluation_button,
                chatgpt_explanation_group, chatgpt_explanation,
            ],
            show_progress="hidden",
            queue=False,
        )

        explain_event = explain_event.then(
            fn=explain_evaluation_with_chatgpt,
            inputs=[evaluation_output_state, eval_mode],
            outputs=[chatgpt_explanation],
            show_progress="hidden",
        )

        explain_event.then(
            fn=hide_chatgpt_explanation_loading,
            inputs=[],
            outputs=[chatgpt_loading, explain_evaluation_button,
                     chatgpt_explanation_group],
            show_progress="hidden",
            queue=False,
        )