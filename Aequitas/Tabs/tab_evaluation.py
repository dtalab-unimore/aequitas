"""
Evaluation tab – UI only.

All business logic lives in:
    eval_config.py   – constants and paths
    eval_scan.py     – PKL scanning and record utilities
    eval_form.py     – cascade-dropdown update logic
    eval_plot.py     – matplotlib plot builders
    eval_output.py   – HTML/CSV formatting and LLM explanation
    eval_run.py      – evaluation runner + Aequitas loading helpers
"""
from functools import partial

import gradio as gr
import pandas as pd

from Aequitas.eval_config import EVAL_MODES
from Aequitas.eval_form import update_eval_form
from Aequitas.eval_output import (
    explain_evaluation_with_chatgpt,
    empty_evaluation_table_html
)
from Aequitas.eval_run import (
    run_evaluation,
    show_evaluation_loading,
    restore_evaluation_view,
    resume_evaluation,
    show_chatgpt_explanation_loading,
    hide_chatgpt_explanation_loading,
)
from Aequitas.ui import section_header, loading_overlay, loading_inline, AUTOSCROLL_LOG_JS


def build_tab():
    # Job id of this browser's latest evaluation, kept in localStorage across
    # page reloads (declared outside the tab, which Gradio mounts lazily).
    eval_job_state = gr.BrowserState(None, storage_key="aequitas_eval_job")

    with gr.Tab("Evaluation") as eval_tab:
        resume_token = gr.State(None)
        # True once this page has shown a run (restored or started here).
        run_shown_state = gr.State(False)

        with gr.Column(elem_classes=["aq-tab"]):

            with gr.Row(elem_classes=["aq-row", "evaluation-main-layout"]):

                # ── LEFT COLUMN ─────────────────────────────────────────────
                with gr.Column(scale=1, elem_classes=["aq-stack"]):

                    with gr.Column(elem_classes=["aq-card", "evaluation-mode-card"]):
                        gr.HTML(section_header("Assessment type"))

                        eval_mode = gr.Radio(
                            choices=EVAL_MODES,
                            value=None,
                            label=None,
                            interactive=True,
                            show_label=False,
                            container=False,
                            elem_classes=["evaluation-radio"],
                        )

                    with gr.Column(visible=False, elem_classes=["aq-card"]) as params_group:
                        gr.HTML(section_header("Result selection"))

                        with gr.Row():
                            task_type = gr.Dropdown(
                                choices=[],
                                value=[],
                                label="Task type",
                                interactive=False,
                                multiselect=True,
                                elem_classes=["dropdown-multiselect-compact"],
                            )
                            model = gr.Dropdown(
                                choices=[],
                                value=[],
                                label="Model",
                                interactive=False,
                                multiselect=True,
                                elem_classes=["dropdown-multiselect-compact"],
                            )

                        with gr.Row():
                            universe = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="Universe",
                                interactive=False,
                                elem_classes=["dropdown-single-compact"],
                            )
                            configuration = gr.Dropdown(
                                choices=[],
                                value=[],
                                label="Configuration",
                                interactive=False,
                                multiselect=True,
                                elem_classes=["dropdown-multiselect-compact"],
                            )

                        gr.HTML(section_header("Evaluation window", spaced=True))

                        with gr.Row():
                            year_start = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="Year start",
                                interactive=False,
                                elem_classes=["dropdown-single-compact"],
                            )
                            year_end = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="Year end",
                                interactive=False,
                                elem_classes=["dropdown-single-compact"],
                            )

                        length_pair = gr.CheckboxGroup(
                            choices=[],
                            value=[],
                            label="Sequence / prediction length",
                            interactive=False,
                        )

                        with gr.Column(visible=False, elem_classes=["aq-plain"]) as top_short_row:
                            gr.HTML(section_header("Portfolio construction", spaced=True))

                            with gr.Row():
                                top_k = gr.Dropdown(
                                    choices=[],
                                    value=None,
                                    label="Top K",
                                    interactive=False,
                                    elem_classes=["dropdown-single-compact"],
                                )
                                short_k = gr.Dropdown(
                                    choices=[],
                                    value=None,
                                    label="Short K",
                                    interactive=False,
                                    elem_classes=["dropdown-single-compact"],
                                )

                    run_btn = gr.Button(
                        "Run evaluation",
                        variant="primary",
                        interactive=False,
                        elem_classes=["aq-primary-btn"],
                    )

                    with gr.Column(elem_classes=["aq-card"]):
                        gr.HTML(section_header("Execution log"))

                        with gr.Accordion(
                            "Show command details",
                            open=False,
                            elem_classes=["aq-accordion"],
                        ):
                            log_box = gr.Textbox(
                                label=None,
                                interactive=False,
                                lines=12,
                                max_lines=12,
                                show_label=False,
                                elem_classes=["aq-log", "aq-log-autoscroll"],
                            )

                # ── RIGHT COLUMN ────────────────────────────────────────────
                with gr.Column(scale=3, elem_classes=["aq-stack"]):

                    with gr.Column(elem_classes=["aq-card", "aq-loading-host"]):
                        gr.HTML(section_header("Evaluation output"))

                        table_loading = gr.HTML(
                            loading_overlay("Running evaluation..."),
                            visible=False,
                        )

                        output_table = gr.HTML(
                            value=empty_evaluation_table_html(),
                            elem_id="evaluation-output-table",
                        )

                        with gr.Row(elem_classes=["aq-chip-row"]):
                            download_evaluation_button = gr.DownloadButton(
                                "Download CSV ↓",
                                value=None,
                                visible=False,
                                elem_classes=["aq-chip-btn"],
                            )

                            explain_evaluation_button = gr.Button(
                                "Explain with ChatGPT",
                                visible=False,
                                interactive=False,
                                elem_classes=["aq-chip-btn"],
                            )

                        chatgpt_loading = gr.HTML(
                            loading_inline("Generating explanation..."),
                            visible=False,
                        )

                        with gr.Column(visible=False, elem_classes=["aq-plain"]) as chatgpt_explanation_group:
                            chatgpt_explanation = gr.Markdown(
                                value="",
                                elem_id="evaluation-chatgpt-explanation",
                                buttons="copy",
                            )

                    with gr.Column(elem_classes=["aq-card", "aq-loading-host", "evaluation-plot-card"]):
                        gr.HTML(section_header("Evaluation plot"))

                        plot_loading = gr.HTML(
                            loading_overlay("Building plot..."),
                            visible=False,
                        )

                        plot_output = gr.Plot(
                            label="",
                            show_label=False,
                            visible=False,
                            container=False,
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

        bind_form_input(eval_mode, "eval_mode")
        bind_form_input(task_type, "task_type")
        bind_form_input(model, "model")
        bind_form_input(universe, "universe")
        bind_form_input(configuration, "configuration")
        bind_form_input(year_start, "year_start")
        bind_form_input(year_end, "year_end")
        bind_form_input(length_pair, "length_pair")
        bind_form_input(top_k, "top_k")
        bind_form_input(short_k, "short_k")

        # ── Run button chain ─────────────────────────────────────────────────
        run_outputs = [
            log_box, output_table, download_evaluation_button,
            explain_evaluation_button, plot_output, evaluation_output_state,
            eval_job_state, table_loading, plot_loading, run_btn,
        ]

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
            outputs=run_outputs,
            show_progress="hidden",
        )

        run_btn.click(fn=lambda: True, outputs=[run_shown_state], show_progress="hidden", queue=False)

        # After a page reload, opening the tab shows the evaluation this
        # browser started (if any); see tab_train.py for why this runs on
        # tab select and streams from resume_token's change event.
        eval_tab.select(
            fn=restore_evaluation_view,
            inputs=[eval_job_state, run_shown_state],
            outputs=[eval_mode, *form_outputs, resume_token, run_shown_state],
            show_progress="hidden",
        )

        resume_token.change(
            fn=resume_evaluation,
            inputs=[eval_job_state],
            outputs=run_outputs,
            show_progress="hidden",
        )

        log_box.change(fn=None, js=AUTOSCROLL_LOG_JS, show_progress="hidden", queue=False)

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
