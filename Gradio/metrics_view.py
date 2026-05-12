import os
import json
import re
import plotly.graph_objects as go
import gradio as gr
import pandas as pd
import tempfile
import uuid

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

SEED = "42"

DATASET_PALETTES = {
    "nasdaq100": ["#c0392b", "#e74c3c", "#f1948a", "#fadbd8"],
    "sp500": ["#1a5276", "#2980b9", "#7fb3d3", "#aed6f1"],
    "dji": ["#1e8449", "#27ae60", "#82e0aa", "#a9dfbf"],
    "sx5e": ["#6c3483", "#8e44ad", "#c39bd3", "#d7bde2"],
    "sxxp": ["#d35400", "#e67e22", "#f0b27a", "#fad7a0"],
    "_default": ["#2c3e50", "#566573", "#99a3a4", "#ccd1d1"],
}


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

def _hide_metrics_download():
    return gr.update(value=None, visible=False)


def _safe_filename_part(value):
    value = str(value or "metrics")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "metrics"


def build_metrics_export_csv(df, datasets, metric_cols, row_keys, task_type, year_range):
    datasets = [d for d in datasets if d in set(df["dataset"])]

    rows = []

    for keys, group in (
        df.sort_values(row_keys)
          .groupby(row_keys, dropna=False, sort=False)
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)

        key_map = dict(zip(row_keys, keys))

        row = {
            "Model": key_map.get("model"),
            "Config": key_map.get("config"),
            "SL": key_map.get("sl"),
            "PL": key_map.get("pl"),
        }

        for ds in datasets:
            ds_row = group[group["dataset"] == ds]

            for metric in metric_cols:
                col_name = f"{str(ds).upper()} - {metric}"

                if ds_row.empty:
                    row[col_name] = ""
                else:
                    row[col_name] = fmt_metric(ds_row.iloc[0].get(metric))

        rows.append(row)

    export_df = pd.DataFrame(rows)

    out_dir = os.path.join(tempfile.gettempdir(), "finbench_metrics_exports")
    os.makedirs(out_dir, exist_ok=True)

    filename = (
        f"metrics_"
        f"{_safe_filename_part(task_type)}_"
        f"{_safe_filename_part(year_range)}_"
        f"{uuid.uuid4().hex[:8]}.csv"
    )

    path = os.path.join(out_dir, filename)
    export_df.to_csv(path, index=False, encoding="utf-8-sig")

    return path


# ── colour helpers ───────────────────────────────────────────────────────────

def get_dataset_colors(dataset: str, n: int) -> list:
    palette = DATASET_PALETTES.get(dataset.lower(), DATASET_PALETTES["_default"])
    if n <= len(palette):
        return palette[:n]
    return [palette[i % len(palette)] for i in range(n)]


# ── path helpers ─────────────────────────────────────────────────────────────

def task_dir(task_type: str) -> str:
    return os.path.join(PROJECT_ROOT, task_type)


def model_results_dir(task_type: str, model_name: str) -> str:
    return os.path.join(PROJECT_ROOT, task_type, model_name, "results")


def seed_dir(task_type: str, model_name: str, dataset: str, config: str) -> str:
    return os.path.join(model_results_dir(task_type, model_name), dataset, config, SEED)


def year_range_dir(task_type: str, model_name: str, dataset: str, config: str,
                   year_range: str) -> str:
    return os.path.join(seed_dir(task_type, model_name, dataset, config), year_range)


# ── parsing helpers ──────────────────────────────────────────────────────────

def parse_metrics_file(file_name: str):
    m = re.match(r"metrics_sl(?P<sl>\d+)_pl(?P<pl>\d+)\.json$", file_name)
    return m.groupdict() if m else None


def parse_config_pl_option(option: str):
    """'MODEL | CONFIG | pl10'  →  (model, config, '10')"""
    if not option:
        return None, None, None
    parts = [p.strip() for p in option.split("|")]
    if len(parts) != 3 or not parts[2].startswith("pl"):
        return None, None, None
    return parts[0], parts[1], parts[2][2:]


def is_year_range_folder(name: str) -> bool:
    return bool(re.match(r"^y\d{4}_y\d{4}$", name))


def unique_sorted(values, numeric=False):
    values = [v for v in values if v is not None and v != ""]
    return sorted(set(values), key=lambda x: int(x) if numeric else x)


# ── listing helpers ──────────────────────────────────────────────────────────

def list_year_ranges(task_type: str) -> list:
    if not task_type:
        return []
    base = task_dir(task_type)
    if not os.path.isdir(base):
        return []
    ranges = set()
    for model_name in os.listdir(base):
        res = model_results_dir(task_type, model_name)
        if not os.path.isdir(res):
            continue
        for dataset in os.listdir(res):
            ds_path = os.path.join(res, dataset)
            if not os.path.isdir(ds_path):
                continue
            for config in os.listdir(ds_path):
                s_path = seed_dir(task_type, model_name, dataset, config)
                if not os.path.isdir(s_path):
                    continue
                for folder in os.listdir(s_path):
                    yr = os.path.join(s_path, folder)
                    if os.path.isdir(yr) and is_year_range_folder(folder):
                        ranges.add(folder)
    return sorted(ranges)


def list_models_for_range(task_type: str, year_range: str) -> list:
    if not task_type or not year_range:
        return []
    base = task_dir(task_type)
    if not os.path.isdir(base):
        return []
    models = set()
    for model_name in os.listdir(base):
        res = model_results_dir(task_type, model_name)
        if not os.path.isdir(res):
            continue
        found = False
        for dataset in os.listdir(res):
            ds_path = os.path.join(res, dataset)
            if not os.path.isdir(ds_path):
                continue
            for config in os.listdir(ds_path):
                yr = year_range_dir(task_type, model_name, dataset, config, year_range)
                if os.path.isdir(yr):
                    models.add(model_name)
                    found = True
                    break
            if found:
                break
    return sorted(models)


def list_config_pl_options(task_type: str, year_range: str, model_names: list) -> list:
    if not task_type or not year_range or not model_names:
        return []
    options = set()
    for model_name in model_names:
        res = model_results_dir(task_type, model_name)
        if not os.path.isdir(res):
            continue
        for dataset in os.listdir(res):
            ds_path = os.path.join(res, dataset)
            if not os.path.isdir(ds_path):
                continue
            for config in os.listdir(ds_path):
                yr = year_range_dir(task_type, model_name, dataset, config, year_range)
                if not os.path.isdir(yr):
                    continue
                for fname in os.listdir(yr):
                    p = parse_metrics_file(fname)
                    if p:
                        options.add(f"{model_name} | {config} | pl{p['pl']}")
    return sorted(options)


def list_datasets_for_config_pl_options(task_type, year_range, config_pl_options) -> list:
    if not task_type or not year_range or not config_pl_options:
        return []
    triples = [parse_config_pl_option(o) for o in config_pl_options]
    triples = [(m, c, pl) for m, c, pl in triples if m and c and pl]
    datasets = set()
    for model_name, config, pl in triples:
        res = model_results_dir(task_type, model_name)
        if not os.path.isdir(res):
            continue
        for dataset in os.listdir(res):
            yr = year_range_dir(task_type, model_name, dataset, config, year_range)
            if not os.path.isdir(yr):
                continue
            for fname in os.listdir(yr):
                p = parse_metrics_file(fname)
                if p and str(p["pl"]) == str(pl):
                    datasets.add(dataset)
    return sorted(datasets)


# ── reset outputs (no panel toggle) ─────────────────────────────────────────

def _reset_results():
    """Return values to clear table + plot controls without touching the panel."""
    return (
        _empty_state_html(),  # metrics_table
        gr.update(choices=[], value=None, interactive=False),  # metric_selector
        gr.update(value=None),  # plot
        gr.update(choices=[], value=None, interactive=False),  # dataset_radio
    )


# ── event handlers ───────────────────────────────────────────────────────────

def on_metrics_task_change(task_type):
    year_ranges = list_year_ranges(task_type)
    return (
        gr.update(choices=year_ranges, value=None, interactive=bool(year_ranges)),  # year_range
        gr.update(choices=[], value=[], interactive=False),  # model
        gr.update(choices=[], value=[], interactive=False),  # config_pl
        gr.update(choices=[], value=[], interactive=False),  # datasets
        *_reset_results(),
        _hide_metrics_download(),
    )


def on_metrics_year_range_change(task_type, year_range):
    models = list_models_for_range(task_type, year_range)
    return (
        gr.update(choices=models, value=[], interactive=bool(models)),  # model
        gr.update(choices=[], value=[], interactive=False),  # config_pl
        gr.update(choices=[], value=[], interactive=False),  # datasets
        *_reset_results(),
        _hide_metrics_download(),
    )


def on_metrics_model_change(task_type, year_range, model_names):
    opts = list_config_pl_options(task_type, year_range, model_names)
    return (
        gr.update(choices=opts, value=[], interactive=bool(opts)),  # config_pl
        gr.update(choices=[], value=[], interactive=False),  # datasets
        *_reset_results(),
        _hide_metrics_download(),
    )


def on_metrics_config_pl_change(task_type, year_range, model_names, config_pl_options):
    datasets = list_datasets_for_config_pl_options(task_type, year_range, config_pl_options)
    return (
        gr.update(choices=datasets, value=[], interactive=bool(datasets)),  # datasets
        *_reset_results(),
        _hide_metrics_download(),
    )


def on_metrics_datasets_change(
        task_type, year_range,
        config_pl_options, datasets,
        current_metric, current_plot_dataset,
):
    if not datasets:
        return (*_reset_results(), _hide_metrics_download())

    df, metric_cols = build_metrics_df(task_type, year_range, config_pl_options, datasets)

    if df is None or df.empty or not metric_cols:
        return (
            _empty_state_html("No metrics found for the current selection."),
            gr.update(choices=[], value=None, interactive=False),
            gr.update(value=None),
            gr.update(choices=[], value=None, interactive=False),
            _hide_metrics_download(),
        )

    valid_ds = [d for d in datasets if d in set(df["dataset"])]
    if not valid_ds:
        return (
            _empty_state_html("No valid datasets found."),
            gr.update(choices=[], value=None, interactive=False),
            gr.update(value=None),
            gr.update(choices=[], value=None, interactive=False),
            _hide_metrics_download(),
        )

    selected_metric = current_metric if current_metric in metric_cols else metric_cols[0]
    plot_dataset = current_plot_dataset if current_plot_dataset in valid_ds else valid_ds[0]
    row_keys = ["model", "config", "sl", "pl"]

    table_html = build_metrics_html_table(df, valid_ds, metric_cols, row_keys)
    fig = build_metrics_plot(df, valid_ds, row_keys, selected_metric, plot_dataset)

    export_path = build_metrics_export_csv(
        df=df,
        datasets=valid_ds,
        metric_cols=metric_cols,
        row_keys=row_keys,
        task_type=task_type,
        year_range=year_range,
    )

    return (
        table_html,
        gr.update(choices=metric_cols, value=selected_metric, interactive=True),
        gr.update(value=fig),
        gr.update(choices=valid_ds, value=plot_dataset, interactive=True),
        gr.update(value=export_path, visible=True),
    )


def update_metrics_plot(
        task_type, year_range, model_names,
        config_pl_options, datasets,
        selected_metric, plot_dataset,
):
    if not all([task_type, year_range, model_names, config_pl_options,
                datasets, selected_metric]):
        return gr.update(value=None)

    df, metric_cols = build_metrics_df(task_type, year_range, config_pl_options, datasets)
    if df is None or selected_metric not in metric_cols:
        return gr.update(value=None)

    valid_ds = [d for d in datasets if d in set(df["dataset"])]
    if not valid_ds:
        return gr.update(value=None)

    if not plot_dataset or plot_dataset not in valid_ds:
        plot_dataset = valid_ds[0]

    fig = build_metrics_plot(df, valid_ds, ["model", "config", "sl", "pl"],
                             selected_metric, plot_dataset)
    return gr.update(value=fig)


# ── data loading ─────────────────────────────────────────────────────────────

def load_metrics_json(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
        return payload["metrics"], payload
    if isinstance(payload, dict):
        return payload, payload
    return {}, payload


def flatten_metrics(metrics: dict) -> dict:
    flat = {}

    def walk(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
        else:
            flat[prefix] = obj

    walk("", metrics)
    return flat


METRIC_CANDIDATES = [
    "MSE", "MAE", "RMSE", "R2",
    "Accuracy", "F1", "Precision", "Recall", "MCC",
    "IC", "RankIC", "ICIR", "RankICIR",
]


def build_metrics_df(task_type, year_range, config_pl_options, datasets):
    triples = [parse_config_pl_option(o) for o in config_pl_options]
    triples = [(m, c, pl) for m, c, pl in triples if m and c and pl]

    rows = []
    for dataset in datasets:
        for model_name, config, pl_value in triples:
            yr = year_range_dir(task_type, model_name, dataset, config, year_range)
            if not os.path.isdir(yr):
                continue
            for fname in os.listdir(yr):
                p = parse_metrics_file(fname)
                if not p or str(p["pl"]) != str(pl_value):
                    continue
                try:
                    metrics, _ = load_metrics_json(os.path.join(yr, fname))
                    flat = flatten_metrics(metrics)
                    rows.append({
                        "model": model_name, "config": config,
                        "dataset": dataset,
                        "sl": int(p["sl"]), "pl": int(p["pl"]),
                        **{m: flat.get(m) for m in METRIC_CANDIDATES},
                    })
                except Exception as e:
                    rows.append({
                        "model": model_name, "config": config,
                        "dataset": dataset, "sl": None, "pl": pl_value,
                        "error": str(e),
                    })

    if not rows:
        return None, None

    df = pd.DataFrame(rows)
    metric_cols = [c for c in METRIC_CANDIDATES
                   if c in df.columns and df[c].notna().any()]
    df = df.sort_values(["model", "config", "pl", "sl", "dataset"])
    return df, metric_cols


# ── plot builder ─────────────────────────────────────────────────────────────

def build_metrics_plot(df, datasets, row_keys, selected_metric, plot_dataset=None):
    if df is None or df.empty or selected_metric is None:
        return go.Figure()

    available = [d for d in datasets if d in set(df["dataset"])]
    if not available:
        return go.Figure()

    if plot_dataset is None or plot_dataset not in available:
        plot_dataset = available[0]

    df_ds = df[df["dataset"] == plot_dataset].copy()
    if df_ds.empty:
        return go.Figure()

    model_configs = sorted({(r["model"], r["config"])
                            for _, r in df_ds[["model", "config"]]
                           .drop_duplicates().iterrows()})
    palette = get_dataset_colors(plot_dataset, len(model_configs))
    color_map = {mc: palette[i] for i, mc in enumerate(model_configs)}

    fig = go.Figure()
    for (model, config), group in df_ds.groupby(["model", "config"], dropna=False):
        grp = group.sort_values("pl")
        x_vals = grp["pl"].tolist()
        y_vals = [float(v) if v is not None and pd.notna(v) else None
                  for v in grp[selected_metric].tolist()]

        if not x_vals or all(v is None for v in y_vals):
            continue

        color = color_map[(model, config)]
        label = f"{model} · {config}"
        fig.add_trace(go.Scatter(
            name=label,
            legendgroup=f"{model}_{config}",
            x=x_vals, y=y_vals,
            mode="lines+markers",
            connectgaps=True,
            line=dict(color=color, width=2.5),
            marker=dict(size=9, color=color, symbol="circle",
                        line=dict(width=2, color="white")),
            hovertemplate=(
                f"<b>{label}</b><br>"
                "pl = <b>%{x}</b><br>"
                f"{selected_metric} = <b>%{{y:.4f}}</b><extra></extra>"
            ),
        ))

    pl_ticks = sorted(df_ds["pl"].dropna().unique().astype(int).tolist())
    y_all = df_ds[selected_metric].dropna().tolist()
    if y_all:
        y_min, y_max = min(y_all), max(y_all)
        pad = (y_max - y_min) * 0.15 if y_max != y_min else 0.1
        y_range = [y_min - pad, y_max + pad]
    else:
        y_range = None

    fig.update_layout(
        title=dict(
            text=f"<b>{plot_dataset.upper()}</b>  ·  {selected_metric} vs Prediction Length",
            font=dict(size=14, family="DM Sans, system-ui, sans-serif", color="#0f172a"),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title=dict(text="Prediction Length", font=dict(size=11, color="#64748b")),
            tickmode="array", tickvals=pl_ticks,
            ticktext=[str(v) for v in pl_ticks],
            tickfont=dict(size=11, color="#334155"),
            gridcolor="#eef2ff", zeroline=False,
            showline=True, linecolor="#e2e8f0",
        ),
        yaxis=dict(
            title=dict(text=selected_metric, font=dict(size=11, color="#64748b")),
            tickfont=dict(size=11, color="#334155"),
            gridcolor="#f1f5f9", zeroline=False,
            showline=True, linecolor="#c7d2fe",
            range=y_range,
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.04,
            xanchor="center", x=0.5,
            font=dict(size=10, color="#334155"),
            bgcolor="rgba(255,255,255,.9)",
            bordercolor="#e2e8f0", borderwidth=1,
        ),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="white",
        margin=dict(t=90, b=60, l=70, r=30),
        hovermode="x unified",
        font=dict(family="DM Mono, Fira Mono, monospace"),
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


# ── HTML table builder ───────────────────────────────────────────────────────

def fmt_metric(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def build_metrics_html_table(df, datasets, metric_cols, row_keys):
    datasets = [d for d in datasets if d in set(df["dataset"])]

    html = """
    <div class="metrics-outer"><div class="metrics-scroll">
    <table class="mpt"><thead><tr>
        <th rowspan="2">Model</th>
        <th rowspan="2">Config</th>
        <th rowspan="2">SL</th>
        <th rowspan="2">PL</th>
    """
    for ds in datasets:
        html += f'<th colspan="{len(metric_cols)}">{ds.upper()}</th>'
    html += "</tr><tr>"
    for _ in datasets:
        for m in metric_cols:
            html += f'<th class="msub">{m}</th>'
    html += "</tr></thead><tbody>"

    model_rowspans = (
        df[row_keys].drop_duplicates()
        .groupby("model").size().to_dict()
    )
    printed_models: set = set()

    for (model_name, config, sl, pl), group in (
            df.sort_values(row_keys)
                    .groupby(row_keys, dropna=False, sort=False)
    ):
        html += "<tr>"
        if model_name not in printed_models:
            rs = model_rowspans.get(model_name, 1)
            html += f'<td rowspan="{rs}" class="model-cell">{model_name}</td>'
            printed_models.add(model_name)
        html += f"<td class='config-cell'>{config}</td><td>{sl}</td><td>{pl}</td>"
        for ds in datasets:
            ds_row = group[group["dataset"] == ds]
            if ds_row.empty:
                html += f'<td class="missing" colspan="{len(metric_cols)}">—</td>'
            else:
                r = ds_row.iloc[0]
                for m in metric_cols:
                    html += f"<td>{fmt_metric(r.get(m))}</td>"
        html += "</tr>"

    html += "</tbody></table></div></div>"
    return html
