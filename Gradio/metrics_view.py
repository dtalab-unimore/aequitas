import os
import json
import re
import plotly.graph_objects as go
import gradio as gr
import pandas as pd


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

ALL_TASKS = ["Regression", "Classification", "Ranking"]
SEED = "42"


DATASET_PALETTES = {
    "nasdaq100":  ["#c0392b", "#e74c3c", "#f1948a", "#fadbd8"],
    "sp500":      ["#1a5276", "#2980b9", "#7fb3d3", "#aed6f1"],
    "dji":        ["#1e8449", "#27ae60", "#82e0aa", "#a9dfbf"],
    "sx5e":       ["#6c3483", "#8e44ad", "#c39bd3", "#d7bde2"],
    "sxxp":       ["#d35400", "#e67e22", "#f0b27a", "#fad7a0"],
    "_default":   ["#2c3e50", "#566573", "#99a3a4", "#ccd1d1"],
}

def hex_to_rgba(hex_color: str, alpha: float = 0.07) -> str:
    """Converte un colore hex in rgba con alpha specificato."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def get_dataset_colors(dataset: str, n: int) -> list:
    """Restituisce n colori dalla palette del dataset, ciclando se necessario."""
    palette = DATASET_PALETTES.get(dataset.lower(), DATASET_PALETTES["_default"])

    if n <= len(palette):
        return palette[:n]

    colors = []
    for i in range(n):
        colors.append(palette[i % len(palette)])
    return colors


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

    fig = go.Figure()

    model_configs = sorted(set(
        (row["model"], row["config"])
        for _, row in df_ds[["model", "config"]].drop_duplicates().iterrows()
    ))
    palette = get_dataset_colors(plot_dataset, len(model_configs))
    color_map = {mc: palette[i] for i, mc in enumerate(model_configs)}

    for (model, config), group in df_ds.groupby(["model", "config"], dropna=False):
        group_sorted = group.sort_values("pl")
        x_vals = group_sorted["pl"].tolist()
        y_vals = [float(v) if v is not None and pd.notna(v) else None
                  for v in group_sorted[selected_metric].tolist()]

        if not x_vals or all(v is None for v in y_vals):
            continue

        color = color_map[(model, config)]
        label = f"{model} · {config}"

        fig.add_trace(go.Scatter(
            name=label,
            legendgroup=f"{model}_{config}",
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            connectgaps=True,
            showlegend=True,
            line=dict(color=color, width=2.5),
            marker=dict(
                size=10,
                color=color,
                symbol="circle",
                line=dict(width=2, color="white"),
            ),
            hovertemplate=(
                f"<b>{label}</b><br>"
                "pl = <b>%{x}</b><br>"
                f"{selected_metric} = <b>%{{y:.4f}}</b>"
                "<extra></extra>"
            ),
        ))

    pl_ticks = sorted(df_ds["pl"].dropna().unique().astype(int).tolist())

    y_vals_all = df_ds[selected_metric].dropna().tolist()
    if y_vals_all:
        y_min, y_max = min(y_vals_all), max(y_vals_all)
        y_pad = (y_max - y_min) * 0.15 if y_max != y_min else 0.1
        y_range = [y_min - y_pad, y_max + y_pad]
    else:
        y_range = None

    fig.update_layout(
        title=dict(
            text=f"<b>{plot_dataset.upper()}</b>  ·  {selected_metric} vs Prediction Length",
            font=dict(size=15, family="Inter, system-ui, sans-serif", color="#0f172a"),
            x=0.5,
            xanchor="center",
        ),

        xaxis=dict(
            title=dict(text="Prediction Length (pl)", font=dict(size=12, color="#64748b")),
            tickmode="array",
            tickvals=pl_ticks,
            ticktext=[str(v) for v in pl_ticks],
            tickfont=dict(size=12, color="#334155"),
            gridcolor="#eef2ff",
            zeroline=False,
            showline=True,
            linecolor="#e2e8f0",
        ),
        yaxis=dict(
            title=dict(text=selected_metric, font=dict(size=12, color="#64748b")),
            tickfont=dict(size=12, color="#334155"),
            gridcolor="#f1f5f9",
            zeroline=False,
            showline=True,
            linecolor="#c7d2fe",
            range=y_range,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#334155"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#e2e8f0",
            borderwidth=1
        ),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="white",
        margin=dict(t=95, b=65, l=75, r=35),
        hovermode="x unified",
        font=dict(family="Inter, system-ui, sans-serif"),

    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


def task_dir(task_type: str) -> str:
    return os.path.join(PROJECT_ROOT, task_type)


def model_results_dir(task_type: str, model_name: str) -> str:
    return os.path.join(PROJECT_ROOT, task_type, model_name, "results")


def dataset_dir(task_type: str, model_name: str, dataset: str) -> str:
    return os.path.join(model_results_dir(task_type, model_name), dataset)


def config_dir(task_type: str, model_name: str, dataset: str, config: str) -> str:
    return os.path.join(dataset_dir(task_type, model_name, dataset), config)


def seed_dir(task_type: str, model_name: str, dataset: str, config: str) -> str:
    return os.path.join(config_dir(task_type, model_name, dataset, config), SEED)


def year_range_dir(task_type: str, model_name: str, dataset: str, config: str, year_range: str) -> str:
    return os.path.join(seed_dir(task_type, model_name, dataset, config), year_range)


def parse_metrics_file(file_name: str):
    match = re.match(r"metrics_sl(?P<sl>\d+)_pl(?P<pl>\d+)\.json$", file_name)
    if not match:
        return None
    return match.groupdict()

def parse_config_pl_option(option: str):
    """
    Example:
      MASTER | MASTER | pl10
      model  | config | pl
    """
    if not option:
        return None, None, None

    parts = [p.strip() for p in option.split("|")]

    if len(parts) != 3:
        return None, None, None

    model_name = parts[0]
    config = parts[1]
    pl_part = parts[2]

    if not pl_part.startswith("pl"):
        return None, None, None

    pl_value = pl_part.replace("pl", "").strip()

    return model_name, config, pl_value


def is_year_range_folder(name: str) -> bool:
    return re.match(r"^y\d{4}_y\d{4}$", name) is not None


def unique_sorted(values, numeric=False):
    values = [v for v in values if v is not None and v != ""]
    if numeric:
        return sorted(set(values), key=lambda x: int(x))
    return sorted(set(values))


def list_year_ranges(task_type: str):
    if not task_type:
        return []

    base = task_dir(task_type)
    if not os.path.isdir(base):
        return []

    ranges = set()

    for model_name in os.listdir(base):
        results_dir = model_results_dir(task_type, model_name)
        if not os.path.isdir(results_dir):
            continue

        for dataset in os.listdir(results_dir):
            dataset_path = os.path.join(results_dir, dataset)
            if not os.path.isdir(dataset_path):
                continue

            for config in os.listdir(dataset_path):
                seed_path = seed_dir(task_type, model_name, dataset, config)
                if not os.path.isdir(seed_path):
                    continue

                for folder in os.listdir(seed_path):
                    yr_path = os.path.join(seed_path, folder)
                    if os.path.isdir(yr_path) and is_year_range_folder(folder):
                        ranges.add(folder)

    return sorted(ranges)


def list_models_for_range(task_type: str, year_range: str):
    if not task_type or not year_range:
        return []

    base = task_dir(task_type)
    if not os.path.isdir(base):
        return []

    models = set()

    for model_name in os.listdir(base):
        results_dir = model_results_dir(task_type, model_name)
        if not os.path.isdir(results_dir):
            continue

        found = False

        for dataset in os.listdir(results_dir):
            dataset_path = os.path.join(results_dir, dataset)
            if not os.path.isdir(dataset_path):
                continue

            for config in os.listdir(dataset_path):
                yr_path = year_range_dir(task_type, model_name, dataset, config, year_range)
                if os.path.isdir(yr_path):
                    models.add(model_name)
                    found = True
                    break

            if found:
                break

    return sorted(models)


def list_config_pl_options(task_type: str, year_range: str, model_names: list):
    if not task_type or not year_range or not model_names:
        return []

    options = set()

    for model_name in model_names:
        results_dir = model_results_dir(task_type, model_name)

        if not os.path.isdir(results_dir):
            continue

        for dataset in os.listdir(results_dir):
            dataset_path = os.path.join(results_dir, dataset)

            if not os.path.isdir(dataset_path):
                continue

            for config in os.listdir(dataset_path):
                yr_base = year_range_dir(task_type, model_name, dataset, config, year_range)

                if not os.path.isdir(yr_base):
                    continue

                for file_name in os.listdir(yr_base):
                    parsed = parse_metrics_file(file_name)

                    if parsed:
                        options.add(f"{model_name} | {config} | pl{parsed['pl']}")

    return sorted(options)

def list_configs_for_range_model(task_type: str, year_range: str, model_name: str):
    if not task_type or not year_range or not model_name:
        return []

    results_dir = model_results_dir(task_type, model_name)
    if not os.path.isdir(results_dir):
        return []

    configs = set()

    for dataset in os.listdir(results_dir):
        dataset_path = os.path.join(results_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue

        for config in os.listdir(dataset_path):
            yr_path = year_range_dir(task_type, model_name, dataset, config, year_range)
            if os.path.isdir(yr_path):
                configs.add(config)

    return sorted(configs)


def list_datasets_for_range_model_config(task_type: str, year_range: str, model_name: str, configs):
    if not task_type or not year_range or not model_name or not configs:
        return []

    results_dir = model_results_dir(task_type, model_name)
    if not os.path.isdir(results_dir):
        return []

    datasets = set()

    for dataset in os.listdir(results_dir):
        for config in configs:
            yr_path = year_range_dir(task_type, model_name, dataset, config, year_range)
            if os.path.isdir(yr_path):
                datasets.add(dataset)

    return sorted(datasets)


def list_pl_for_selection(task_type: str, year_range: str, model_name: str, configs, datasets):
    if not task_type or not year_range or not model_name or not configs or not datasets:
        return []

    pls = set()

    for dataset in datasets:
        for config in configs:
            yr_path = year_range_dir(task_type, model_name, dataset, config, year_range)

            if not os.path.isdir(yr_path):
                continue

            for file_name in os.listdir(yr_path):
                parsed = parse_metrics_file(file_name)
                if parsed:
                    pls.add(parsed["pl"])

    return unique_sorted(pls, numeric=True)

def list_datasets_for_config_pl_options(task_type, year_range, config_pl_options):
    if not task_type or not year_range or not config_pl_options:
        return []

    selected_triples = [
        parse_config_pl_option(option)
        for option in config_pl_options
    ]
    selected_triples = [
        t for t in selected_triples
        if t[0] is not None and t[1] is not None and t[2] is not None
    ]

    datasets = set()

    for model_name, config, pl in selected_triples:
        results_dir = model_results_dir(task_type, model_name)

        if not os.path.isdir(results_dir):
            continue

        for dataset in os.listdir(results_dir):
            yr_path = year_range_dir(task_type, model_name, dataset, config, year_range)

            if not os.path.isdir(yr_path):
                continue

            for file_name in os.listdir(yr_path):
                parsed = parse_metrics_file(file_name)

                if parsed and str(parsed["pl"]) == str(pl):
                    datasets.add(dataset)

    return sorted(datasets)


def on_metrics_task_change(task_type):
    year_ranges = list_year_ranges(task_type)

    return (
        gr.update(choices=year_ranges, value=None, interactive=bool(year_ranges)),
        gr.update(choices=[], value=[], interactive=False),
        gr.update(choices=[], value=[], interactive=False),
        gr.update(choices=[], value=[], interactive=False),
        "",
        gr.update(visible=False),
        gr.update(choices=[], value=None, interactive=False, visible=False),
        gr.update(value=None, visible=False),
        gr.update(choices=[], value=None, interactive=False, visible=False),
    )


def on_metrics_year_range_change(task_type, year_range):
    models = list_models_for_range(task_type, year_range)

    return (
        gr.update(choices=models, value=[], interactive=bool(models)),
        gr.update(choices=[], value=[], interactive=False),
        gr.update(choices=[], value=[], interactive=False),
        "",
        gr.update(visible=False),
        gr.update(choices=[], value=None, interactive=False, visible=False),
        gr.update(value=None, visible=False),
        gr.update(choices=[], value=None, interactive=False, visible=False)
    )


def on_metrics_model_change(task_type, year_range, model_names):
    config_pl_options = list_config_pl_options(task_type, year_range, model_names)

    return (
        gr.update(choices=config_pl_options, value=[], interactive=bool(config_pl_options)),
        gr.update(choices=[], value=[], interactive=False),
        "",
        gr.update(visible=False),
        gr.update(choices=[], value=None, interactive=False, visible=False),
        gr.update(value=None, visible=False),
        gr.update(choices=[], value=None, interactive=False, visible=False),

    )


def on_metrics_config_pl_change(task_type, year_range, model_names, config_pl_options):
    datasets = list_datasets_for_config_pl_options(task_type, year_range, config_pl_options)

    return (
        gr.update(choices=datasets, value=[], interactive=bool(datasets)),
        "",
        gr.update(visible=False),
        gr.update(choices=[], value=None, interactive=False, visible=False),
        gr.update(value=None, visible=False),
        gr.update(choices=[], value=None, interactive=False, visible=False),
    )

def on_metrics_datasets_change(
    task_type,
    year_range,
    config_pl_options,
    datasets,
    current_metric,
    current_plot_dataset,
):

    df, metric_cols = build_metrics_df(
        task_type=task_type,
        year_range=year_range,
        config_pl_options=config_pl_options,
        datasets=datasets,
    )

    valid_datasets = [d for d in datasets if d in set(df["dataset"])]

    selected_metric = current_metric if current_metric in metric_cols else metric_cols[0]
    plot_dataset = (
        current_plot_dataset
        if current_plot_dataset in valid_datasets
        else valid_datasets[0]
    )

    row_keys = ["model", "config", "sl", "pl"]

    table_html = build_metrics_html_table(
        df=df,
        datasets=valid_datasets,
        metric_cols=metric_cols,
        row_keys=row_keys,
    )

    fig = build_metrics_plot(
        df=df,
        datasets=valid_datasets,
        row_keys=row_keys,
        selected_metric=selected_metric,
        plot_dataset=plot_dataset,
    )

    return (
        table_html,
        gr.update(visible=True),
        gr.update(
            choices=metric_cols,
            value=selected_metric,
            interactive=True,
            visible=True,
        ),
        gr.update(value=fig, visible=True),
        gr.update(
            choices=valid_datasets,
            value=plot_dataset,
            interactive=True,
            visible=True,
        ),
    )

def load_metrics_json(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
        return payload["metrics"], payload

    if isinstance(payload, dict):
        return payload, payload

    return {}, payload


def flatten_metrics(metrics: dict):
    flat = {}

    def walk(prefix, obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                walk(next_prefix, value)
        else:
            flat[prefix] = obj

    walk("", metrics)
    return flat


def build_metrics_df(task_type, year_range, config_pl_options, datasets):
    rows = []

    selected_triples = []
    for option in config_pl_options:
        model_name, config, pl_value = parse_config_pl_option(option)
        if model_name and config and pl_value:
            selected_triples.append((model_name, config, pl_value))

    for dataset in datasets:
        for model_name, config, pl_value in selected_triples:
            yr_path = year_range_dir(task_type, model_name, dataset, config, year_range)

            if not os.path.isdir(yr_path):
                continue
            for file_name in os.listdir(yr_path):
                parsed = parse_metrics_file(file_name)
                if not parsed or str(parsed["pl"]) != str(pl_value):
                    continue
                path = os.path.join(yr_path, file_name)
                try:
                    metrics, _ = load_metrics_json(path)
                    flat = flatten_metrics(metrics)
                    rows.append({
                        "model": model_name, "config": config,
                        "dataset": dataset,
                        "sl": int(parsed["sl"]), "pl": int(parsed["pl"]),
                        "MSE": flat.get("MSE"), "MAE": flat.get("MAE"),
                        "RMSE": flat.get("RMSE"), "R2": flat.get("R2"),
                        "Accuracy": flat.get("Accuracy"), "F1": flat.get("F1"),
                        "Precision": flat.get("Precision"), "Recall": flat.get("Recall"),
                        "IC": flat.get("IC"), "RankIC": flat.get("RankIC"),
                        "ICIR": flat.get("ICIR"), "RankICIR": flat.get("RankICIR"),
                    })
                except Exception as e:
                    rows.append({"model": model_name, "config": config,
                                 "dataset": dataset, "sl": None, "pl": pl_value, "error": str(e)})

    if not rows:
        return None, None

    df = pd.DataFrame(rows)
    metric_candidates = ["MSE","MAE","RMSE","R2","Accuracy","F1","Precision",
                         "Recall","MCC","IC","RankIC","ICIR","RankICIR"]
    metric_cols = [c for c in metric_candidates if c in df.columns and df[c].notna().any()]
    df = df.sort_values(["model", "config", "pl", "sl", "dataset"])
    return df, metric_cols

def update_metrics_plot(
    task_type,
    year_range,
    model_names,
    config_pl_options,
    datasets,
    selected_metric,
    plot_dataset,
):
    if not task_type or not year_range or not model_names or not config_pl_options or not datasets or not selected_metric:
        return gr.update(value=None, visible=True)

    df, metric_cols = build_metrics_df(
        task_type=task_type,
        year_range=year_range,
        config_pl_options=config_pl_options,
        datasets=datasets,
    )

    if df is None or selected_metric not in metric_cols:
        return gr.update(value=None, visible=True)

    valid_datasets = [d for d in datasets if d in set(df["dataset"])]

    if not valid_datasets:
        return gr.update(value=None, visible=True)

    if not plot_dataset or plot_dataset not in valid_datasets:
        plot_dataset = valid_datasets[0]

    fig = build_metrics_plot(
        df=df,
        datasets=valid_datasets,
        row_keys=["model", "config", "sl", "pl"],
        selected_metric=selected_metric,
        plot_dataset=plot_dataset,
    )

    return gr.update(value=fig, visible=True)



def fmt_metric(value):
    if value is None or pd.isna(value):
        return "-"

    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def build_metrics_html_table(df, datasets, metric_cols, row_keys):
    datasets = [d for d in datasets if d in set(df["dataset"])]

    html = """
    <div class="metrics-outer">
    <div class="metrics-scroll">
    <table class="mpt">
        <thead>
            <tr>
                <th rowspan="2">Model</th>
                <th rowspan="2">Config</th>
                <th rowspan="2">SL</th>
                <th rowspan="2">PL</th>
    """

    for dataset in datasets:
        html += f'<th colspan="{len(metric_cols)}">{dataset.upper()}</th>'

    html += "<tr>"
    for _dataset in datasets:
        for metric in metric_cols:
            html += f'<th class="msub">{metric}</th>'
    html += "</tr></thead><tbody>"

    grouped = df.sort_values(row_keys).groupby(row_keys, dropna=False, sort=False)

    model_rowspans = (
        df[row_keys]
        .drop_duplicates()
        .groupby("model")
        .size()
        .to_dict()
    )

    printed_models = set()

    for (model_name, config, sl, pl), group in grouped:
        html += "<tr>"

        if model_name not in printed_models:
            rowspan = model_rowspans.get(model_name, 1)
            html += f'<td rowspan="{rowspan}" class="model-cell">{model_name}</td>'
            printed_models.add(model_name)

        html += f"<td class='config-cell'>{config}</td>"
        html += f"<td>{sl}</td>"
        html += f"<td>{pl}</td>"

        for dataset in datasets:
            ds_row = group[group["dataset"] == dataset]
            if ds_row.empty:
                for _ in metric_cols:
                    html += "<td class='missing'>—</td>"
            else:
                ds_row = ds_row.iloc[0]
                for metric in metric_cols:
                    html += f"<td>{fmt_metric(ds_row.get(metric))}</td>"

        html += "</tr>"

    html += "</tbody></table></div></div>"
    return html