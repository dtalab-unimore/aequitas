import json
import os
import subprocess
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

def _normalize_model_name(model_name):
    return str(model_name or "").strip().lower()


def model_accepts_seq_len_arg(model_name) -> bool:
    """
    HIST e DiscoverPLF non hanno un parametro configurabile --seq_len.
    """
    return _normalize_model_name(model_name) not in {"hist", "discoverplf"}


def normalize_seq_len_for_model(model_name, sl_value):
    model_key = _normalize_model_name(model_name)

    if model_key in {"hist", "discoverplf"}:
        return None

    if model_key == "finformer":
        return None

    return sl_value

def dispatch_train(
    universe,
    task_type,
    model_name,
    config_name,
    sl_value,
    pl_value,
    start_train_date,
    end_train_date,
    start_valid_date,
    end_valid_date,
    log_fn,
):
    seq_len_value = normalize_seq_len_for_model(model_name, sl_value)

    cmd = [
        sys.executable, "-u", "-m", f"{task_type}.{model_name}.train",
        "--universe", universe,
        "--model_name", config_name,
        "--pred_len", str(pl_value),
        "--start_date", start_train_date,
        "--end_train_date", end_train_date,
        "--start_valid_date", start_valid_date,
        "--end_valid_date", end_valid_date,
    ]

    if model_accepts_seq_len_arg(model_name) and seq_len_value is not None:
        cmd.extend(["--seq_len", str(seq_len_value)])

    if model_name.lower() == "hist":
        cmd.extend(["--model_name", "HIST"])
        cmd.extend(["--run_name", config_name])

    log_fn(f"Execution: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={
            **os.environ,
            "PYTHONPATH": PROJECT_ROOT,
            "PYTHONUNBUFFERED": "1",
        },
    )

    output_lines = []
    inside_json_output = False

    assert process.stdout is not None

    for line in iter(process.stdout.readline, ""):
        line = line.rstrip("\n")

        if not line:
            continue

        stripped = line.strip()
        output_lines.append(line)

        if stripped.startswith("{"):
            inside_json_output = True
            continue

        if inside_json_output:
            continue

        log_fn(line)

    process.stdout.close()
    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"The process failed with an exit code {return_code}.\n\n"
            + "\n".join(output_lines[-80:])
        )

    stdout_text = "\n".join(output_lines)

    start_marker = "__TRAIN_JSON_START__"
    end_marker = "__TRAIN_JSON_END__"

    start_idx = stdout_text.find(start_marker)
    end_idx = stdout_text.find(end_marker)

    if start_idx == -1 or end_idx == -1:
        raise RuntimeError(
            "JSON output not found in the training stdout.\n\n"
            + "\n".join(output_lines[-80:])
        )

    json_str = stdout_text[start_idx + len(start_marker):end_idx].strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Invalid JSON output from training script: {e}\n\n"
            f"Extracted JSON string:\n{json_str}"
        )
