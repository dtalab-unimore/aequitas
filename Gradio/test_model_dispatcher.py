import subprocess
import json
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

def dispatch_test(universe, task_type, test_model, config_name, sl_value, pl_value,
                  start_test_date, end_date, log_fn):

    cmd = [
        sys.executable, "-m", f"{task_type}.{test_model}.test",
        "--universe", universe,
        "--config_name", config_name,
        "--sl_value", str(sl_value),
        "--pl_value", str(pl_value),
        "--start_test_date", start_test_date,
        "--end_date", end_date,
        "--task_type", task_type,
        "--model_name", test_model,
    ]

    log_fn(f"Esecuzione: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,       # <- esattamente come lanciarlo dal terminale
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": PROJECT_ROOT}  # forza il path corretto
    )

    stdout, stderr = process.communicate()

    if stderr:
        log_fn(f"STDERR: {stderr}")

    if process.returncode != 0:
        raise RuntimeError(f"Test fallito (exit code {process.returncode}):\n{stderr}")

    lines = [l for l in stdout.strip().splitlines() if l.strip()]

    json_start = next(i for i, l in enumerate(lines) if l.strip().startswith('{'))
    json_str = '\n'.join(lines[json_start:])
    return json.loads(json_str)