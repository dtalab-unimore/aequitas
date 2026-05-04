import subprocess
import json
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

def dispatch_test(
    universe,
    task_type,
    test_model,
    config_name,
    sl_value,
    pl_value,
    start_test_date,
    end_date,
    log_fn,
):
    cmd = [
        sys.executable, "-u", "-m", f"{task_type}.{test_model}.test",
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

    json_start = None

    for i, line in enumerate(output_lines):
        if line.strip().startswith("{"):
            json_start = i
            break

    if json_start is None:
        raise RuntimeError(
            "JSON output not found in the test's stdout.\n\n"
            + "\n".join(output_lines[-80:])
        )

    json_str = "\n".join(output_lines[json_start:])

    return json.loads(json_str)