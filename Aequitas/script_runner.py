"""
Run a model's train.py / test.py as a subprocess and stream its output.

Every stdout line is forwarded to `log_fn` (the UI adds the timestamp), except
the JSON result, which the script prints between the markers defined in
utils/logger.py and which is returned to the caller.
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime

from utils.logger import RESULT_START, RESULT_END

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# Lines kept for the error report when the script fails.
TAIL_LINES = 80

SUBPROCESS_ENV = {
    "PYTHONPATH": PROJECT_ROOT,
    "PYTHONUNBUFFERED": "1",
    # Progress bars redraw with "\r" and flood the log box: disable them.
    "TQDM_DISABLE": "1",
    # Silence TensorFlow C++ info/warning messages.
    "TF_CPP_MIN_LOG_LEVEL": "2",
}


def run_model_script(module: str, args: list[str], log_fn) -> dict:
    """Run `python -m <module> <args>` from the project root and return its JSON result."""
    cmd = [sys.executable, "-u", "-m", module, *args]
    log_fn(f"Command: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, **SUBPROCESS_ENV},
    )

    output_lines = []
    result_lines = []
    inside_result = False

    assert process.stdout is not None

    for line in iter(process.stdout.readline, ""):
        line = line.rstrip("\n")
        stripped = line.strip()

        if stripped == RESULT_START:
            inside_result = True
            continue

        if stripped == RESULT_END:
            inside_result = False
            continue

        if inside_result:
            result_lines.append(line)
            continue

        if not stripped:
            continue

        output_lines.append(line)
        log_fn(line)

    process.stdout.close()
    return_code = process.wait()

    tail = "\n".join(output_lines[-TAIL_LINES:])

    if return_code != 0:
        raise RuntimeError(f"The process failed with exit code {return_code}.\n\n{tail}")

    if not result_lines:
        raise RuntimeError(f"No JSON result found in the output of {module}.\n\n{tail}")

    json_str = "\n".join(result_lines)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON result from {module}: {e}\n\n{json_str}")


def _timestamped(msg: str) -> str:
    return f"[{datetime.now():%H:%M:%S}] {msg}"


def stream_job_logs(job, action: str):
    """
    Run `job(log_fn)` in a background thread and stream its progress.

    `action` names the job in the status lines ("Training", "Test").
    Yields `(logs_text, result_text, running)` tuples until the job ends;
    on success the result is the job's return value as JSON, on failure
    the traceback.
    """
    logs = []
    log_queue = queue.Queue()
    done_queue = queue.Queue()

    def push_log(msg):
        log_queue.put(_timestamped(msg))

    def worker():
        try:
            push_log(f"{action} started.")
            done_queue.put(("ok", job(push_log)))
        except Exception:
            push_log(f"{action} failed.")
            done_queue.put(("error", traceback.format_exc()))

    threading.Thread(target=worker, daemon=True).start()

    while True:
        while not log_queue.empty():
            logs.append(log_queue.get())

        if not done_queue.empty() and log_queue.empty():
            status, payload = done_queue.get()

            if status == "ok":
                logs.append(_timestamped(f"{action} completed successfully."))
                result_text = json.dumps(payload, indent=2, ensure_ascii=False)
            else:
                result_text = payload

            yield "\n".join(logs), result_text, False
            return

        yield "\n".join(logs), "", True
        time.sleep(0.1)
