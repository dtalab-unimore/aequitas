"""
Run a model's train.py / test.py as a subprocess and stream its output.

Every stdout line is forwarded to `log_fn` (the UI adds the timestamp), except
the JSON result, which the script prints between the markers defined in
utils/logger.py and which is returned to the caller.
"""
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
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


# Finished jobs kept in memory, so a reloaded page can still show their outcome.
FINISHED_JOBS_KEPT = 20


@dataclass
class Job:
    action: str
    meta: dict
    logs: list = field(default_factory=list)
    status: str = "running"  # "running", "ok" or "error"
    result_text: str = ""
    started_at: float = field(default_factory=time.time)


_JOBS: dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()


def start_job(job_fn, action: str, meta: dict | None = None) -> str:
    """
    Run `job_fn(log_fn)` in a background thread and return the job id.

    The job lives in a server-side registry, independent of the browser
    request that started it: closing or reloading the page does not stop it,
    and `follow_job(job_id)` can re-attach to it at any time.
    `action` names the job in the status lines ("Training", "Test");
    `meta` holds whatever the UI needs to restore its view of the run.
    """
    job_id = uuid.uuid4().hex
    job = Job(action=action, meta=meta or {})

    with _JOBS_LOCK:
        _JOBS[job_id] = job
        finished = [jid for jid, j in _JOBS.items() if j.status != "running"]
        for jid in finished[:-FINISHED_JOBS_KEPT]:
            del _JOBS[jid]

    def log_fn(msg):
        job.logs.append(_timestamped(msg))

    def worker():
        log_fn(f"{action} started.")
        try:
            output = job_fn(log_fn)
            job.result_text = json.dumps(output, indent=2, ensure_ascii=False)
            log_fn(f"{action} completed successfully.")
            job.status = "ok"
        except Exception:
            job.result_text = traceback.format_exc()
            log_fn(f"{action} failed.")
            job.status = "error"

    threading.Thread(target=worker, daemon=True).start()
    return job_id


def get_job(job_id) -> Job | None:
    return _JOBS.get(job_id) if job_id else None


def follow_job(job_id):
    """Yield `(logs_text, result_text, running)` for a job until it ends."""
    job = get_job(job_id)
    if job is None:
        return

    while True:
        # Read the status first: the worker sets it after the last log line.
        running = job.status == "running"
        yield "\n".join(job.logs), job.result_text, running

        if not running:
            return

        time.sleep(0.1)


def stream_job_logs(job_fn, action: str):
    """Start a job and follow it (for callers that don't need to re-attach)."""
    yield from follow_job(start_job(job_fn, action))
