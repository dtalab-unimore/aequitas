"""
Uniform logging for the train.py / test.py entry points.

The Aequitas UI runs these scripts as subprocesses (see Aequitas/script_runner.py):
every stdout line is shown in the UI log box, which adds the timestamp, and the
final result is read as JSON between RESULT_START and RESULT_END.
Messages should therefore be plain English text, without timestamps.
"""
import json

RESULT_START = "__RESULT_JSON_START__"
RESULT_END = "__RESULT_JSON_END__"


def _log(log_fn, msg):
    """Send `msg` to `log_fn` when given (in-process use), otherwise print it to stdout."""
    if log_fn is not None:
        log_fn(msg)
    else:
        print(msg, flush=True)


def emit_result(output: dict):
    """Print the script result as JSON, wrapped in the markers parsed by the UI."""
    print(RESULT_START, flush=True)
    print(json.dumps(output, indent=4, ensure_ascii=False, default=str), flush=True)
    print(RESULT_END, flush=True)
