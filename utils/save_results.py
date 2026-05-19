import json
import pickle

from .load_configuation import _log


def save_results(
        args,
        metrics,
        preds,
        labels,
        output_dates,
        input_dates,
        tickers_to_date,
        metrics_path,
        log_fn=None
):
    metrics = {k: float(v) for k, v in metrics.items()}
    results = {
        "metrics": metrics,
        "preds": preds,
        "labels": labels,
        "pred_date": output_dates,
        "last_date": input_dates,
        "tickers": tickers_to_date
    }

    results_pkl = f"{metrics_path}/results_sl{args.seq_len}_pl{args.pred_len}.pkl"
    metrics_json = f"{metrics_path}/metrics_sl{args.seq_len}_pl{args.pred_len}.json"

    with open(results_pkl, "wb") as f:
        pickle.dump(results, f)

    with open(metrics_json, "w") as f:
        json.dump(metrics, f)

    _log(log_fn, "Test completato con successo.")

    return metrics, results_pkl, metrics_json