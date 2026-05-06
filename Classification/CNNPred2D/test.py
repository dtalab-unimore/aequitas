import json
import os
from argparse import ArgumentParser
from os.path import join

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from Classification.CNNPred2D.load_data import construct_data_warehouse_2d, split_train_val_test
from Classification.CNNPred2D.train import prediction, create_model, get_metrics
from utils.load_configuation import _log, load_configuration
from utils.save_results import save_results

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

def run_test(
        task_type,
        test_model,
        universe,
        config_name,
        sl_value,
        pl_value,
        start_test_date,
        end_date,
        log_fn=None,
):
    _log(log_fn, "Preparazione del test...")

    args, model_save_path, metrics_path = load_configuration(
        task_type, test_model, universe, config_name, sl_value, pl_value, start_test_date, end_date, PROJECT_ROOT, log_fn
    )

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    data_warehouse, n_stocks, n_features = construct_data_warehouse_2d(args)
    data_warehouse, dates = split_train_val_test(data_warehouse, args)

    filepath = join(model_save_path, 'model.weights.h5')
    model = create_model(args, n_features)
    model.load_weights(filepath)

    K.clear_session()
    preds, labels, tickers, last_date, pred_date, preds_prob = prediction(data_warehouse, model, dates, args, idx=(4, 5))

    metrics = get_metrics(preds, labels)

    metrics, results_pkl, metrics_json = save_results(
        args,
        metrics,
        preds,
        np.expand_dims(labels, axis=1),
        pred_date,
        last_date,
        tickers,
        metrics_path
    )

    return {
        "metrics": metrics,
        "results_path": results_pkl,
        "metrics_path": metrics_json,
        "device": str(gpus),
    }

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--task_type", type=str, default='Classification')
    parser.add_argument("--model_name", type=str, default='CNNPred2D')
    parser.add_argument("--universe", type=str, default='sp500')
    parser.add_argument("--config_name", type=str, default='CNNPred2D')
    parser.add_argument("--sl_value", type=int, default=60)
    parser.add_argument("--pl_value", type=int, default=5)
    parser.add_argument("--start_test_date", type=str, default='2023-01-01')
    parser.add_argument("--end_date", type=str, default='2023-12-31')
    args = parser.parse_args()

    out = run_test(
        task_type=args.task_type,
        test_model=args.model_name,
        universe=args.universe,
        config_name=args.config_name,
        sl_value=args.sl_value,
        pl_value=args.pl_value,
        start_test_date=args.start_test_date,
        end_date=args.end_date,
    )
    print(json.dumps(out, indent=4))