import json
import os
import shutil
import tempfile
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from tensorflow.keras import backend as K, callbacks
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential

from .load_data import construct_data_warehouse_3d
from utils.load_configuation import create_saving_path
from utils.logger import emit_result
from utils.save_results import save_results

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))


def get_metrics(preds, labels):
    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    return {
        "F1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "Accuracy": accuracy_score(labels, preds),
        "Precision": precision_score(labels, preds, average="macro", zero_division=0),
        "Recall": recall_score(labels, preds, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(labels, preds),
    }


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        y_true = K.cast(y_true, K.floatx())
        y_pred = K.squeeze(y_pred, axis=-1)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

    def precision(y_true, y_pred):
        y_true = K.cast(y_true, K.floatx())
        y_pred = K.squeeze(y_pred, axis=-1)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        return true_positives / (predicted_positives + K.epsilon())

    precision_pos = precision(y_true, y_pred)
    recall_pos = recall(y_true, y_pred)
    precision_neg = precision(K.ones_like(y_true) - y_true, K.ones_like(y_pred) - K.clip(y_pred, 0, 1))
    recall_neg = recall(K.ones_like(y_true) - y_true, K.ones_like(y_pred) - K.clip(y_pred, 0, 1))
    f_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos + K.epsilon())
    f_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg + K.epsilon())
    return (f_pos + f_neg) / 2


def create_model(args, n_stocks, n_features):
    model = Sequential([
        Conv2D(args.number_filter[0], (1, 1), activation="relu",
               input_shape=(n_stocks, args.seq_len, n_features), data_format="channels_last"),
        Conv2D(args.number_filter[1], (n_stocks, 3), activation="relu"),
        MaxPool2D(pool_size=(1, 2)),
        Conv2D(args.number_filter[2], (1, 3), activation="relu"),
        MaxPool2D(pool_size=(1, 2)),
        Flatten(),
        Dropout(args.dropout),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="Adam", loss="mae", metrics=["acc", f1])
    return model


def checkpoint_path(model_save_path, stock_index):
    checkpoint_dir = os.path.join(model_save_path, "3D-models")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, f"stock_{stock_index}.weights.h5")


def predict_all_stocks(dataset, args, model_save_path, n_stocks, n_features, split_index):
    probabilities = []
    for stock_index in range(n_stocks):
        K.clear_session()
        model = create_model(args, n_stocks, n_features)
        model.load_weights(checkpoint_path(model_save_path, stock_index))
        probabilities.append(model.predict(dataset[split_index], verbose=0).reshape(-1))
    probabilities = np.asarray(probabilities).T
    return (probabilities > 0.5).astype(int), probabilities


def train_models(dataset, args, model_save_path, n_stocks, n_features):
    for stock_index in range(n_stocks):
        K.clear_session()
        model = create_model(args, n_stocks, n_features)
        best_model = callbacks.ModelCheckpoint(
            checkpoint_path(model_save_path, stock_index), monitor="val_f1", verbose=0,
            save_best_only=True, save_weights_only=True, mode="max"
        )
        print(f"Training stock {stock_index + 1}/{n_stocks}", flush=True)
        model.fit(
            dataset[0], dataset[1][:, stock_index], epochs=args.epochs, batch_size=128, verbose=1,
            callbacks=[best_model], validation_data=(dataset[2], dataset[3][:, stock_index])
        )


def _save_split_results(args, dataset, dates, tickers, model_save_path, metrics_path):
    n_stocks = dataset[0].shape[1]
    n_features = dataset[0].shape[-1]
    val_preds, _ = predict_all_stocks(dataset, args, model_save_path, n_stocks, n_features, 2)
    val_metrics = {key: float(value) for key, value in get_metrics(val_preds, dataset[3]).items()}
    with open(os.path.join(metrics_path, f"val_metrics_sl{args.seq_len}_pl{args.pred_len}.json"), "w") as handle:
        json.dump(val_metrics, handle, indent=4)

    test_preds, test_probabilities = predict_all_stocks(dataset, args, model_save_path, n_stocks, n_features, 4)
    predictions = [row.reshape(-1, 1) for row in test_probabilities]
    labels = [row.reshape(-1, 1) for row in np.asarray(dataset[5])]
    return save_results(
        args, get_metrics(test_preds, dataset[5]), predictions, labels,
        dates["pred_date"], dates["last_date"], [tickers] * len(predictions), metrics_path
    )


def run_train(args):
    model_save_path, metrics_path = create_saving_path(args, "Classification", "CNNPred3D", PROJECT_ROOT)
    tmp_root = os.path.join(CURRENT_DIR, "tmp")
    os.makedirs(tmp_root, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="tf_run_", dir=tmp_root)
    os.environ["TMPDIR"] = tmpdir
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.random.set_seed(args.seed)
        dataset, tickers, dates = construct_data_warehouse_3d(args)
        if not tickers or dataset[0].size == 0:
            raise ValueError("No usable stocks or training windows were found for CNNPred3D.")
        args.trained_tickers = list(tickers)
        with open(os.path.join(model_save_path, "config_params.json"), "w") as handle:
            json.dump(vars(args), handle, indent=4)
        n_stocks = dataset[0].shape[1]
        n_features = dataset[0].shape[-1]
        train_models(dataset, args, model_save_path, n_stocks, n_features)
        metrics, results_pkl, metrics_json = _save_split_results(
            args, dataset, dates, tickers, model_save_path, metrics_path
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return {"model_path": model_save_path, "metrics": metrics,
            "results_path": results_pkl, "metrics_path": metrics_json}


def build_parser():
    parser = ArgumentParser(description="Train the CNNPred3D classification model")
    parser.add_argument("--data_path", default="./Evaluation/data")
    parser.add_argument("--universe", default="sp500")
    parser.add_argument("--model_name", default="CNNPred3D")
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--start_date", default="2021-06-01")
    parser.add_argument("--end_train_date", default="2021-12-31")
    parser.add_argument("--start_valid_date", default="2022-06-01")
    parser.add_argument("--end_valid_date", default="2022-12-31")
    parser.add_argument("--start_test_date", default="2023-01-01")
    parser.add_argument("--end_date", default="2023-12-31")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--number_filter", type=int, nargs="+", default=[8, 8, 8])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    output = run_train(build_parser().parse_args())
    emit_result(output)
