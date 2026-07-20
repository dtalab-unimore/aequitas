import os
import pickle
import numpy as np
from scipy.special import softmax


def convert_predictions_to_rankings(
        model_name: str,
        predictions: list[np.ndarray] | np.ndarray
) -> list[np.ndarray] | np.ndarray:
    """ Convert classification predictions to portfolio ranking scores """

    # Already return ranking scores
    if model_name in ["Adv-ALSTM", 'CNNPred2D']:
        assert isinstance(predictions, np.ndarray)
        return predictions

    if model_name in ["THGNN", 'CNNPred3D']:
        assert isinstance(predictions, list)
        assert all([isinstance(arr, np.ndarray) and arr.ndim == 2 for arr in predictions])

        return predictions

    # Convert logits to buy probabilities
    if model_name in ["DGDNN", "MAN-SF"]:
        assert isinstance(predictions, list)
        assert all([isinstance(arr, np.ndarray) and arr.ndim == 2 for arr in predictions])

        return [softmax(arr, axis=1)[:, [1]] for arr in predictions]

    # Long/short strategy: positive scores for long, negative for short
    if model_name == "HGTAN":
        assert isinstance(predictions, list)
        assert all([isinstance(arr, np.ndarray) and arr.ndim == 2 for arr in predictions])

        results = []

        for logits in predictions:
            probs = softmax(logits, axis=1)
            predicted_class: np.ndarray = np.argmax(probs, axis=1)

            scores = np.full(len(logits), np.nan)
            scores[predicted_class == 1] = probs[predicted_class == 1, 1]  # Long
            scores[predicted_class == 0] = -probs[predicted_class == 0, 0]  # Short

            results.append(scores[:, np.newaxis])

        return results

    raise ValueError(f"Unknown model: {model_name}")


def normalize_classification_labels(labels, predictions):
    """Match classification labels to converted scores, preserving container type."""
    if isinstance(predictions, np.ndarray):
        label_array = np.asarray(labels)
        if label_array.size != predictions.size:
            raise ValueError(
                f"Prediction/label size mismatch: "
                f"{predictions.shape} != {label_array.shape}"
            )
        return label_array.reshape(predictions.shape)

    if not isinstance(predictions, list):
        raise ValueError("Classification predictions must be a list or numpy array")

    if isinstance(labels, np.ndarray):
        labels = list(labels)
    if not isinstance(labels, list):
        raise ValueError("Classification labels must be a list or numpy array")
    if len(labels) != len(predictions):
        raise ValueError(
            f"Prediction/label length mismatch: {len(predictions)} != {len(labels)}"
        )

    normalized = []
    for index, (label, prediction) in enumerate(zip(labels, predictions)):
        label_array = np.asarray(label)
        if label_array.size != prediction.size:
            raise ValueError(
                f"Prediction/label size mismatch at index {index}: "
                f"{prediction.shape} != {label_array.shape}"
            )
        normalized.append(label_array.reshape(prediction.shape))
    return normalized


def align_dgdnn_result_lengths(results):
    """Trim legacy DGDNN output by one sample when its dates are off by one."""
    fields = ["preds", "labels", "tickers", "last_date", "pred_date"]
    lengths = {field: len(results[field]) for field in fields}
    if len(set(lengths.values())) == 1:
        return results

    minimum = min(lengths.values())
    maximum = max(lengths.values())
    if maximum - minimum != 1:
        details = ", ".join(f"{key}: {value}" for key, value in lengths.items())
        raise ValueError(f"Inconsistent DGDNN result lengths: {details}")

    print(
        "  ! Legacy DGDNN off-by-one detected; "
        f"aligning all result collections to {minimum} samples."
    )
    for field in fields:
        results[field] = results[field][:minimum]
    return results


def convert_classification_preds(pred_paths: list[str], model_name: str) -> list[str]:
    converted_preds_paths = []

    for pred_file in pred_paths:
        converted_path = pred_file.replace("/results/", "/results_converted/")
        os.makedirs(os.path.dirname(converted_path), exist_ok=True)

        # Load original predictions
        with open(pred_file, 'rb') as f:
            results = pickle.load(f)

        results['preds'] = convert_predictions_to_rankings(model_name=model_name, predictions=results['preds'])
        results['labels'] = normalize_classification_labels(
            results['labels'], results['preds']
        )
        if model_name == "DGDNN":
            results = align_dgdnn_result_lengths(results)

        # Save converted predictions
        with open(converted_path, 'wb') as f:
            pickle.dump(results, f)

        converted_preds_paths.append(converted_path)
        print(f'  ✓ Converted {pred_file} - {converted_path}')

    print(f'Completed {model_name} ({len(pred_paths)} files)\n')
    return converted_preds_paths


def convert_daily_to_cumulative_returns(preds_paths: list[str], sl: int, pl: int):
    x = pl
    if pl == 2 or pl == 6:
        x = x - pl

    converted_paths = []

    for pred_file in preds_paths:
        converted_path = pred_file.replace("/results/", "/results_converted/")
        os.makedirs(os.path.dirname(converted_path), exist_ok=True)

        # Load original predictions
        with open(pred_file, 'rb') as f:
            results = pickle.load(f)

        cumulative_returns = np.prod(1 + results['preds'][:, :x], axis=1) - 1
        results['preds'] = cumulative_returns[:, np.newaxis]
        cumulative_returns = np.prod(1 + results['labels'][:, :x], axis=1) - 1
        results['labels'] = cumulative_returns[:, np.newaxis]
        results['pred_date'] = [dt[x - 1] for dt in results['pred_date']]

        # Save converted results
        with open(converted_path, 'wb') as f:
            pickle.dump(results, f)

        converted_paths.append(converted_path)
        print(f'  ✓ Converted {pred_file} - {converted_path}')

    print(f'Completed ({len(preds_paths)} files)\n')
    return converted_paths, pl


