"""Fail-closed frame-level metrics for PANDA dataset protocols."""

from typing import Tuple

import numpy as np

_PRIMARY_METRICS = {
    "ucf": "auc",
    "ub": "auc",
    "csad": "auc",
    "xd": "ap",
}


def get_primary_metric(dataset_name: str) -> str:
    """Return the published frame-level primary metric for a dataset."""
    dataset_name = dataset_name.lower()
    if dataset_name not in _PRIMARY_METRICS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Valid options: {sorted(_PRIMARY_METRICS)}"
        )
    return _PRIMARY_METRICS[dataset_name]


def _validated_arrays(y_true, y_pred) -> Tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    if truth.ndim != 1 or prediction.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional arrays")
    if truth.size == 0 or prediction.size == 0:
        raise ValueError("Metric inputs must not be empty")
    if truth.size != prediction.size:
        raise ValueError(
            f"Metric input lengths differ: y_true={truth.size}, y_pred={prediction.size}"
        )
    try:
        truth_finite = np.isfinite(truth).all()
        prediction_finite = np.isfinite(prediction).all()
    except TypeError as exc:
        raise ValueError("Metric inputs must be numeric") from exc
    if not truth_finite or not prediction_finite:
        raise ValueError("Metric inputs must contain only finite values")
    classes = np.unique(truth)
    if not np.isin(classes, [0, 1]).all():
        raise ValueError(f"y_true must be binary; got classes {classes.tolist()}")
    if classes.size != 2:
        raise ValueError(
            f"Metric requires both normal and abnormal frames; got {classes.tolist()}"
        )
    return truth, prediction


def compute_auc(y_true, y_pred) -> float:
    """Compute frame-level ROC-AUC, rejecting invalid/degenerate inputs."""
    from sklearn.metrics import roc_auc_score

    truth, prediction = _validated_arrays(y_true, y_pred)
    return float(roc_auc_score(truth, prediction))


def compute_ap(y_true, y_pred) -> float:
    """Compute frame-level Average Precision, rejecting invalid inputs."""
    from sklearn.metrics import average_precision_score

    truth, prediction = _validated_arrays(y_true, y_pred)
    return float(average_precision_score(truth, prediction))


def compute_primary_metric(dataset_name: str, y_true, y_pred) -> float:
    """Compute a dataset's published frame-level primary metric."""
    metric = get_primary_metric(dataset_name)
    return (
        compute_auc(y_true, y_pred) if metric == "auc" else compute_ap(y_true, y_pred)
    )
