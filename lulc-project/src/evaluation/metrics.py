"""Standard classification metrics reported in the results tables."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    cohen_kappa_score, confusion_matrix, roc_auc_score,
)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray | None = None) -> dict:
    """
    Args:
        y_true: (N,) integer class labels
        y_pred: (N,) integer predicted labels
        y_probs: optional (N, K) predicted probabilities, needed for ROC-AUC
    Returns:
        dict of metric name -> value
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }
    if y_probs is not None:
        try:
            num_classes = y_probs.shape[-1]
            if num_classes == 2:
                # sklearn's multi_class="ovr" path requires >2 classes; for
                # binary classification, roc_auc_score wants the positive
                # class's probability as a 1D array.
                metrics["roc_auc_ovr"] = roc_auc_score(y_true, y_probs[:, 1])
            else:
                metrics["roc_auc_ovr"] = roc_auc_score(y_true, y_probs, multi_class="ovr")
        except ValueError:
            # e.g. a class missing from y_true in a tiny fold; skip rather than crash a run
            metrics["roc_auc_ovr"] = float("nan")
    return metrics


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
