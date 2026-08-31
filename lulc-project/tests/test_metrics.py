import numpy as np
from src.evaluation.metrics import compute_all_metrics, compute_confusion_matrix


def test_perfect_predictions_give_perfect_scores():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = y_true.copy()
    m = compute_all_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["cohen_kappa"] == 1.0


def test_all_wrong_binary_gives_zero_kappa_or_below():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    m = compute_all_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.0
    assert m["cohen_kappa"] <= 0.0


def test_confusion_matrix_shape_and_diagonal():
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 1, 2, 1])  # last sample misclassified as class 1
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=3)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1
    assert cm[1, 1] == 1
    assert cm[2, 2] == 1
    assert cm[2, 1] == 1  # one true-class-2 sample predicted as class 1


def test_roc_auc_computed_when_probs_given():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    m = compute_all_metrics(y_true, y_pred, y_probs)
    assert "roc_auc_ovr" in m
    assert 0.0 <= m["roc_auc_ovr"] <= 1.0
