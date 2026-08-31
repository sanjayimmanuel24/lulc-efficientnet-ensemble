import numpy as np
import pytest
from src.losses.label_smoothing import smooth_targets, label_smoothing_cross_entropy


def test_smooth_targets_sum_to_one():
    labels = np.array([0, 2, 1])
    y = smooth_targets(labels, num_classes=4, smoothing=0.1)
    assert np.allclose(y.sum(axis=-1), 1.0, atol=1e-6)


def test_smooth_targets_zero_smoothing_is_one_hot():
    labels = np.array([1, 0])
    y = smooth_targets(labels, num_classes=3, smoothing=0.0)
    expected = np.array([[0, 1, 0], [1, 0, 0]], dtype=float)
    assert np.allclose(y, expected)


def test_smooth_targets_true_class_gets_most_mass():
    labels = np.array([2])
    y = smooth_targets(labels, num_classes=5, smoothing=0.2)
    assert y[0, 2] == y.max()
    # off-class value should be exactly smoothing / K
    assert np.isclose(y[0, 0], 0.2 / 5)


def test_smooth_targets_rejects_invalid_smoothing():
    with pytest.raises(ValueError):
        smooth_targets(np.array([0]), num_classes=3, smoothing=1.0)
    with pytest.raises(ValueError):
        smooth_targets(np.array([0]), num_classes=3, smoothing=-0.1)


def test_loss_matches_plain_ce_when_smoothing_zero():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(20, 5))
    labels = rng.integers(0, 5, size=20)

    # plain cross-entropy computed directly
    log_probs = logits - (logits.max(-1, keepdims=True) +
                           np.log(np.exp(logits - logits.max(-1, keepdims=True)).sum(-1, keepdims=True)))
    plain_ce = -np.mean(log_probs[np.arange(20), labels])

    smoothed_ce = label_smoothing_cross_entropy(logits, labels, num_classes=5, smoothing=0.0)
    assert np.isclose(plain_ce, smoothed_ce, atol=1e-6)


def test_loss_increases_with_smoothing_on_confident_correct_logits():
    # If the model is already very confident and correct, adding label
    # smoothing should *increase* the loss (it penalizes over-confidence
    # on purpose), which is the entire point of using it.
    logits = np.array([[10.0, -10.0, -10.0]])
    labels = np.array([0])
    ce = label_smoothing_cross_entropy(logits, labels, num_classes=3, smoothing=0.0)
    ce_smoothed = label_smoothing_cross_entropy(logits, labels, num_classes=3, smoothing=0.1)
    assert ce_smoothed > ce
