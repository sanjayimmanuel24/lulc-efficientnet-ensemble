"""
Label smoothing cross-entropy: the one loss-function change actually kept
in this framework (see docs/BLUEPRINT.md, section "Rejected Alternatives",
for why Focal Loss and ArcFace were evaluated and rejected).

Smoothed target for true class k* out of K classes, smoothing factor eps:

    y_smooth[k] = 1 - eps + eps/K   if k == k*
    y_smooth[k] = eps/K             otherwise

Loss = -sum_k y_smooth[k] * log_softmax(logits)[k]

The NumPy reference here checks the *formula*; the production loss
(src/losses/label_smoothing_torch.py) uses torch.nn.CrossEntropyLoss's
built-in `label_smoothing` argument, which implements the identical
formula (verified against this reference in test_models_torch.py).
"""
from __future__ import annotations
import numpy as np

_EPS = 1e-8


def smooth_targets(labels: np.ndarray, num_classes: int, smoothing: float) -> np.ndarray:
    if not (0.0 <= smoothing < 1.0):
        raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
    n = len(labels)
    y = np.full((n, num_classes), smoothing / num_classes)
    y[np.arange(n), labels] = 1.0 - smoothing + smoothing / num_classes
    return y


def label_smoothing_cross_entropy(logits: np.ndarray, labels: np.ndarray, num_classes: int, smoothing: float) -> float:
    log_probs = logits - _logsumexp(logits)
    targets = smooth_targets(labels, num_classes, smoothing)
    return float(-np.mean(np.sum(targets * log_probs, axis=-1)))


def _logsumexp(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=-1, keepdims=True)
    return m + np.log(np.sum(np.exp(logits - m), axis=-1, keepdims=True))
