"""
Post-hoc calibration (core contribution #3).

Temperature scaling fits a single scalar T > 0 on a held-out validation
split, dividing logits by T before the softmax. It does not change argmax
predictions (accuracy is unaffected) but reshapes confidence so it matches
empirical accuracy -- important for a fused, confidence-driven ensemble
where the fusion weights in `fusion.py` are themselves confidence-based:
if per-branch confidences are miscalibrated, the fusion weighting inherits
that bias. Calibration and adaptive fusion are therefore two parts of one
coherent argument, not two unrelated bullet points.

Pure NumPy: fitting T here is a 1-D scalar optimization and does not need
a deep learning framework. In the full pipeline this is fit once after
training, using the frozen model's validation logits.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar

_EPS = 1e-8


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _nll(temperature: float, logits: np.ndarray, labels: np.ndarray) -> float:
    probs = softmax(logits, temperature)
    correct = probs[np.arange(len(labels)), labels]
    return -np.mean(np.log(np.clip(correct, _EPS, 1.0)))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Find T minimizing NLL on a validation set. Returns the fitted scalar."""
    result = minimize_scalar(_nll, bounds=(0.05, 10.0), method="bounded", args=(logits, labels))
    return float(result.x)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE: weighted average gap between confidence and accuracy across bins."""
    confidences = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if not in_bin.any():
            continue
        bin_acc = accuracies[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def adaptive_expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                                          n_bins: int = 15) -> float:
    """
    ECE with EQUAL-MASS bins instead of equal-width bins.

    Why this exists: equal-width ECE is unreliable exactly where this project
    operates. At ~99% accuracy nearly every sample lands in the top confidence
    bin, so the statistic is dominated by one bin and most bins are empty --
    the estimate is biased low and has high variance. Since C1's only surviving
    claim rests on ECE differences of ~0.001, that fragility is load-bearing.

    Equal-mass binning (each bin holds ~N/n_bins samples, split at quantiles of
    the confidence distribution) keeps every bin populated and is the standard
    remedy (Nixon et al. 2019, "Measuring Calibration in Deep Learning").

    Report both: agreement between them is evidence the calibration result is
    real rather than a binning artefact.
    """
    confidences = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    accuracies = (predictions == labels).astype(np.float64)
    n = len(labels)
    if n == 0:
        return 0.0

    # Quantile edges; duplicates collapse when many samples share a confidence.
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(confidences, quantiles))
    if len(edges) < 2:
        return float(abs(accuracies.mean() - confidences.mean()))

    ece = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = (confidences >= lo) & (confidences <= hi) if i == 0 else                  (confidences > lo) & (confidences <= hi)
        if not in_bin.any():
            continue
        ece += (in_bin.sum() / n) * abs(accuracies[in_bin].mean() - confidences[in_bin].mean())
    return float(ece)


def maximum_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """MCE: worst-case (not weighted-average) gap between confidence and accuracy."""
    confidences = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    gaps = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if not in_bin.any():
            continue
        gaps.append(abs(accuracies[in_bin].mean() - confidences[in_bin].mean()))
    return float(max(gaps)) if gaps else 0.0


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multi-class Brier score: mean squared error between probs and one-hot labels."""
    num_classes = probs.shape[-1]
    one_hot = np.eye(num_classes)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))
