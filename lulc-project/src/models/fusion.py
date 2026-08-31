"""
Confidence-Aware Adaptive Fusion (core contribution #1).

Reference (NumPy) implementation of the fusion rule used to combine the
RGB branch and the Spectral-Index branch. This is the formula whose
correctness actually matters for the paper's claims, so it is written and
tested independently of the deep learning framework.

The production model (src/models/ensemble.py) implements the identical
formula directly in torch (softmax/entropy are one-liners in both
frameworks) rather than importing this module, to avoid GPU<->CPU/NumPy
round trips inside the training loop. The two implementations are tested
against each other in tests/test_models_torch.py (requires torch).

Rule
----
For each sample, each branch b in {A, B} produces a softmax probability
vector p_b over K classes. Define the branch's predictive entropy:

    H(p_b) = -sum_k p_b[k] * log(p_b[k])          (nats)

Confidence is the entropy normalized against the maximum possible entropy
for K classes (uniform distribution), so it lies in [0, 1]:

    c_b = 1 - H(p_b) / log(K)

A branch that is completely certain (one-hot output) has c_b = 1.
A branch outputting the uniform distribution has c_b = 0.

Fusion weights are the normalized confidences:

    w_A = c_A / (c_A + c_B + eps)
    w_B = c_B / (c_A + c_B + eps)
    p_fused = w_A * p_A + w_B * p_B

This reduces to plain averaging (the naive ensemble baseline) exactly when
c_A == c_B, which is what makes it a fair, ablatable generalization of the
baseline rather than an unrelated add-on.
"""
from __future__ import annotations
import numpy as np

_EPS = 1e-8


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    """probs: (..., K) softmax probabilities. Returns (...,) entropy in nats."""
    safe = np.clip(probs, _EPS, 1.0)
    return -np.sum(probs * np.log(safe), axis=-1)


def confidence(probs: np.ndarray, num_classes: int) -> np.ndarray:
    """Entropy-based confidence in [0, 1], 1 = certain, 0 = uniform/maximally unsure."""
    max_entropy = np.log(num_classes)
    return 1.0 - predictive_entropy(probs) / max_entropy


def confidence_weighted_fusion(probs_a: np.ndarray, probs_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        probs_a, probs_b: (N, K) softmax probability arrays from branch A and B.

    Returns:
        fused_probs: (N, K)
        weight_a: (N,) per-sample weight given to branch A
        weight_b: (N,) per-sample weight given to branch B
    """
    if probs_a.shape != probs_b.shape:
        raise ValueError(f"Branch outputs must match shape, got {probs_a.shape} vs {probs_b.shape}")
    num_classes = probs_a.shape[-1]
    c_a = confidence(probs_a, num_classes)
    c_b = confidence(probs_b, num_classes)
    denom = c_a + c_b + _EPS
    w_a = c_a / denom
    w_b = c_b / denom
    fused = w_a[..., None] * probs_a + w_b[..., None] * probs_b
    return fused, w_a, w_b
