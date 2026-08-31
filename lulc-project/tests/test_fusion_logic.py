import numpy as np
import pytest
from src.models.fusion import predictive_entropy, confidence, confidence_weighted_fusion


def test_entropy_zero_for_one_hot():
    one_hot = np.array([[1.0, 0.0, 0.0, 0.0]])
    assert np.isclose(predictive_entropy(one_hot), 0.0, atol=1e-6)


def test_entropy_maximal_for_uniform():
    k = 4
    uniform = np.full((1, k), 1.0 / k)
    assert np.isclose(predictive_entropy(uniform), np.log(k), atol=1e-6)


def test_confidence_bounds():
    k = 5
    one_hot = np.eye(k)[0][None, :]
    uniform = np.full((1, k), 1.0 / k)
    assert np.isclose(confidence(one_hot, k), 1.0, atol=1e-6)
    assert np.isclose(confidence(uniform, k), 0.0, atol=1e-6)


def test_weights_always_sum_to_one():
    rng = np.random.default_rng(1)
    for _ in range(20):
        logits_a = rng.normal(size=(1, 6))
        logits_b = rng.normal(size=(1, 6))
        pa = np.exp(logits_a) / np.exp(logits_a).sum(-1, keepdims=True)
        pb = np.exp(logits_b) / np.exp(logits_b).sum(-1, keepdims=True)
        _, wa, wb = confidence_weighted_fusion(pa, pb)
        assert np.isclose(wa + wb, 1.0, atol=1e-6)


def test_confident_branch_dominates_fusion():
    # Branch A is certain (one-hot on class 2), branch B is uniform/unsure.
    k = 4
    p_a = np.array([[0.0, 0.0, 1.0, 0.0]])
    p_b = np.full((1, k), 1.0 / k)
    fused, w_a, w_b = confidence_weighted_fusion(p_a, p_b)
    assert w_a > w_b
    # Fused prediction should still pick class 2, and put most mass there.
    assert fused.argmax() == 2
    assert fused[0, 2] > 0.5


def test_equal_confidence_reduces_to_simple_average():
    # This is the key ablation property: when both branches are equally
    # confident, the adaptive rule must collapse to the naive baseline
    # (plain averaging), so the paper can honestly claim it *generalizes*
    # the baseline rather than being an unrelated alternative.
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(1, 5))
    p = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    # Same distribution on both branches -> identical entropy -> equal weights
    fused, w_a, w_b = confidence_weighted_fusion(p, p)
    assert np.isclose(w_a, 0.5, atol=1e-6)
    assert np.isclose(w_b, 0.5, atol=1e-6)
    assert np.allclose(fused, p, atol=1e-6)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        confidence_weighted_fusion(np.zeros((2, 3)), np.zeros((2, 4)))


def test_fused_output_is_valid_probability_distribution():
    rng = np.random.default_rng(3)
    la = rng.normal(size=(10, 7))
    lb = rng.normal(size=(10, 7))
    pa = np.exp(la) / np.exp(la).sum(-1, keepdims=True)
    pb = np.exp(lb) / np.exp(lb).sum(-1, keepdims=True)
    fused, _, _ = confidence_weighted_fusion(pa, pb)
    assert np.allclose(fused.sum(axis=-1), 1.0, atol=1e-6)
    assert (fused >= 0).all()
