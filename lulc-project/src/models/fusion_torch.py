"""
Torch-native version of the confidence-weighted fusion rule specified and
unit-tested (in NumPy) in src/models/fusion.py. Implemented directly in
torch ops rather than importing that module, since bouncing tensors to
NumPy and back inside a training/inference loop is wasteful and would
break GPU autograd. See src/models/fusion.py's docstring for the formula
and rationale. test_models_torch.py checks that this produces the same
numbers as the NumPy reference on identical inputs.
"""
import torch

_EPS = 1e-8


def predictive_entropy(probs: torch.Tensor) -> torch.Tensor:
    safe = probs.clamp_min(_EPS)
    return -(probs * safe.log()).sum(dim=-1)


def confidence(probs: torch.Tensor, num_classes: int) -> torch.Tensor:
    max_entropy = torch.log(torch.tensor(float(num_classes), device=probs.device))
    return 1.0 - predictive_entropy(probs) / max_entropy


def confidence_weighted_fusion(probs_a: torch.Tensor, probs_b: torch.Tensor):
    """
    Args:
        probs_a, probs_b: (B, K) softmax probabilities.
    Returns:
        fused_probs (B, K), weight_a (B,), weight_b (B,)
    """
    num_classes = probs_a.shape[-1]
    c_a = confidence(probs_a, num_classes)
    c_b = confidence(probs_b, num_classes)
    denom = c_a + c_b + _EPS
    w_a = c_a / denom
    w_b = c_b / denom
    fused = w_a.unsqueeze(-1) * probs_a + w_b.unsqueeze(-1) * probs_b
    return fused, w_a, w_b
