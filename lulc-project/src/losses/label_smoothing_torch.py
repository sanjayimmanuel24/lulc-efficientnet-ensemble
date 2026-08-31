"""
Production label-smoothing loss. Uses torch's built-in
`label_smoothing` argument (implements the exact formula documented and
unit-tested in src/losses/label_smoothing.py) rather than a hand-rolled
version -- no reason to reimplement a one-line, well-tested library
feature. Equivalence with the NumPy reference is checked in
test_models_torch.py.
"""
import torch.nn as nn


def build_criterion(smoothing: float = 0.1) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=smoothing)
