"""
Tests for the spectral-branch input modes (branch-input ablation axis).

Motivation: with the default "rgb_plus_indices" mode the spectral branch's input
is a strict superset of the RGB branch's, so the branches cannot fail
independently and confidence-aware fusion (C1) has nothing to arbitrate. These
modes let that be tested rather than assumed.
"""
import pytest

from src.data.dataset import SPECTRAL_BRANCH_MODES, spectral_in_chans


def test_rgb_plus_indices_channel_count():
    assert spectral_in_chans("rgb_plus_indices", 1) == 4
    assert spectral_in_chans("rgb_plus_indices", 2) == 5


def test_indices_only_channel_count():
    assert spectral_in_chans("indices_only", 1) == 1
    assert spectral_in_chans("indices_only", 3) == 3


def test_nonrgb_channel_count():
    # NIR + SWIR + the index maps
    assert spectral_in_chans("nonrgb", 1) == 3
    assert spectral_in_chans("nonrgb", 2) == 4


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        spectral_in_chans("everything", 1)


def test_all_declared_modes_are_supported():
    for mode in SPECTRAL_BRANCH_MODES:
        assert spectral_in_chans(mode, 1) >= 1


def test_only_default_mode_includes_rgb_channels():
    """The point of the non-default modes: no RGB overlap with branch A."""
    num_idx = 1
    assert spectral_in_chans("rgb_plus_indices", num_idx) - num_idx == 3
    assert spectral_in_chans("indices_only", num_idx) - num_idx == 0
    assert spectral_in_chans("nonrgb", num_idx) - num_idx == 2


def test_model_accepts_non_default_spectral_channels():
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from src.models.ensemble import DualBranchEfficientNet

    chans = spectral_in_chans("nonrgb", 1)
    model = DualBranchEfficientNet(num_classes=4, spectral_in_chans=chans, pretrained=False)
    assert model.spectral_in_chans == chans

    out = model(torch.randn(2, 3, 32, 32), torch.randn(2, chans, 32, 32))
    assert out["logits_rgb"].shape == (2, 4)
    assert out["logits_spectral"].shape == (2, 4)
    assert out["fused_probs"].shape == (2, 4)


def test_model_rejects_zero_channel_spectral_branch():
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    from src.models.ensemble import DualBranchEfficientNet

    with pytest.raises(ValueError):
        DualBranchEfficientNet(num_classes=4, spectral_in_chans=0, pretrained=False)


# --- C3: attention switch (was hardcoded; configs/config.yaml advertised it) ---

def test_attention_none_replaces_eca_with_identity():
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from src.models.ensemble import DualBranchEfficientNet

    m = DualBranchEfficientNet(num_classes=4, pretrained=False, attention="none")
    assert isinstance(m.branch_rgb.attention, torch.nn.Identity)
    assert isinstance(m.branch_spectral.attention, torch.nn.Identity)
    assert m.attention_mode == "none"


def test_attention_eca_is_the_default():
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    from src.models.ensemble import DualBranchEfficientNet
    from src.models.attention import ECALayer

    m = DualBranchEfficientNet(num_classes=4, pretrained=False)
    assert isinstance(m.branch_rgb.attention, ECALayer)
    assert m.attention_mode == "eca"


def test_attention_ablation_changes_only_eca_parameters():
    """The ablation must remove ECA and nothing else, or it measures the wrong thing."""
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    from src.models.ensemble import DualBranchEfficientNet

    with_eca = DualBranchEfficientNet(num_classes=4, pretrained=False, attention="eca")
    without = DualBranchEfficientNet(num_classes=4, pretrained=False, attention="none")
    n_with = sum(p.numel() for p in with_eca.parameters())
    n_without = sum(p.numel() for p in without.parameters())
    eca_params = sum(p.numel() for p in with_eca.branch_rgb.attention.parameters()) + \
                 sum(p.numel() for p in with_eca.branch_spectral.attention.parameters())
    assert n_with - n_without == eca_params
    assert 0 < eca_params < 100          # ECA is a k-length 1D conv: single digits per module


def test_attention_none_still_produces_valid_outputs():
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from src.models.ensemble import DualBranchEfficientNet

    m = DualBranchEfficientNet(num_classes=4, pretrained=False, attention="none")
    out = m(torch.randn(2, 3, 32, 32), torch.randn(2, 4, 32, 32))
    assert out["fused_probs"].shape == (2, 4)
    assert torch.allclose(out["fused_probs"].sum(-1), torch.ones(2), atol=1e-5)


def test_unknown_attention_mode_raises():
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    from src.models.ensemble import DualBranchEfficientNet

    with pytest.raises(ValueError):
        DualBranchEfficientNet(num_classes=4, pretrained=False, attention="cbam")
