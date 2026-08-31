"""
Tests for the single-backbone literature baseline (BLUEPRINT Section 9/10).

The baseline must be measured on OUR folds under the SAME protocol as the
proposed model, or the comparison to Helber et al. (2019) is not like-for-like.
"""
import pytest


def _model(**kw):
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    from src.models.baseline import SingleBackboneBaseline
    kw.setdefault("pretrained", False)
    kw.setdefault("backbone_name", "resnet18")   # small stand-in for resnet50 in tests
    return SingleBackboneBaseline(num_classes=4, **kw)


def test_output_dict_matches_dual_branch_interface():
    torch = pytest.importorskip("torch")
    out = _model()(torch.randn(2, 3, 32, 32))
    for key in ("logits_rgb", "logits_spectral", "fused_probs", "weight_rgb", "weight_spectral"):
        assert key in out


def test_spectral_input_is_accepted_and_ignored():
    """The training loop passes both tensors; the baseline must not need special-casing."""
    torch = pytest.importorskip("torch")
    m = _model()
    rgb = torch.randn(2, 3, 32, 32)
    a = m(rgb, torch.randn(2, 4, 32, 32))
    b = m(rgb, torch.randn(2, 7, 32, 32))
    assert torch.allclose(a["logits_rgb"], b["logits_rgb"])


def test_fused_probs_are_a_plain_softmax_not_a_fusion():
    torch = pytest.importorskip("torch")
    m = _model()
    out = m(torch.randn(2, 3, 32, 32))
    assert torch.allclose(out["fused_probs"], torch.softmax(out["logits_rgb"], dim=-1), atol=1e-6)
    assert torch.allclose(out["fused_probs"].sum(-1), torch.ones(2), atol=1e-5)


def test_weights_report_all_mass_on_the_single_branch():
    torch = pytest.importorskip("torch")
    out = _model()(torch.randn(3, 3, 32, 32))
    assert torch.allclose(out["weight_rgb"], torch.ones(3))
    assert torch.allclose(out["weight_spectral"], torch.zeros(3))


def test_temperature_scaling_applies():
    torch = pytest.importorskip("torch")
    m = _model()
    x = torch.randn(2, 3, 32, 32)
    before = m(x)["fused_probs"].max(-1).values
    m.set_temperatures(3.0)
    after = m(x)["fused_probs"].max(-1).values
    assert (after < before).all()          # higher T -> less confident


def test_duplicate_logits_would_double_the_loss_without_zero_weighting():
    """
    Guards the wiring contract: logits_spectral aliases logits_rgb, so a runner that
    forgets branch_loss_weight_spectral=0 silently doubles this run's learning signal.
    """
    torch = pytest.importorskip("torch")
    from src.training.train import TrainConfig, compute_loss
    from src.losses.label_smoothing_torch import build_criterion

    out = _model()(torch.randn(4, 3, 32, 32))
    labels = torch.tensor([0, 1, 2, 3])
    crit = build_criterion(0.1)
    both = compute_loss(out, labels, crit, TrainConfig(num_classes=4))
    single = compute_loss(out, labels, crit,
                          TrainConfig(num_classes=4, branch_loss_weight_spectral=0.0))
    assert torch.allclose(both, single * 2, atol=1e-5)
