"""
These tests require torch and timm. They are written to be run in the
project's actual training environment (GPU cluster / Colab / local
machine with requirements.txt installed) -- NOT inside the lightweight
sandbox used to build this repo, which has no GPU and where installing
the full CUDA-bundled torch wheel would cost several GB for no benefit.
`pytest.importorskip` makes that an explicit, visible skip rather than a
silent gap: run `pip install -r requirements.txt && pytest tests/` in a
real environment to execute these.

The single most important test here is
`test_overfit_single_batch_sanity_check`: the classic "can this model
actually learn anything at all" check recommended before trusting any
larger training run.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from src.models.attention import ECALayer
from src.models.ensemble import DualBranchEfficientNet
from src.models.fusion_torch import confidence_weighted_fusion as torch_fusion
from src.models.fusion import confidence_weighted_fusion as numpy_fusion
from src.losses.label_smoothing import label_smoothing_cross_entropy
from src.losses.label_smoothing_torch import build_criterion
from src.explainability.gradcam import GradCAM


def test_eca_preserves_shape():
    x = torch.randn(2, 32, 16, 16)
    layer = ECALayer(channels=32)
    out = layer(x)
    assert out.shape == x.shape


def test_torch_fusion_matches_numpy_reference():
    rng = np.random.default_rng(0)
    logits_a = rng.normal(size=(8, 6)).astype(np.float32)
    logits_b = rng.normal(size=(8, 6)).astype(np.float32)
    probs_a_np = np.exp(logits_a) / np.exp(logits_a).sum(-1, keepdims=True)
    probs_b_np = np.exp(logits_b) / np.exp(logits_b).sum(-1, keepdims=True)

    fused_np, wa_np, wb_np = numpy_fusion(probs_a_np, probs_b_np)
    fused_t, wa_t, wb_t = torch_fusion(torch.tensor(probs_a_np), torch.tensor(probs_b_np))

    assert np.allclose(fused_np, fused_t.numpy(), atol=1e-5)
    assert np.allclose(wa_np, wa_t.numpy(), atol=1e-5)
    assert np.allclose(wb_np, wb_t.numpy(), atol=1e-5)


def test_torch_label_smoothing_matches_numpy_reference():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(16, 5)).astype(np.float32)
    labels = rng.integers(0, 5, size=16)

    numpy_loss = label_smoothing_cross_entropy(logits, labels, num_classes=5, smoothing=0.1)
    criterion = build_criterion(smoothing=0.1)
    torch_loss = criterion(torch.tensor(logits), torch.tensor(labels, dtype=torch.long)).item()

    assert np.isclose(numpy_loss, torch_loss, atol=1e-4)


def test_dual_branch_forward_shapes():
    model = DualBranchEfficientNet(num_classes=10, num_spectral_indices=1, pretrained=False)
    rgb = torch.randn(2, 3, 64, 64)
    rgb_plus_idx = torch.randn(2, 4, 64, 64)  # 3 RGB + 1 NDVI channel
    out = model(rgb, rgb_plus_idx)
    assert out["logits_rgb"].shape == (2, 10)
    assert out["logits_spectral"].shape == (2, 10)
    assert out["fused_probs"].shape == (2, 10)
    assert torch.allclose(out["fused_probs"].sum(dim=-1), torch.ones(2), atol=1e-5)


def test_default_temperature_is_identity():
    """Before calibration is fit, temperature=1.0 must be a strict no-op."""
    model = DualBranchEfficientNet(num_classes=5, num_spectral_indices=1, pretrained=False)
    assert model.temperature_rgb.item() == 1.0
    assert model.temperature_spectral.item() == 1.0


def test_set_temperatures_changes_fused_confidence_but_not_argmax():
    """
    Temperature scaling must reshape confidence without changing which
    class wins (Section 6 of the blueprint: T only rescales logits, it
    doesn't reorder them within a single branch). A larger T should make
    that branch's contribution less peaked (lower max probability).
    """
    torch.manual_seed(0)
    model = DualBranchEfficientNet(num_classes=6, num_spectral_indices=1, pretrained=False)
    model.eval()
    rgb = torch.randn(3, 3, 64, 64)
    rgb_plus_idx = torch.randn(3, 4, 64, 64)

    with torch.no_grad():
        out_before = model(rgb, rgb_plus_idx)
        preds_before = out_before["fused_probs"].argmax(dim=-1)

        model.set_temperatures(t_rgb=5.0, t_spectral=5.0)  # smooth both branches a lot
        out_after = model(rgb, rgb_plus_idx)
        preds_after = out_after["fused_probs"].argmax(dim=-1)

    assert torch.equal(preds_before, preds_after), "temperature scaling must not change predictions"
    assert out_after["fused_probs"].max(dim=-1).values.mean() < out_before["fused_probs"].max(dim=-1).values.mean()


def test_overfit_single_batch_sanity_check():
    """
    The classic sanity check: a correctly-wired model + loss + optimizer
    must be able to drive the loss on a single fixed batch down close to
    zero within a modest number of steps. If this fails, there is a bug in
    the model, the loss, or the optimizer wiring -- no amount of full-scale
    training will fix that. This is run before any real training run.
    """
    torch.manual_seed(0)
    model = DualBranchEfficientNet(num_classes=4, num_spectral_indices=1, pretrained=False)
    rgb = torch.randn(4, 3, 64, 64)
    rgb_plus_idx = torch.randn(4, 4, 64, 64)
    labels = torch.tensor([0, 1, 2, 3])

    criterion = build_criterion(smoothing=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(60):
        optimizer.zero_grad()
        out = model(rgb, rgb_plus_idx)
        loss = criterion(out["logits_rgb"], labels) + criterion(out["logits_spectral"], labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.1, f"loss did not collapse on a single batch: {losses[0]} -> {losses[-1]}"

    with torch.no_grad():
        final_out = model(rgb, rgb_plus_idx)
        preds = final_out["fused_probs"].argmax(dim=-1)
    assert (preds == labels).all(), "model failed to memorize a 4-sample batch"


def test_gradcam_produces_correct_shape_heatmap():
    model = DualBranchEfficientNet(num_classes=4, num_spectral_indices=1, pretrained=False)
    model.eval()
    target_layer = model.branch_rgb.attention  # post-attention feature map
    cam = GradCAM(model.branch_rgb, target_layer)

    rgb = torch.randn(1, 3, 64, 64, requires_grad=True)

    def forward_fn(x):
        return model.branch_rgb(x)

    heatmap = cam(rgb, class_idx=0, forward_fn=forward_fn)
    assert heatmap.shape == (64, 64)
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0 + 1e-5
    cam.remove_hooks()
