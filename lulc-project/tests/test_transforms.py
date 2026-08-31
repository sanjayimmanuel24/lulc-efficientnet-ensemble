"""
Tests for EuroSATTransform, especially the augmentation added for BLUEPRINT
ablation #9. The load-bearing property is that augmentation is applied to the
combined tensor BEFORE the RGB view is sliced, so the two branches can never
receive differently-oriented views of the same patch.
"""
import pytest
import torch

from src.data.transforms import EuroSATTransform


def _combined(c=4, h=8, w=8):
    return torch.arange(float(c * h * w)).reshape(c, h, w) / (c * h * w)


def test_output_shape_and_resize():
    t = EuroSATTransform(image_size=16)
    out = t(_combined())
    assert out.shape == (4, 16, 16)


def test_no_augmentation_is_deterministic():
    t = EuroSATTransform(image_size=8, train=False)
    x = _combined()
    assert torch.allclose(t(x), t(x))


def test_augment_requires_train_flag():
    """augment=True on a val/test loader must be inert -- evaluation is never randomised."""
    t = EuroSATTransform(image_size=8, train=False, augment=True)
    assert t.augment is False
    x = _combined()
    assert torch.allclose(t(x), t(x))


def test_augmentation_actually_changes_some_outputs():
    t = EuroSATTransform(image_size=8, train=True, augment=True)
    x = _combined()
    torch.manual_seed(0)
    outs = [t(x) for _ in range(12)]
    assert any(not torch.allclose(outs[0], o) for o in outs[1:])


def test_augmentation_keeps_branch_views_aligned():
    """
    The invariant: the RGB branch input is a slice of the same augmented tensor
    the spectral branch sees. If the spectral channels were rotated differently
    from RGB, the spatial-misalignment bug this design prevents would be back.
    """
    t = EuroSATTransform(image_size=8, train=True, augment=True)
    # channel 3 is a copy of channel 0, so after any shared geometry they must match
    x = _combined()
    x[3] = x[0].clone()
    torch.manual_seed(1)
    for _ in range(10):
        out = t(x)
        rgb_r = out[0] * torch.tensor(0.229) + torch.tensor(0.485)   # undo ImageNet norm on R
        assert torch.allclose(rgb_r, out[3], atol=1e-4)


def test_augmentation_is_seed_controlled():
    t = EuroSATTransform(image_size=8, train=True, augment=True)
    x = _combined()
    torch.manual_seed(7); a = [t(x) for _ in range(5)]
    torch.manual_seed(7); b = [t(x) for _ in range(5)]
    for u, v in zip(a, b):
        assert torch.allclose(u, v)


def test_augmentation_preserves_value_set():
    """Flips/rot90 permute pixels; they must not interpolate or invent values."""
    t = EuroSATTransform(image_size=8, train=True, augment=True)
    x = _combined()
    torch.manual_seed(2)
    out = t(x)
    assert torch.allclose(torch.sort(out[3].flatten()).values,
                          torch.sort(x[3].flatten()).values, atol=1e-5)


def test_non_rgb_channels_are_clipped_to_physical_range():
    """Raw NIR/SWIR exceed 1.0 over cloud and snow; unclipped they skew the spectral branch."""
    t = EuroSATTransform(image_size=4, train=False)
    x = torch.zeros(5, 4, 4)
    x[3] = 9.0     # absurd reflectance spike
    x[4] = -5.0
    out = t(x)
    assert out[3].max() <= 1.5 + 1e-6
    assert out[4].min() >= -1.0 - 1e-6


def test_rgb_channels_still_imagenet_normalised():
    t = EuroSATTransform(image_size=4, train=False)
    x = torch.full((4, 4, 4), 0.485)
    out = t(x)
    assert torch.allclose(out[0], torch.zeros(4, 4), atol=1e-4)
