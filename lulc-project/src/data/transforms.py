"""
Preprocessing transform for a single (3+I, H, W) combined RGB+index tensor
(see src/data/dataset.py::EuroSATMSDataset -- the RGB-only branch input is
always a slice of this same transformed tensor, never a separately
transformed copy, so the two branches can never see misaligned spatial
transforms).

Two things happen here, and only to the first 3 (RGB) channels:
  1. Resize to the current progressive-resizing stage's target size.
  2. ImageNet mean/std normalization -- because both branches' backbones
     start from ImageNet-pretrained weights, whose early layers expect
     input in that normalized range.

The extra spectral-index channel(s) (e.g. NDVI) are resized identically
but NOT ImageNet-normalized: NDVI is already bounded to roughly [-1, 1],
which is a reasonably well-scaled range for a conv net on its own, and
there is no meaningful "ImageNet statistics" for an index channel to be
normalized against in the first place.

`train=True` applies dihedral augmentation (random horizontal/vertical flip
plus a random multiple of 90 degrees). These are the label-preserving
transforms for overhead imagery specifically: a satellite patch has no
canonical "up", so a rotated or mirrored forest is still a forest. Colour
jitter and shear are deliberately NOT applied -- the spectral index channels
carry physical reflectance ratios that photometric jitter would corrupt.

Crucially the augmentation is applied to the WHOLE combined tensor before the
RGB view is sliced out, so both branches always see the same geometry. That is
the invariant the single-transform design exists to protect, and it is exactly
what would break if each branch were transformed independently.

Augmentation is opt-in (`train=True` is only passed for training loaders, and
runners expose --augment) so that runs completed before it existed remain
reproducible.
"""
import torch
import torch.nn.functional as F

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class EuroSATTransform:
    def __init__(self, image_size: int, train: bool = False, augment: bool = False):
        self.image_size = image_size
        self.train = train
        # augmentation only ever fires on training loaders, never val/test
        self.augment = augment and train

    def __call__(self, combined: torch.Tensor) -> torch.Tensor:
        """combined: (3+I, H, W) raw reflectance + index tensor. Returns same shape, resized/normalized."""
        x = combined.unsqueeze(0)
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = x.squeeze(0)

        if self.augment:
            x = self._augment(x)

        rgb = x[:3].clamp(0.0, 1.0)  # clip rare out-of-range reflectance (cloud/snow/quantization artifacts)
        rgb = (rgb - _IMAGENET_MEAN.to(x.device)) / _IMAGENET_STD.to(x.device)
        # Non-RGB channels are reflectance bands and index maps. Indices are already
        # bounded to [-1,1]; raw NIR/SWIR reflectance can exceed 1 over cloud and snow,
        # the same artefact the RGB clamp handles, so clip those to a physical range too.
        rest = x[3:].clamp(-1.0, 1.5)
        return torch.cat([rgb, rest], dim=0)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dihedral group D4: random flips + 90-degree rotations, applied to ALL
        channels at once so the RGB slice and the spectral channels stay aligned.
        Uses torch's global RNG, so TrainConfig.seed controls it.
        """
        if torch.rand(()) < 0.5:
            x = torch.flip(x, dims=[2])          # horizontal
        if torch.rand(()) < 0.5:
            x = torch.flip(x, dims=[1])          # vertical
        k = int(torch.randint(0, 4, ()))
        if k:
            x = torch.rot90(x, k, dims=[1, 2])
        return x
