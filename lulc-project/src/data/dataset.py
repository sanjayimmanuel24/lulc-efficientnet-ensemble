"""
Dataset classes.

EuroSATMSDataset reads the EuroSAT *multispectral* release (13-band
Sentinel-2 GeoTIFFs, not the RGB-only JPEG release) so that NDVI/NDWI/NDBI
are computed from real reflectance bands rather than approximated from
RGB -- see src/data/spectral_indices.py docstring for why that distinction
matters. Download the "EuroSATallBands" release before using this class;
this repo does not bundle satellite imagery.

SyntheticLULCDataset produces random tensors with the same shapes/dict
keys the real dataset returns, so the rest of the pipeline (model, loss,
training loop, tests) can be built and verified without needing the
dataset downloaded first -- useful for onboarding a new environment or CI.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.spectral_indices import stack_indices

# Sentinel-2 band indices within the 13-band EuroSAT MS GeoTIFF stack.
# (B02=Blue, B03=Green, B04=Red, B08=NIR, B11=SWIR1)
BAND_INDEX = {"blue": 1, "green": 2, "red": 3, "nir": 7, "swir": 10}

# What the spectral branch actually receives. This is an ablation axis, not a
# free choice: the first real run showed the default "rgb_plus_indices" branch
# strictly dominates the RGB branch (its input is a superset), so the two
# branches share failure modes and confidence-aware fusion has nothing to
# arbitrate -- see results/branch_breakdown.json.
#   rgb_plus_indices : [R,G,B] + index maps          (3+I ch) -- blueprint default
#   indices_only     : index maps alone              (I ch)
#   nonrgb           : [NIR,SWIR] + index maps       (2+I ch) -- disjoint from RGB
SPECTRAL_BRANCH_MODES = ("rgb_plus_indices", "indices_only", "nonrgb")


def spectral_in_chans(mode: str, num_indices: int) -> int:
    """Input channel count for the spectral branch under `mode`."""
    if mode == "rgb_plus_indices":
        return 3 + num_indices
    if mode == "indices_only":
        return num_indices
    if mode == "nonrgb":
        return 2 + num_indices
    raise ValueError(f"spectral_branch_mode must be one of {SPECTRAL_BRANCH_MODES}, got {mode!r}")


class EuroSATMSDataset(Dataset):
    def __init__(self, file_paths: list[str], labels: list[int], image_size: int = 224,
                 indices: tuple[str, ...] = ("ndvi",), transform=None,
                 spectral_branch_mode: str = "rgb_plus_indices"):
        try:
            import rasterio  # local import: only needed if this class is actually used
        except ImportError as e:
            raise ImportError(
                "EuroSATMSDataset requires `rasterio` to read multispectral GeoTIFFs. "
                "Install it with `pip install rasterio` (see requirements.txt)."
            ) from e
        # NOTE: deliberately not stored on the instance. On Windows, DataLoader
        # workers are spawned, which pickles the dataset object -- and a module
        # object is not picklable ("TypeError: cannot pickle 'module' object").
        # The import above stays as an eager, friendly-error check; __getitem__
        # re-imports from sys.modules, which is a dict lookup, not a real import.
        self.file_paths = file_paths
        self.labels = labels
        self.image_size = image_size
        self.indices = indices
        self.transform = transform
        # validates the mode eagerly; raises for anything unrecognised
        self.spectral_channels = spectral_in_chans(spectral_branch_mode, len(indices))
        self.spectral_branch_mode = spectral_branch_mode

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict:
        import rasterio  # cached in sys.modules; see __init__ for why it isn't an attribute

        with rasterio.open(self.file_paths[idx]) as src:
            data = src.read().astype(np.float32) / 10000.0  # Sentinel-2 reflectance scaling (10000 = 100%)

        bands = {name: data[i] for name, i in BAND_INDEX.items()}
        rgb = np.stack([bands["red"], bands["green"], bands["blue"]], axis=0)
        index_maps = stack_indices(bands, indices=self.indices)
        # RGB always leads the stack so the single transform below can normalise
        # exactly the first 3 channels; the branch views are sliced out after.
        parts = [rgb]
        if self.spectral_branch_mode == "nonrgb":
            parts.append(np.stack([bands["nir"], bands["swir"]], axis=0))
        parts.append(index_maps)
        combined = np.concatenate(parts, axis=0)
        combined_t = torch.from_numpy(combined)

        # Transform the single combined tensor once, then slice out the RGB
        # view -- NOT two independent transform() calls on separately-built
        # tensors. Two independent calls would apply e.g. resize
        # deterministically-fine today, but would silently diverge (RGB
        # branch flipped one way, spectral branch flipped another) the
        # moment any random augmentation (random flip/crop) is added later.
        # One transform call, one slice, is correct by construction.
        if self.transform is not None:
            combined_t = self.transform(combined_t)
        rgb_t = combined_t[:3]
        spectral_t = combined_t if self.spectral_branch_mode == "rgb_plus_indices" else combined_t[3:]

        return {
            "rgb": rgb_t,
            "spectral": spectral_t,
            # Back-compat alias: identical tensor, kept so existing callers and
            # tests that ask for "rgb_plus_indices" keep working. Prefer "spectral".
            "rgb_plus_indices": spectral_t,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class SyntheticLULCDataset(Dataset):
    """Random data with the real dataset's shapes/keys, for pipeline testing without real imagery."""

    def __init__(self, num_samples: int = 32, num_classes: int = 10, num_indices: int = 1, image_size: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.num_samples = num_samples
        self.image_size = image_size
        self.rgb = rng.normal(size=(num_samples, 3, image_size, image_size)).astype(np.float32)
        self.extra_indices = rng.uniform(-1, 1, size=(num_samples, num_indices, image_size, image_size)).astype(np.float32)
        self.labels = rng.integers(0, num_classes, size=num_samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        rgb = torch.from_numpy(self.rgb[idx])
        rgb_plus_idx = torch.from_numpy(np.concatenate([self.rgb[idx], self.extra_indices[idx]], axis=0))
        return {
            "rgb": rgb,
            "rgb_plus_indices": rgb_plus_idx,
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }
