"""
Spectral index computation for Sentinel-2 multispectral bands.

Deliberately pure NumPy / framework-independent: these formulas are the
scientific core of the "spectral-fusion" contribution, so their correctness
must be verifiable on its own, independent of whatever deep learning
framework the rest of the pipeline uses.

Requires the true Sentinel-2 bands (Green=B03, Red=B04, NIR=B08, SWIR1=B11),
NOT an approximation from RGB-only imagery. This matters: many student
projects compute a fake "NDVI" from RGB jpgs, which is not NDVI at all
(RGB has no near-infrared channel) and would not survive peer review.
Use the EuroSAT MS (multispectral, 13-band GeoTIFF) release, not the
RGB-only release, when these indices are required.
"""
from __future__ import annotations
import numpy as np

_EPS = 1e-8


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index. Range ~[-1, 1]."""
    return (nir - red) / (nir + red + _EPS)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalized Difference Water Index (McFeeters). Range ~[-1, 1]."""
    return (green - nir) / (green + nir + _EPS)


def ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalized Difference Built-up Index. Range ~[-1, 1]."""
    return (swir - nir) / (swir + nir + _EPS)


def stack_indices(bands: dict[str, np.ndarray], indices: tuple[str, ...] = ("ndvi",)) -> np.ndarray:
    """
    Compute the requested indices and stack them as extra channels.

    Args:
        bands: dict with keys among {"red", "green", "nir", "swir"}, each an
            (H, W) array of reflectance values.
        indices: which indices to compute, e.g. ("ndvi", "ndwi").

    Returns:
        (len(indices), H, W) array.
    """
    computed = {
        "ndvi": lambda: ndvi(bands["nir"], bands["red"]),
        "ndwi": lambda: ndwi(bands["green"], bands["nir"]),
        "ndbi": lambda: ndbi(bands["swir"], bands["nir"]),
    }
    missing = [i for i in indices if i not in computed]
    if missing:
        raise ValueError(f"Unknown indices requested: {missing}")
    return np.stack([computed[i]() for i in indices], axis=0)
