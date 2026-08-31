import numpy as np
import pytest
from src.data.spectral_indices import ndvi, ndwi, ndbi, stack_indices


def test_ndvi_hand_computed():
    # Healthy vegetation: high NIR, low red reflectance -> NDVI close to +1
    nir = np.array([0.5])
    red = np.array([0.1])
    result = ndvi(nir, red)
    expected = (0.5 - 0.1) / (0.5 + 0.1)
    assert np.isclose(result, expected, atol=1e-6)
    assert result > 0.5  # vegetation signature


def test_ndvi_water_or_bare_soil_is_negative_or_low():
    # Water: low NIR relative to red -> NDVI negative
    nir = np.array([0.05])
    red = np.array([0.1])
    result = ndvi(nir, red)
    assert result < 0


def test_ndvi_no_div_by_zero():
    nir = np.array([0.0])
    red = np.array([0.0])
    result = ndvi(nir, red)
    assert np.isfinite(result).all()


def test_ndvi_range_bounded():
    rng = np.random.default_rng(0)
    nir = rng.uniform(0, 1, 1000)
    red = rng.uniform(0, 1, 1000)
    result = ndvi(nir, red)
    assert result.min() >= -1.0 - 1e-6
    assert result.max() <= 1.0 + 1e-6


def test_ndwi_hand_computed():
    green = np.array([0.4])
    nir = np.array([0.1])
    result = ndwi(green, nir)
    expected = (0.4 - 0.1) / (0.4 + 0.1)
    assert np.isclose(result, expected, atol=1e-6)


def test_ndbi_hand_computed():
    swir = np.array([0.4])
    nir = np.array([0.2])
    result = ndbi(swir, nir)
    expected = (0.4 - 0.2) / (0.4 + 0.2)
    assert np.isclose(result, expected, atol=1e-6)


def test_stack_indices_shape_and_order():
    bands = {
        "red": np.full((4, 4), 0.1),
        "green": np.full((4, 4), 0.3),
        "nir": np.full((4, 4), 0.5),
        "swir": np.full((4, 4), 0.4),
    }
    out = stack_indices(bands, indices=("ndvi", "ndwi"))
    assert out.shape == (2, 4, 4)
    assert np.isclose(out[0, 0, 0], ndvi(np.array(0.5), np.array(0.1)))
    assert np.isclose(out[1, 0, 0], ndwi(np.array(0.3), np.array(0.5)))


def test_stack_indices_rejects_unknown_index():
    bands = {"red": np.zeros((2, 2)), "nir": np.zeros((2, 2))}
    with pytest.raises(ValueError):
        stack_indices(bands, indices=("ndvi", "not_a_real_index"))
