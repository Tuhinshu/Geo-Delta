"""
Unit tests for Radiometric Percentile Normalizer (Law 2).
"""

import pytest
import numpy as np

from app.services.normalizer import RadiometricNormalizer


def test_percentile_normalization_uint16():
    """
    Law 2: 16-bit percentile dynamic range scaling maps valid uint16 values to [0.0, 1.0].
    """
    normalizer = RadiometricNormalizer(p_low=2.0, p_high=98.0, nodata_values=(0, 65535))

    # Synthetic 4-band uint16 raster (4, 100, 100)
    c, h, w = 4, 100, 100
    raster = np.random.randint(1000, 5000, size=(c, h, w), dtype=np.uint16)
    
    # Inject nodata values
    raster[:, 0:10, 0:10] = 0
    raster[:, 90:100, 90:100] = 65535

    norm_float, visual_uint8 = normalizer.normalize(raster)

    # 1. Float output invariants
    assert norm_float.shape == (c, h, w)
    assert norm_float.dtype == np.float32
    assert norm_float.min() >= 0.0
    assert norm_float.max() <= 1.0

    # 2. Nodata pixels must be mapped to 0.0
    assert np.all(norm_float[:, 0:10, 0:10] == 0.0)
    assert np.all(norm_float[:, 90:100, 90:100] == 0.0)

    # 3. Visual uint8 invariants
    assert visual_uint8.shape == (c, h, w)
    assert visual_uint8.dtype == np.uint8
    assert visual_uint8.min() >= 0
    assert visual_uint8.max() <= 255
    assert np.all(visual_uint8[:, 0:10, 0:10] == 0)


def test_normalizer_2d_raster():
    """
    Verify normalizer handles 2D single-band inputs correctly.
    """
    normalizer = RadiometricNormalizer()
    raster_2d = np.linspace(500, 3000, 10000, dtype=np.uint16).reshape((100, 100))

    norm_float, visual_uint8 = normalizer.normalize(raster_2d)
    assert norm_float.ndim == 2
    assert norm_float.shape == (100, 100)
    assert visual_uint8.ndim == 2
    assert visual_uint8.shape == (100, 100)
    assert norm_float.min() == 0.0
    assert norm_float.max() == 1.0


def test_normalizer_homogeneous_band():
    """
    Verify normalizer gracefully handles homogeneous/constant bands without division by zero.
    """
    normalizer = RadiometricNormalizer()
    constant_raster = np.full((3, 50, 50), 2000, dtype=np.uint16)

    norm_float, visual_uint8 = normalizer.normalize(constant_raster)
    assert not np.isnan(norm_float).any()
    assert not np.isinf(norm_float).any()
    assert norm_float.shape == (3, 50, 50)


def test_panchromatic_luminance():
    """
    Verify panchromatic luminance band: I_lum = 0.299 * R + 0.587 * G + 0.114 * B.
    """
    normalizer = RadiometricNormalizer()
    c, h, w = 4, 64, 64
    raster = np.random.randint(1000, 4000, size=(c, h, w), dtype=np.uint16)
    norm_float, _ = normalizer.normalize(raster)

    lum = RadiometricNormalizer.compute_luminance(norm_float, red_idx=0, green_idx=1, blue_idx=2)
    assert lum.shape == (h, w)
    assert lum.dtype == np.float32
    assert lum.min() >= 0.0
    assert lum.max() <= 1.0

    # Test exact weighting on pure colors
    pure_red = np.zeros((3, 10, 10), dtype=np.float32)
    pure_red[0] = 1.0  # Red = 1.0
    assert np.allclose(RadiometricNormalizer.compute_luminance(pure_red), 0.299, atol=1e-5)

    pure_green = np.zeros((3, 10, 10), dtype=np.float32)
    pure_green[1] = 1.0  # Green = 1.0
    assert np.allclose(RadiometricNormalizer.compute_luminance(pure_green), 0.587, atol=1e-5)

    pure_blue = np.zeros((3, 10, 10), dtype=np.float32)
    pure_blue[2] = 1.0  # Blue = 1.0
    assert np.allclose(RadiometricNormalizer.compute_luminance(pure_blue), 0.114, atol=1e-5)


def test_invalid_percentiles():
    """
    Verify validation of percentile bounds.
    """
    with pytest.raises(ValueError):
        RadiometricNormalizer(p_low=95.0, p_high=5.0)

    with pytest.raises(ValueError):
        RadiometricNormalizer(p_low=-1.0, p_high=98.0)
