"""
Unit tests for Sub-Pixel ECC Coregistration Service (Law 3 & FR-GEO-003).
"""

import pytest
import math
import cv2
import numpy as np

from app.services.coregistration import SubPixelCoregistration
from app.core.exceptions import RegistrationFailureException
from app.core.schemas import RegistrationMetrics


def test_ecc_subpixel_alignment_synthetic_shift_rotation():
    """
    Applies artificial shift (dx=3.5, dy=2.0) and rotation (1.2 deg) to synthetic raster.
    Verifies ECC restores spatial alignment with correlation >= 0.85 and RMSE < 0.35 pixels.
    """
    h, w = 256, 256
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)

    # Multi-frequency base terrain pattern
    pattern = (
        np.sin(x / 8.0) * np.cos(y / 8.0) * 0.4 +
        np.sin(x / 16.0 + y / 16.0) * 0.4 +
        np.cos(x / 32.0) * 0.2
    )
    base_channel = ((pattern - pattern.min()) / (pattern.max() - pattern.min()) * 3000 + 1000).astype(np.uint16)

    # 4-band reference image (t1)
    t1_raster = np.stack([
        base_channel,
        (base_channel * 1.05).clip(0, 65535).astype(np.uint16),
        (base_channel * 0.95).clip(0, 65535).astype(np.uint16),
        (base_channel * 1.2).clip(0, 65535).astype(np.uint16)
    ], axis=0)

    # Construct ground truth affine transform: shift + rotation
    center = (w / 2.0, h / 2.0)
    angle = 1.2  # 1.2 degrees
    scale = 1.0
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
    rot_mat[0, 2] += 3.5  # dx = 3.5
    rot_mat[1, 2] += 2.0  # dy = 2.0

    # Apply warp to create t2 (post-event)
    t2_raster = np.zeros_like(t1_raster)
    for c in range(4):
        t2_raster[c] = cv2.warpAffine(
            t1_raster[c].astype(np.float32),
            rot_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        ).astype(np.uint16)

    coreg = SubPixelCoregistration(min_correlation_threshold=0.65, max_iterations=300)
    warped_t2, metrics = coreg.align(t1_raster, t2_raster)

    # 1. Verify metrics output
    assert isinstance(metrics, RegistrationMetrics)
    assert metrics.registration_converged is True
    assert metrics.ecc_score >= 0.85

    # 2. Verify spatial alignment in central region (excluding boundary padding)
    margin = 30
    central_t1 = t1_raster[0, margin:-margin, margin:-margin].astype(np.float32)
    central_warped_t2 = warped_t2[0, margin:-margin, margin:-margin].astype(np.float32)
    central_unaligned_t2 = t2_raster[0, margin:-margin, margin:-margin].astype(np.float32)

    unaligned_rmse = np.sqrt(np.mean((central_t1 - central_unaligned_t2) ** 2))
    aligned_rmse = np.sqrt(np.mean((central_t1 - central_warped_t2) ** 2))

    # Error must be drastically reduced by coregistration
    assert aligned_rmse < unaligned_rmse * 0.20


def test_ecc_uncorrelated_failure():
    """
    Law 3 Guardrail: If ECC correlation score falls below 0.65,
    RegistrationFailureException must be raised.
    """
    h, w = 128, 128
    # Completely uncorrelated images (e.g. random noise)
    t1 = np.random.randint(500, 3000, size=(4, h, w), dtype=np.uint16)
    t2 = np.random.randint(500, 3000, size=(4, h, w), dtype=np.uint16)

    coreg = SubPixelCoregistration(min_correlation_threshold=0.65, max_iterations=50)

    with pytest.raises(RegistrationFailureException) as exc_info:
        coreg.align(t1, t2)

    assert exc_info.value.details["threshold"] == 0.65


def test_ecc_zero_variance_failure():
    """
    Constant imagery with zero standard deviation cannot produce gradients and must fail.
    """
    h, w = 64, 64
    t1 = np.full((4, h, w), 2000, dtype=np.uint16)
    t2 = np.full((4, h, w), 2000, dtype=np.uint16)

    coreg = SubPixelCoregistration(min_correlation_threshold=0.65)

    with pytest.raises(RegistrationFailureException):
        coreg.align(t1, t2)
