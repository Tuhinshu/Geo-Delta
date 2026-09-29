"""
Sub-Pixel Image Coregistration Service for GeoDelta
Implements Law 3: Enhanced Correlation Coefficient (ECC) Sub-Pixel Alignment (FR-GEO-003).

Formulation:
    W* = argmax_W ECC(I_t1^lum, W(I_t2^lum; W))
Where I^lum = 0.299 * Red + 0.587 * Green + 0.114 * Blue.

Enforces:
    - Minimum correlation score threshold (tau_ecc >= 0.65)
    - Trips RegistrationFailureException on failure or divergence
    - Warps all C channels of I_t2 using W*
"""

from typing import Tuple, Optional, List
import logging
import cv2
import numpy as np

from app.core.exceptions import RegistrationFailureException
from app.core.schemas import RegistrationMetrics
from app.core.config import settings
from app.services.normalizer import RadiometricNormalizer

# Public alias — tests import RegistrationFailureError from this module
RegistrationFailureError = RegistrationFailureException

logger = logging.getLogger(__name__)


class SubPixelCoregistration:
    """
    Sub-pixel image coregistration engine utilizing OpenCV Enhanced Correlation Coefficient (ECC).
    Aligns post-event imagery (t2) to pre-event reference (t1) at sub-pixel precision.
    """

    def __init__(
        self,
        min_correlation_threshold: Optional[float] = None,
        max_iterations: int = 200,
        termination_eps: float = 1e-5,
        motion_type: int = cv2.MOTION_AFFINE,
        min_threshold: Optional[float] = None
    ):
        """
        Args:
            min_correlation_threshold: Minimum ECC score required (default: settings.MIN_ECC_CORRELATION_SCORE = 0.65).
            max_iterations: Maximum optimization iterations for ECC gradient ascent.
            termination_eps: Parameter convergence threshold.
            motion_type: OpenCV warp motion model (default: cv2.MOTION_AFFINE).
            min_threshold: Alias for min_correlation_threshold.
        """
        threshold = min_threshold if min_threshold is not None else min_correlation_threshold
        self.min_threshold = (
            threshold
            if threshold is not None
            else settings.MIN_ECC_CORRELATION_SCORE
        )
        self.max_iterations = max_iterations
        self.termination_eps = termination_eps
        self.motion_type = motion_type

    def align(
        self,
        t1_raster: np.ndarray,
        t2_raster: np.ndarray,
        t1_luminance: Optional[np.ndarray] = None,
        t2_luminance: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, RegistrationMetrics]:
        """
        Executes sub-pixel ECC coregistration and warps t2_raster into spatial alignment with t1_raster.

        Args:
            t1_raster: Reference raster array of shape (C, H, W) or (H, W).
            t2_raster: Target raster array of shape (C, H, W) or (H, W) to warp.
            t1_luminance: Optional pre-computed 2D float32 luminance array for t1.
            t2_luminance: Optional pre-computed 2D float32 luminance array for t2.

        Returns:
            Tuple of:
                - warped_t2: Aligned t2 raster with identical shape and dtype as input t2_raster.
                - metrics: RegistrationMetrics containing ecc_score, warp_matrix, and convergence flag.

        Raises:
            RegistrationFailureException: If ECC correlation score is below threshold or fails to converge.
        """
        # Ensure dimensions match
        if t1_raster.shape != t2_raster.shape:
            raise ValueError(
                f"Raster shape mismatch: t1 shape {t1_raster.shape} vs t2 shape {t2_raster.shape}"
            )

        is_2d = t1_raster.ndim == 2
        working_t1 = t1_raster[np.newaxis, ...] if is_2d else t1_raster
        working_t2 = t2_raster[np.newaxis, ...] if is_2d else t2_raster

        channels, height, width = working_t1.shape

        # 1. Compute or extract luminance for ECC optimization
        if t1_luminance is None:
            if channels >= 3:
                norm_t1, _ = RadiometricNormalizer().normalize(working_t1)
                lum_t1 = RadiometricNormalizer.compute_luminance(norm_t1)
            else:
                norm_t1, _ = RadiometricNormalizer().normalize(working_t1[0])
                lum_t1 = norm_t1.astype(np.float32)
        else:
            lum_t1 = t1_luminance.astype(np.float32)

        if t2_luminance is None:
            if channels >= 3:
                norm_t2, _ = RadiometricNormalizer().normalize(working_t2)
                lum_t2 = RadiometricNormalizer.compute_luminance(norm_t2)
            else:
                norm_t2, _ = RadiometricNormalizer().normalize(working_t2[0])
                lum_t2 = norm_t2.astype(np.float32)
        else:
            lum_t2 = t2_luminance.astype(np.float32)

        # Ensure luminance is 2D float32
        lum_t1 = np.ascontiguousarray(lum_t1, dtype=np.float32)
        lum_t2 = np.ascontiguousarray(lum_t2, dtype=np.float32)

        # Check for non-trivial variance
        if np.std(lum_t1) < 1e-4 or np.std(lum_t2) < 1e-4:
            raise RegistrationFailureException(
                ecc_score=0.0,
                min_threshold=self.min_threshold
            )

        # 2. Initialize warp matrix
        if self.motion_type == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            # 2x3 for translation, euclidean, affine
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.max_iterations,
            self.termination_eps
        )

        try:
            ret = cv2.findTransformECC(
                templateImage=lum_t1,
                inputImage=lum_t2,
                warpMatrix=warp_matrix,
                motionType=self.motion_type,
                criteria=criteria
            )

            if isinstance(ret, tuple):
                ecc_score, warp_matrix = ret
            else:
                ecc_score = ret

            ecc_score = float(np.clip(ecc_score, 0.0, 1.0))

        except cv2.error as e:
            logger.warning(f"OpenCV findTransformECC failed to converge: {e}")
            raise RegistrationFailureException(
                ecc_score=0.0,
                min_threshold=self.min_threshold
            )

        # 3. Enforce Law 3 threshold invariant (ecc_score >= 0.65)
        if ecc_score < self.min_threshold:
            logger.warning(
                f"ECC correlation score {ecc_score:.4f} is below safety threshold {self.min_threshold:.4f}"
            )
            raise RegistrationFailureException(
                ecc_score=ecc_score,
                min_threshold=self.min_threshold
            )

        # 4. Warp all C channels of t2_raster
        warped_t2 = np.zeros_like(working_t2)
        warp_flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP

        for c_idx in range(channels):
            channel_data = working_t2[c_idx]
            original_dtype = channel_data.dtype

            if self.motion_type == cv2.MOTION_HOMOGRAPHY:
                warped_channel = cv2.warpPerspective(
                    channel_data.astype(np.float32),
                    warp_matrix,
                    (width, height),
                    flags=warp_flags,
                    borderMode=cv2.BORDER_REFLECT
                )
            else:
                warped_channel = cv2.warpAffine(
                    channel_data.astype(np.float32),
                    warp_matrix,
                    (width, height),
                    flags=warp_flags,
                    borderMode=cv2.BORDER_REFLECT
                )

            if np.issubdtype(original_dtype, np.integer):
                warped_t2[c_idx] = np.clip(np.round(warped_channel), 0, np.iinfo(original_dtype).max).astype(original_dtype)
            else:
                warped_t2[c_idx] = warped_channel.astype(original_dtype)

        metrics = RegistrationMetrics(
            ecc_score=ecc_score,
            warp_matrix=warp_matrix.tolist(),
            registration_converged=True
        )

        final_warped = warped_t2[0] if is_2d else warped_t2
        return final_warped, metrics
