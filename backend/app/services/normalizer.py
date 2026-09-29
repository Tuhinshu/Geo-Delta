"""
Radiometric Normalization & Dynamic Range Scaling for GeoDelta
Implements Law 2: 16-Bit Radiometric Percentile Normalization Invariant.

Formulation:
    I_norm = clip((I - P_2) / (P_98 - P_2 + eps), 0.0, 1.0)
Calculated strictly over valid, non-nodata pixels per channel.
Preserves raw radiometric arrays for physical index calculations (NDVI, NDWI).
"""

from typing import Tuple, Optional, Sequence
import numpy as np


class RadiometricNormalizer:
    """
    Normalizes multi-spectral satellite imagery (uint16 / float32)
    using robust percentile dynamic range scaling (P2 - P98).
    """

    def __init__(
        self,
        p_low: float = 2.0,
        p_high: float = 98.0,
        nodata_values: Sequence[int] = (0, 65535),
        epsilon: float = 1e-6
    ):
        """
        Args:
            p_low: Lower cumulative percentile (default: 2.0%)
            p_high: Upper cumulative percentile (default: 98.0%)
            nodata_values: Sequence of nodata values to mask out from percentile calculation
            epsilon: Numerical stability constant preventing division by zero
        """
        if not (0.0 <= p_low < p_high <= 100.0):
            raise ValueError(f"Invalid percentiles: p_low={p_low}, p_high={p_high}. Must be 0 <= p_low < p_high <= 100.")
        self.p_low = p_low
        self.p_high = p_high
        self.nodata_values = set(nodata_values)
        self.epsilon = epsilon

    def normalize(
        self,
        raster: np.ndarray,
        nodata: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalizes multi-band raster into float32 [0.0, 1.0] and visual uint8 [0, 255].
        
        Args:
            raster: Multi-spectral array of shape (C, H, W) or single-band (H, W).
            nodata: Optional explicit nodata value.
            
        Returns:
            Tuple of:
                - normalized_float: np.ndarray (float32) in range [0.0, 1.0]
                - visual_uint8: np.ndarray (uint8) in range [0, 255]
        """
        is_2d = raster.ndim == 2
        if is_2d:
            working_raster = raster[np.newaxis, ...]  # Shape (1, H, W)
        elif raster.ndim == 3:
            working_raster = raster
        else:
            raise ValueError(f"Expected 2D or 3D raster array, got shape {raster.shape}")

        c, h, w = working_raster.shape
        normalized = np.zeros((c, h, w), dtype=np.float32)

        for band_idx in range(c):
            band_data = working_raster[band_idx]
            
            # Determine valid pixel mask
            valid_mask = np.isfinite(band_data)
            for nd in self.nodata_values:
                valid_mask &= (band_data != nd)
            if nodata is not None:
                valid_mask &= (band_data != nodata)

            valid_pixels = band_data[valid_mask]

            if valid_pixels.size == 0:
                # All pixels are nodata or invalid
                normalized[band_idx] = 0.0
                continue

            p2 = float(np.percentile(valid_pixels, self.p_low))
            p98 = float(np.percentile(valid_pixels, self.p_high))
            delta = p98 - p2

            if delta <= self.epsilon:
                # Homogeneous or uniform surface reflectance
                band_norm = np.zeros_like(band_data, dtype=np.float32)
                band_norm[valid_mask] = 0.5
            else:
                scaled = (band_data.astype(np.float32) - p2) / (delta + self.epsilon)
                band_norm = np.clip(scaled, 0.0, 1.0)
                band_norm[~valid_mask] = 0.0

            normalized[band_idx] = band_norm

        visual_uint8 = np.clip(np.round(normalized * 255.0), 0, 255).astype(np.uint8)

        if is_2d:
            return normalized[0], visual_uint8[0]
        return normalized, visual_uint8

    @staticmethod
    def compute_luminance(
        normalized_raster: np.ndarray,
        red_idx: int = 0,
        green_idx: int = 1,
        blue_idx: int = 2
    ) -> np.ndarray:
        """
        Computes panchromatic / luminance channel from normalized optical bands:
            I_lum = 0.299 * Red + 0.587 * Green + 0.114 * Blue
        Used for sub-pixel ECC coregistration (Law 3).

        Args:
            normalized_raster: Multi-band float32 array in [0.0, 1.0], shape (C, H, W).
            red_idx: 0-indexed channel position for Red (default: 0)
            green_idx: 0-indexed channel position for Green (default: 1)
            blue_idx: 0-indexed channel position for Blue (default: 2)

        Returns:
            np.ndarray (float32) of shape (H, W) in range [0.0, 1.0]
        """
        if normalized_raster.ndim != 3 or normalized_raster.shape[0] < 3:
            raise ValueError(
                f"Normalized raster must have at least 3 channels (C >= 3, H, W). Got shape {normalized_raster.shape}"
            )

        r = normalized_raster[red_idx].astype(np.float32)
        g = normalized_raster[green_idx].astype(np.float32)
        b = normalized_raster[blue_idx].astype(np.float32)

        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return np.clip(luminance, 0.0, 1.0)
