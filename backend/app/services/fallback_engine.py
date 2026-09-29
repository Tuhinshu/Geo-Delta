"""
GeoDelta Deterministic CPU Fallback Engine (NFR-SAFE-001, NFR-SAFE-003)
Zero-crash demo invariant fallback executing adaptive Otsu spectral differencing
via OpenCV and NumPy when GPU acceleration or VLM workers are unavailable/timeout.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.services.footprint_calculator import GeodesicFootprintCalculator
from app.services.vectorizer import VectorizerService


def compute_spectral_difference(
    raster_t1: np.ndarray,
    raster_t2: np.ndarray,
    blur_ksize: int = 5,
) -> np.ndarray:
    """
    Computes absolute multi-band spectral difference between t1 and t2 with Gaussian smoothing.
    
    Args:
        raster_t1: Pre-event optical raster (H, W, C) or (H, W) or (C, H, W).
        raster_t2: Post-event optical raster.
        blur_ksize: Kernel size for Gaussian noise suppression.
        
    Returns:
        2D float32 array in [0.0, 1.0] representing normalized change intensity.
    """
    arr1 = np.asarray(raster_t1, dtype=np.float32)
    arr2 = np.asarray(raster_t2, dtype=np.float32)

    # Transpose if channel-first (C, H, W)
    if arr1.ndim == 3 and arr1.shape[0] in (1, 3, 4) and arr1.shape[0] < arr1.shape[1]:
        arr1 = np.transpose(arr1, (1, 2, 0))
    if arr2.ndim == 3 and arr2.shape[0] in (1, 3, 4) and arr2.shape[0] < arr2.shape[1]:
        arr2 = np.transpose(arr2, (1, 2, 0))

    # Mean absolute difference across spectral bands
    diff = np.mean(np.abs(arr2 - arr1), axis=-1) if arr1.ndim == 3 else np.abs(arr2 - arr1)

    # Gaussian blur to reduce high-frequency speckle noise
    if blur_ksize > 1 and blur_ksize % 2 == 1:
        diff = cv2.GaussianBlur(diff, (blur_ksize, blur_ksize), 0)

    # Normalize to [0.0, 1.0]
    min_val, max_val = float(diff.min()), float(diff.max())
    if max_val > min_val:
        norm_diff = (diff - min_val) / (max_val - min_val)
    else:
        norm_diff = np.zeros_like(diff, dtype=np.float32)

    return norm_diff.astype(np.float32)


def adaptive_otsu_threshold(
    diff_map: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Applies adaptive Otsu automatic binarization to the spectral difference map.
    
    Args:
        diff_map: Normalized 2D float32 difference map in [0.0, 1.0].
        
    Returns:
        Tuple of (probability_heatmap, optimal_otsu_threshold)
    """
    # Scale to uint8 [0, 255] for OpenCV Otsu
    diff_u8 = np.clip(diff_map * 255.0, 0, 255).astype(np.uint8)
    otsu_val, binary_mask = cv2.threshold(
        diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    normalized_thresh = float(otsu_val) / 255.0

    # Boost probabilities of regions exceeding the Otsu threshold
    prob_map = diff_map.copy()
    mask_bool = binary_mask > 0
    prob_map[mask_bool] = np.maximum(prob_map[mask_bool], normalized_thresh)
    prob_map[~mask_bool] = np.minimum(prob_map[~mask_bool], normalized_thresh * 0.5)

    return prob_map, normalized_thresh


class FallbackDifferencingEngine:
    """
    High-reliability CPU fallback change detection engine.
    Ensures zero-crash system resilience in air-gapped deployments.
    """

    def __init__(self, default_gsd: float = 10.0) -> None:
        self.default_gsd = default_gsd
        self.vectorizer = VectorizerService()
        self.footprint_calculator = GeodesicFootprintCalculator()

    def process_change_detection(
        self,
        raster_t1: np.ndarray,
        raster_t2: np.ndarray,
        affine_transform: Optional[Any] = None,
        query_text: str = "Tactical Change Detection (CPU Fallback)",
        confidence_threshold: float = 0.70,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end CPU spectral differencing, vector extraction, and metric calculation.
        """
        t0 = time.perf_counter()

        # 1. Compute smoothed multi-spectral difference
        diff_map = compute_spectral_difference(raster_t1, raster_t2)

        # 2. Adaptive Otsu thresholding
        prob_map, otsu_tau = adaptive_otsu_threshold(diff_map)

        # 3. Vectorize candidate regions
        vector_res = self.vectorizer.extract_polygons(
            prob_map=prob_map,
            transform=affine_transform,
            threshold=confidence_threshold,
            tactical_class=f"Detected Surface Anomaly ({query_text[:30]})"
        )

        # 4. Filter sub-tactical clutter and compute geodesic footprints
        features = self.footprint_calculator.filter_clutter_and_calculate_footprints(
            polygons=vector_res.features,
            min_area_sq_m=50.0
        )

        total_area = sum(f.area_sq_meters for f in features)
        total_latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "status": "COMPLETED",
            "fallback_mode": True,
            "engine": "OpenCV-Otsu-SpectralDifferencing-CPU",
            "otsu_threshold": otsu_tau,
            "total_features_detected": len(features),
            "total_area_altered_sq_m": total_area,
            "total_area_altered_ha": total_area / 10000.0,
            "polygons": [f.model_dump() for f in features],
            "total_latency_ms": round(total_latency_ms, 2),
        }
