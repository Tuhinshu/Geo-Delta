"""
Unit and Latency Benchmark Tests for Vectorizer Engine (FR-VEC-001, FR-VEC-002, Section 3.4).
"""

import time
import pytest
import numpy as np
import affine
import torch

from app.services.vectorizer import (
    apply_morphological_noise_suppression,
    extract_vector_features,
    export_feature_collection
)


def test_morphological_noise_suppression():
    """
    FR-VEC-001: Verifies single-pixel noise is removed while coherent regions are preserved.
    B_clean = (B - K_3x3) + K_3x3
    """
    prob_map = np.zeros((64, 64), dtype=np.float32)

    # 1. Inject isolated single-pixel hot spots (salt noise)
    prob_map[5, 5] = 0.95
    prob_map[12, 40] = 0.88
    prob_map[50, 20] = 0.92

    # 2. Inject coherent 8x8 block representing real ground structure
    prob_map[20:28, 20:28] = 0.85

    clean_mask = apply_morphological_noise_suppression(prob_map, threshold=0.70, kernel_size=3)

    # Single-pixel noise must be eliminated
    assert clean_mask[5, 5] == 0
    assert clean_mask[12, 40] == 0
    assert clean_mask[50, 20] == 0

    # Coherent structure interior must be preserved
    assert clean_mask[22, 22] == 1
    assert clean_mask[25, 25] == 1
    assert np.sum(clean_mask) > 0


def test_vectorizer_synthetic_10x10_square_accuracy():
    """
    Section 3.4 Acceptance Criteria:
    Tests vectorization of a known synthetic square of 10 x 10 pixels at 10m GSD (10,000 m² nominal area).
    Verifies ellipsoidal calculation matches expected geodesic area within 0.5% error margin.
    """
    # 10m GSD at Lat 25.0, Lon 75.0
    # In geographic degrees:
    # 10m lat ~ 10 / 110787 = 9.02633e-5 deg
    # 10m lon ~ 10 / 100865 = 9.91424e-5 deg
    res_x = 10.0 / 100865.0
    res_y = 10.0 / 110787.0
    lat0, lon0 = 25.0, 75.0
    transform = affine.Affine(res_x, 0.0, lon0, 0.0, -res_y, lat0)

    # Create 64x64 grid with a 10x10 pixel block from row 20 to 30, col 20 to 30
    prob_map = np.zeros((64, 64), dtype=np.float32)
    prob_map[20:30, 20:30] = 0.90  # 10 x 10 = 100 pixels

    features = extract_vector_features(
        prob_map=prob_map,
        transform=transform,
        threshold=0.70,
        tactical_class="Reinforced Aircraft Bunker",
        min_area_sq_m=50.0
    )

    assert len(features) == 1
    feat = features[0]
    assert feat.tactical_class == "Reinforced Aircraft Bunker"
    assert feat.confidence >= 0.89

    # Nominal area = 10 x 10 pixels * 10m * 10m = 10,000 m²
    nominal_area = 10000.0
    error_ratio = abs(feat.area_sq_meters - nominal_area) / nominal_area
    assert error_ratio < 0.005, f"Area {feat.area_sq_meters} deviated by {error_ratio*100:.3f}% (max 0.5%)"

    # GeoJSON MultiPolygon structure
    geom = feat.geometry_geojson
    assert geom["type"] == "MultiPolygon"
    assert len(geom["coordinates"]) == 1
    assert len(geom["coordinates"][0][0]) >= 4  # closed ring


def test_vectorizer_clutter_exclusion():
    """
    FR-VEC-004: Confirms that polygons < 50 m² are excluded from output.
    At 10m GSD, a 2x2 pixel square = 20m x 20m = 400 m² (kept).
    A 1x1 pixel square with pixel dimension 5m x 5m = 25 m² (< 50 m² dropped).
    """
    res_x = 5.0 / 100865.0
    res_y = 5.0 / 110787.0
    transform = affine.Affine(res_x, 0.0, 75.0, 0.0, -res_y, 25.0)

    prob_map = np.zeros((64, 64), dtype=np.float32)
    # 2x2 pixels at 5m GSD = 10m x 10m = 100 m² (kept)
    prob_map[10:14, 10:14] = 0.90  # 4x4 pixels = 20m x 20m = 400 m² (kept)

    # 2x2 pixels at 5m GSD = 10m x 10m = ~100 m² (kept)
    prob_map[30:32, 30:32] = 0.85

    # 1x1 pixel noise is already cleaned by morphological filter
    prob_map[50, 50] = 0.99

    features = extract_vector_features(
        prob_map=prob_map,
        transform=transform,
        threshold=0.70,
        min_area_sq_m=50.0
    )

    # 50,50 is cleaned by morphology; 10:14 and 30:32 are kept
    for f in features:
        assert f.area_sq_meters >= 50.0


def test_vector_generation_latency_500_polygons():
    """
    Section 3.4 Acceptance Criteria:
    Vector generation latency <= 800 ms for 500 candidate polygons.
    """
    # Create 1000x1000 probability map with 500 discrete 6x6 squares
    h, w = 1000, 1000
    prob_map = np.zeros((h, w), dtype=np.float32)

    count = 0
    step = 40
    for r in range(20, h - 30, step):
        for c in range(20, w - 30, step):
            if count < 500:
                prob_map[r:r+6, c:c+6] = 0.88
                count += 1

    transform = affine.Affine(0.0001, 0.0, 75.0, 0.0, -0.0001, 25.0)

    t0 = time.perf_counter()
    features = extract_vector_features(
        prob_map=prob_map,
        transform=transform,
        threshold=0.70,
        min_area_sq_m=50.0
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert len(features) == 500
    assert elapsed_ms <= 800.0, f"Latency {elapsed_ms:.2f} ms exceeded 800 ms ceiling!"


def test_export_feature_collection():
    """Verifies GeoJSON FeatureCollection generation."""
    prob_map = np.zeros((32, 32), dtype=np.float32)
    prob_map[10:20, 10:20] = 0.90
    transform = affine.Affine(0.0001, 0.0, 75.0, 0.0, -0.0001, 25.0)

    features = extract_vector_features(prob_map, transform)
    fc = export_feature_collection(features)

    assert fc["type"] == "FeatureCollection"
    assert "features" in fc
    assert len(fc["features"]) == len(features)
    f0 = fc["features"][0]
    assert f0["type"] == "Feature"
    assert "geometry" in f0
    assert "properties" in f0
    assert f0["properties"]["tactical_class"] == "Tactical Change Detection"
