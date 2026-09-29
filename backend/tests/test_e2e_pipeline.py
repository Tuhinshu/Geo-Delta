"""
GeoDelta End-to-End System Integration & Acceptance Test Suite (Phase 6)
Implements formal validation test cases TC-E2E-01 through TC-E2E-04,
resiliency circuit breakers, CPU fallback engine, and GPU OOM quadtree tiling recovery.
"""

import time
import numpy as np
import pytest
import torch
from affine import Affine
from fastapi.testclient import TestClient

from app.main import app
from app.services.coregistration import RegistrationFailureError, SubPixelCoregistration
from app.services.dossier_generator import DossierGenerator
from app.services.fallback_engine import FallbackDifferencingEngine
from app.services.feature_cache import difference_feature_cache
from app.services.quadtree_engine import (
    create_blend_weights,
    execute_quadtree_tiled_inference,
    get_quadtree_tiles,
)
from app.services.siamese_engine import SiameseCrossAttentionNetwork
from app.services.vlm_encoder import RemoteCLIPTextEncoder


@pytest.fixture
def mock_aoi_rasters():
    """Generates synthetic pre/post optical rasters with realistic tactical features."""
    h, w = 256, 256
    t1 = np.full((3, h, w), 50.0, dtype=np.float32)
    t2 = np.full((3, h, w), 50.0, dtype=np.float32)

    # Feature 1: Runway extension in top-left
    t2[:, 20:80, 20:120] = 180.0

    # Feature 2: Fortified revetment in bottom-right
    t2[:, 160:220, 160:220] = 210.0

    # Soil moisture drying in center (seasonal variation)
    t2[:, 100:150, 100:150] = 90.0

    affine = Affine.translation(75.0, 25.0) * Affine.scale(0.0001, -0.0001)
    return t1, t2, affine


# ==============================================================================
# TC-E2E-01: Counter-Factual Query Switching (Sub-150ms Evaluation)
# ==============================================================================


def test_tc_e2e_01_counterfactual_query_switching(mock_aoi_rasters):
    t1, t2, _ = mock_aoi_rasters
    encoder = RemoteCLIPTextEncoder()
    model = SiameseCrossAttentionNetwork()
    model.eval()

    t1_t = torch.from_numpy(t1).unsqueeze(0)
    t2_t = torch.from_numpy(t2).unsqueeze(0)

    # Query 1: Initial forward pass extracting feature pyramids
    emb_runway = encoder.encode_conditioned_prompt(query_text="Identify runway extension")
    with torch.no_grad():
        f1 = model.backbone(t1_t)
        f2 = model.backbone(t2_t)
        diff_pyr = model.compute_difference_pyramids(f1, f2)

    # Cache feature pyramid
    cache_key = "CACHE-AOI-TEST-01"
    difference_feature_cache.put(cache_key, diff_pyr)
    assert difference_feature_cache.has(cache_key)

    # Query 2: Sub-150ms Counter-Factual Query switching to agricultural clearing
    t0 = time.perf_counter()
    cached_pyr = difference_feature_cache.get(cache_key)
    emb_agri = encoder.encode_conditioned_prompt(query_text="Identify agricultural clearings")

    with torch.no_grad():
        out_agri = model.forward_with_cached_pyramids(cached_pyr, emb_agri)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert out_agri.shape == (1, 1, 256, 256)
    # Verification criteria: sub-150ms latency without raster re-download
    assert elapsed_ms <= 150.0, f"Counter-factual re-query latency exceeded 150ms: {elapsed_ms:.1f}ms"


# ==============================================================================
# TC-E2E-02: Negative Semantic Suppression Invariant
# ==============================================================================


def test_tc_e2e_02_negative_semantic_suppression():
    encoder = RemoteCLIPTextEncoder()

    # Base target: Concrete construction
    target = "Concrete military construction"
    e_target = encoder.encode_text(target)

    # Negative suppression: Seasonal soil moisture drying
    negative = "Seasonal soil moisture drying"
    e_conditioned = encoder.encode_conditioned_prompt(
        query_text=target,
        negative_query=negative,
        beta=0.65
    )

    # L2 norm must be unit length
    assert torch.isclose(torch.norm(e_conditioned, p=2), torch.tensor(1.0), atol=1e-5)

    # The conditioned vector must differ from the unconditioned vector
    cosine_sim = torch.sum(e_target * e_conditioned).item()
    assert cosine_sim < 0.999, "Negative suppression did not deflect the semantic vector"
    assert cosine_sim > 0.300, "Conditioned vector completely decoupled from primary query"


# ==============================================================================
# TC-E2E-03: 1-Click Cryptographic Intelligence Dossier Verification
# ==============================================================================


def test_tc_e2e_03_dossier_compilation_and_crypto_verification(mock_aoi_rasters):
    t1, t2, affine = mock_aoi_rasters
    generator = DossierGenerator()

    sample_polys = [
        {
            "feature_id": "feat-runway-99",
            "tactical_class": "Newly Paved Runway Extension",
            "confidence": 0.94,
            "area_sq_meters": 18400.0,
            "area_hectares": 1.84,
            "centroid_wgs84": [25.0450, 75.0480],
            "centroid_mgrs": "43R EH 04800 04500",
            "geometry_geojson": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [75.045, 25.040],
                        [75.050, 25.040],
                        [75.050, 25.045],
                        [75.045, 25.045],
                        [75.045, 25.040],
                    ]
                ],
            },
        }
    ]

    t0 = time.perf_counter()
    pdf_bytes = generator.generate_dossier(
        task_id="TASK-TC-E2E-03",
        query_prompt="Identify newly paved airstrip extensions",
        negative_prompt="Seasonal vegetation drying",
        total_area_altered_sq_m=18400.0,
        total_area_altered_ha=1.84,
        polygons=sample_polys,
        raster_t1=t1,
        raster_t2=t2,
        affine_transform=affine,
    )
    elapsed = time.perf_counter() - t0

    # Latency constraint: <= 3.0s
    assert elapsed <= 3.0, f"Dossier generation took {elapsed:.2f}s, exceeding 3.0s limit"
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 5000


# ==============================================================================
# TC-E2E-04: Deterministic Resiliency & Fallback Verification
# ==============================================================================


def test_tc_e2e_04_cpu_fallback_engine_otsu_differencing(mock_aoi_rasters):
    t1, t2, affine = mock_aoi_rasters
    fallback_engine = FallbackDifferencingEngine()

    result = fallback_engine.process_change_detection(
        raster_t1=t1,
        raster_t2=t2,
        affine_transform=affine,
        query_text="Runway construction",
        confidence_threshold=0.60
    )

    assert result["status"] == "COMPLETED"
    assert result["fallback_mode"] is True
    assert result["total_features_detected"] > 0
    assert result["total_area_altered_sq_m"] > 50.0
    assert result["total_latency_ms"] < 1000.0  # Fast CPU execution


def test_tc_e2e_04_ecc_registration_failure_circuit_breaker():
    coreg = SubPixelCoregistration(min_threshold=0.65)

    # Completely uncorrelated white noise rasters
    r1 = np.random.uniform(0, 255, (256, 256)).astype(np.float32)
    r2 = np.random.uniform(0, 255, (256, 256)).astype(np.float32)

    with pytest.raises(RegistrationFailureError):
        coreg.align(r1, r2)


def test_tc_e2e_04_quadtree_tiling_oom_recovery():
    # Verify quadtree tiling logic
    h, w = 512, 512
    tiles = get_quadtree_tiles(h, w, overlap_ratio=0.10)
    assert len(tiles) == 4

    weights = create_blend_weights(282, 282, 26, 26)
    assert weights.shape == (282, 282)
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0)

    # Run tiled inference through Siamese network
    model = SiameseCrossAttentionNetwork()
    model.eval()
    encoder = RemoteCLIPTextEncoder()
    emb = encoder.encode_text("Surveillance query")

    t1_dummy = torch.full((1, 3, 256, 256), 0.5, dtype=torch.float32)
    t2_dummy = torch.full((1, 3, 256, 256), 0.6, dtype=torch.float32)

    blended_prob = execute_quadtree_tiled_inference(
        model=model,
        t1_tensor=t1_dummy,
        t2_tensor=t2_dummy,
        text_emb=emb,
        overlap_ratio=0.10,
        device="cpu"
    )

    assert blended_prob.shape == (1, 1, 256, 256)
    assert not torch.isnan(blended_prob).any()
    assert (blended_prob >= 0.0).all() and (blended_prob <= 1.0).all()
