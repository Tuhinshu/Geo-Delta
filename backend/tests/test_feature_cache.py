"""
Unit & Benchmarking tests for Feature Pyramid Cache & Counter-Factual Latency (FR-NLQ-004).
"""

import time
import pytest
import torch
import torch.nn.functional as F

from app.services.feature_cache import FeaturePyramidCache
from app.services.siamese_engine import SiameseCrossAttentionEngine
from app.services.vlm_encoder import RemoteCLIPTextEncoder


def test_feature_cache_put_get():
    """
    FR-NLQ-004: Tests caching and retrieval of difference feature pyramids.
    """
    cache = FeaturePyramidCache(max_cached_scenes=4, default_ttl_seconds=60)
    key = cache.generate_cache_key("aoi_sector_4", "2025-01-15", "2025-06-10")

    # Mock pyramid
    p1 = torch.zeros(1, 64, 32, 32)
    p2 = torch.zeros(1, 128, 16, 16)
    p3 = torch.zeros(1, 256, 8, 8)
    p4 = torch.zeros(1, 512, 4, 4)
    pyramid = (p1, p2, p3, p4)

    cache.put(key, pyramid, target_size=(128, 128))

    assert cache.has(key) is True
    cached_val = cache.get(key)
    assert cached_val is not None
    retrieved, target_size = cached_val
    assert len(retrieved) == 4
    assert retrieved[0].shape == (1, 64, 32, 32)
    assert target_size == (128, 128)


def test_feature_cache_lru_eviction():
    """
    Verifies oldest entry is evicted when max_cached_scenes is exceeded.
    """
    cache = FeaturePyramidCache(max_cached_scenes=2)
    k1 = "key_1"
    k2 = "key_2"
    k3 = "key_3"

    mock_pyramid = (torch.zeros(1, 64, 8, 8), torch.zeros(1, 128, 4, 4), torch.zeros(1, 256, 2, 2), torch.zeros(1, 512, 1, 1))

    cache.put(k1, mock_pyramid, (32, 32))
    cache.put(k2, mock_pyramid, (32, 32))
    assert cache.has(k1) and cache.has(k2)

    # Insert 3rd item -> k1 must be evicted
    cache.put(k3, mock_pyramid, (32, 32))
    assert not cache.has(k1)
    assert cache.has(k2)
    assert cache.has(k3)


def test_counterfactual_latency_sub_150ms():
    """
    Acceptance Criteria (Section 2.4):
    Counter-factual query re-evaluation runs in < 150 ms without accessing raw rasters.
    """
    model = SiameseCrossAttentionEngine(in_channels=4, embed_dim=512)
    model.eval()
    vlm = RemoteCLIPTextEncoder()
    cache = FeaturePyramidCache()

    h, w = 256, 256
    i_t1 = torch.rand(1, 4, h, w)
    i_t2 = torch.rand(1, 4, h, w)

    key = cache.generate_cache_key("test_aoi", "20250115", "20250610")

    with torch.no_grad():
        # Pre-execution: initial query and feature caching
        _, _, f_delta_pyramid = model.extract_feature_pyramids(i_t1, i_t2)
        cache.put(key, f_delta_pyramid, target_size=(h, w))

        # Counter-factual query re-evaluation: "Show unpaved roadway grading"
        t0 = time.perf_counter()

        cached_entry = cache.get(key)
        assert cached_entry is not None
        cached_pyramid, target_size = cached_entry

        # Generate new conditioning vector
        e_star_new = vlm.encode_conditioned_prompt("Show unpaved roadway grading", negative_query="seasonal agriculture")

        # Fast forward pass (cross-attention + decoder only)
        prob_map = model.forward_from_features(cached_pyramid, e_star_new, target_size=target_size)

        latency_ms = (time.perf_counter() - t0) * 1000.0

    assert prob_map.shape == (1, 1, h, w)
    # Verification: must execute in under 150 ms
    assert latency_ms < 150.0, f"Counter-factual re-query took {latency_ms:.2f}ms, expected < 150ms"
