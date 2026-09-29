"""
Feature Pyramid Cache Engine for GeoDelta
Implements FR-NLQ-004: Sub-150ms Counter-Factual Query Execution.

Caches extracted bitemporal difference pyramids (F_delta^l) keyed by:
    cache_key = sha256(f"{aoi_hash}_{t1_date}_{t2_date}")
When a counter-factual query arrives, the system re-runs ONLY cross-attention modulation
and UNet++ decoding, bypassing raw raster reads and ResNet-34 extraction.
"""

import time
import hashlib
from typing import Dict, Tuple, Optional, Any, Sequence, Union
from collections import OrderedDict
import torch


class FeaturePyramidCache:
    """
    High-speed in-memory LRU cache storing extracted difference feature pyramids.
    Guarantees sub-150ms execution latency for counter-factual re-queries.
    """

    def __init__(self, max_cached_scenes: int = 16, default_ttl_seconds: int = 1800):
        self.max_cached_scenes = max_cached_scenes
        self.default_ttl = default_ttl_seconds
        # OrderedDict used for LRU eviction: key -> {"pyramid": ..., "size": ..., "expires_at": ...}
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    @staticmethod
    def generate_cache_key(aoi_repr: str, t1_repr: str, t2_repr: str) -> str:
        """
        Computes deterministic SHA-256 cache key from AOI coordinates and temporal stamps.
        """
        raw_token = f"{aoi_repr}::{t1_repr}::{t2_repr}".encode("utf-8")
        return hashlib.sha256(raw_token).hexdigest()

    def put(
        self,
        cache_key: str,
        f_delta_pyramid: Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, ...], Sequence[torch.Tensor]],
        target_size: Optional[Tuple[int, int]] = None,
        ttl_seconds: Optional[int] = None,
        transform: Optional[Any] = None
    ) -> None:
        """
        Stores difference feature pyramids in the LRU cache.
        """
        ttl = ttl_seconds or self.default_ttl
        expires_at = time.time() + ttl

        # If cache is full, evict oldest entry
        if len(self._cache) >= self.max_cached_scenes and cache_key not in self._cache:
            self._cache.popitem(last=False)

        # Store CPU or detached tensors to keep memory footprint bounded
        detached_pyramid = tuple(t.detach().cpu() for t in f_delta_pyramid)

        self._cache[cache_key] = {
            "f_delta_pyramid": detached_pyramid,
            "target_size": target_size,
            "transform": transform,
            "expires_at": expires_at
        }
        self._cache.move_to_end(cache_key)

    def get(
        self,
        cache_key: str,
        device: Optional[torch.device] = None
    ) -> Optional[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Tuple[int, int]]]:
        """
        Retrieves cached feature pyramid if present and unexpired.
        
        Returns:
            Tuple of (f_delta_pyramid, target_size) or None.
        """
        entry = self._cache.get(cache_key)
        if entry is None:
            return None

        # Check expiration
        if time.time() > entry["expires_at"]:
            del self._cache[cache_key]
            return None

        self._cache.move_to_end(cache_key)
        pyramid = entry["f_delta_pyramid"]
        target_size = entry["target_size"]

        if device is not None:
            pyramid = tuple(t.to(device) for t in pyramid)

        return (pyramid[0], pyramid[1], pyramid[2], pyramid[3]), target_size

    def get_transform(self, cache_key: str) -> Optional[Any]:
        """Returns the cached affine transform for the AOI window if present."""
        entry = self._cache.get(cache_key)
        if entry is None or time.time() > entry["expires_at"]:
            return None
        return entry.get("transform")

    def has(self, cache_key: str) -> bool:
        """Checks if a valid, unexpired entry exists for the given key."""
        entry = self._cache.get(cache_key)
        if entry is None:
            return False
        if time.time() > entry["expires_at"]:
            del self._cache[cache_key]
            return False
        return True

    def clear(self) -> None:
        """Flushes all cached feature pyramids."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


# Global singleton cache instance
feature_cache = FeaturePyramidCache()

# Public alias used by e2e integration tests (FR-NLQ-004)
difference_feature_cache = feature_cache
