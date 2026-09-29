"""
Custom Exception Classes and Circuit Breakers for GeoDelta GEOINT Platform
Enforces automated failure handling and operational safety rules.
"""

from typing import Optional, Dict, Any


class GeoDeltaBaseException(Exception):
    """Base exception for all GeoDelta domain errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class CloudCoverExceededException(GeoDeltaBaseException):
    """Tripped when optical cloud cover exceeds the 35% safety threshold (NFR-SAFE-001)."""
    def __init__(self, cloud_ratio: float, threshold: float = 0.35):
        msg = (
            f"Optical cloud occlusion ({cloud_ratio * 100:.1f}%) exceeds the "
            f"maximum permitted threshold ({threshold * 100:.1f}%). "
            f"Optical analysis aborted. Switch to Sentinel-1 SAR imagery."
        )
        super().__init__(msg, {"cloud_ratio": cloud_ratio, "threshold": threshold})


class RegistrationFailureException(GeoDeltaBaseException):
    """Tripped when sub-pixel ECC coregistration score falls below 0.65 (NFR-SAFE-002)."""
    def __init__(self, ecc_score: float, min_threshold: float = 0.65):
        msg = (
            f"Sub-pixel image coregistration failed with ECC score {ecc_score:.3f} "
            f"(threshold: {min_threshold:.3f}). "
            f"Processing aborted to prevent false-positive terrain edge lines."
        )
        super().__init__(msg, {"ecc_score": ecc_score, "threshold": min_threshold})


class SubNyquistResolutionException(GeoDeltaBaseException):
    """Tripped when target entity dimension violates the Nyquist-Shannon resolution law (Law 1)."""
    def __init__(self, entity_name: str, entity_dimension: float, native_gsd: float):
        msg = (
            f"Requested entity '{entity_name}' (approx dimension {entity_dimension:.1f}m) "
            f"is below the Nyquist-Shannon limit (2 x GSD = {2 * native_gsd:.1f}m) "
            f"for native sensor resolution {native_gsd:.1f}m GSD. Query rejected."
        )
        super().__init__(msg, {
            "entity": entity_name,
            "dimension": entity_dimension,
            "native_gsd": native_gsd,
            "nyquist_min": 2 * native_gsd
        })


class GPUOutOfMemoryException(GeoDeltaBaseException):
    """Tripped when GPU VRAM is exhausted during inference (NFR-SAFE-003)."""
    def __init__(self, aoi_dimensions: tuple):
        msg = (
            f"CUDA Out-Of-Memory detected on raster dimensions {aoi_dimensions}. "
            f"Activating dynamic 4-way overlapping quadtree tiling."
        )
        super().__init__(msg, {"dimensions": aoi_dimensions})


class RasterIngestionException(GeoDeltaBaseException):
    """Tripped when COG byte-range streaming or metadata extraction fails."""
    def __init__(self, raster_uri: str, reason: str):
        msg = f"Failed to ingest COG raster from '{raster_uri}': {reason}"
        super().__init__(msg, {"uri": raster_uri, "reason": reason})


# Invariant aliases
SpatialResolutionException = SubNyquistResolutionException
