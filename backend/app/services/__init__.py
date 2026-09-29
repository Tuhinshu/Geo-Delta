"""
GeoDelta Geospatial Services Package
Enforces remote-sensing physical laws, sub-pixel coregistration, COG streaming,
RemoteCLIP semantic projection, and Siamese Cross-Attention deep learning inference.
"""

from app.services.normalizer import RadiometricNormalizer
from app.services.coregistration import SubPixelCoregistration
from app.services.cog_streamer import COGStreamer, StreamedRasterWindow
from app.services.vlm_encoder import RemoteCLIPTextEncoder
from app.services.siamese_engine import SiameseCrossAttentionEngine
from app.services.feature_cache import FeaturePyramidCache, feature_cache

__all__ = [
    "RadiometricNormalizer",
    "SubPixelCoregistration",
    "COGStreamer",
    "StreamedRasterWindow",
    "RemoteCLIPTextEncoder",
    "SiameseCrossAttentionEngine",
    "FeaturePyramidCache",
    "feature_cache"
]
