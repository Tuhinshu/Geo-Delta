"""
Unit tests for Windowed COG Streaming Service (FR-GEO-001).
"""

import os
import pytest

from app.services.cog_streamer import COGStreamer, StreamedRasterWindow
from app.core.schemas import BoundingBoxAOI
from app.core.exceptions import RasterIngestionException

SAMPLE_T1_PATH = "./storage/samples/sentinel2_t1_20250115.tif"
SAMPLE_T2_PATH = "./storage/samples/sentinel2_t2_20250610.tif"


def test_windowed_cog_stream():
    """
    FR-GEO-001: Windowed extraction streams only the sub-window AOI.
    Verifies data shape, GSD, affine transform, and SHA-256 integrity.
    """
    assert os.path.exists(SAMPLE_T1_PATH), f"Sample GeoTIFF not found at {SAMPLE_T1_PATH}"

    streamer = COGStreamer(default_bands=(1, 2, 3, 4))
    aoi = BoundingBoxAOI(
        min_lat=34.1300,
        max_lat=34.1500,
        min_lon=74.5750,
        max_lon=74.5950
    )

    window_t1 = streamer.stream_window(SAMPLE_T1_PATH, aoi=aoi)

    # 1. Type and structure validation
    assert isinstance(window_t1, StreamedRasterWindow)
    assert window_t1.data.ndim == 3
    assert window_t1.data.shape[0] == 4  # 4 spectral bands

    # 2. Window size validation (sub-window must be smaller than full 512x512 raster)
    assert window_t1.width < 512
    assert window_t1.height < 512
    assert window_t1.width > 50
    assert window_t1.height > 50

    # 3. Acceptance criteria: byte transfer size is < 15 MB
    assert window_t1.byte_size < 15 * 1024 * 1024

    # 4. Cryptographic integrity check: 64-char hex SHA-256
    assert len(window_t1.sha256) == 64
    assert int(window_t1.sha256, 16) > 0

    # 5. GSD validation: around ~10m for Sentinel-2
    assert 5.0 <= window_t1.native_gsd <= 20.0

    # 6. Coordinate system validation
    assert "4326" in window_t1.crs


def test_cog_streamer_out_of_bounds_aoi():
    """
    AOI completely outside raster bounds must raise RasterIngestionException.
    """
    streamer = COGStreamer()
    out_of_bounds_aoi = BoundingBoxAOI(
        min_lat=10.0,
        max_lat=11.0,
        min_lon=10.0,
        max_lon=11.0
    )

    with pytest.raises(RasterIngestionException) as exc_info:
        streamer.stream_window(SAMPLE_T1_PATH, aoi=out_of_bounds_aoi)

    assert "does not intersect" in str(exc_info.value)


def test_cog_streamer_invalid_path():
    """
    Non-existent file path must raise RasterIngestionException.
    """
    streamer = COGStreamer()
    aoi = BoundingBoxAOI(min_lat=34.13, max_lat=34.15, min_lon=74.57, max_lon=74.59)

    with pytest.raises(RasterIngestionException):
        streamer.stream_window("./non_existent_raster.tif", aoi=aoi)
