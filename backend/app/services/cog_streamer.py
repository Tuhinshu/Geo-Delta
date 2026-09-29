"""
Cloud-Optimized GeoTIFF (COG) Windowed Streaming Service for GeoDelta
Implements FR-GEO-001: Sub-window byte-range extraction without full scene downloads.

Enforces:
    - Windowed rasterio reads over BoundingBoxAOI
    - Reprojection of geographic AOI to native raster CRS if needed
    - Calculation of native GSD (meters/pixel) for Nyquist-Shannon verification
    - SHA-256 cryptographic checksum calculation for chain of custody
"""

import os
import math
import hashlib
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window, transform as window_transform
from rasterio.warp import transform_bounds
import affine

from app.core.schemas import BoundingBoxAOI
from app.core.exceptions import RasterIngestionException


@dataclass
class StreamedRasterWindow:
    """Represents an extracted sub-window from a Cloud-Optimized GeoTIFF."""
    data: np.ndarray                     # (C, H, W) numpy array
    transform: affine.Affine             # Affine transform for the extracted sub-window
    crs: str                             # Dataset Coordinate Reference System (e.g. "EPSG:4326")
    bounds: Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat) in EPSG:4326
    native_gsd: float                    # Ground Sample Distance in meters/pixel
    sha256: str                          # SHA-256 cryptographic checksum of window bytes
    byte_size: int                       # Raw byte payload size
    width: int                           # Window width in pixels
    height: int                          # Window height in pixels


class COGStreamer:
    """
    High-throughput windowed streamer for Cloud-Optimized GeoTIFFs (COGs).
    Supports local filesystem paths, S3 URIs, and HTTP/HTTPS range endpoints.
    """

    def __init__(self, default_bands: Sequence[int] = (1, 2, 3, 4)):
        self.default_bands = list(default_bands)

    @staticmethod
    def _calculate_gsd(src: rasterio.DatasetReader, mean_lat: float = 0.0) -> float:
        """
        Calculates the native Ground Sample Distance (GSD) in meters.
        """
        x_res = abs(src.res[0])
        y_res = abs(src.res[1])

        # If CRS is geographic (degrees), convert to approximate ground meters
        if src.crs and src.crs.is_geographic:
            lat_rad = math.radians(mean_lat)
            meters_per_deg_lat = 111132.954 - 559.822 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
            meters_per_deg_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)
            gsd_x = x_res * meters_per_deg_lon
            gsd_y = y_res * meters_per_deg_lat
            return float(math.sqrt(gsd_x * gsd_y))

        # Projected CRS (already in meters or linear units)
        return float(math.sqrt(x_res * y_res))

    def stream_window(
        self,
        raster_uri: str,
        aoi: BoundingBoxAOI,
        bands: Optional[Sequence[int]] = None
    ) -> StreamedRasterWindow:
        """
        Streams a discrete sub-window matching the geographic AOI from the target COG.

        Args:
            raster_uri: Local file path, /vsicurl/ HTTP URL, or S3 path.
            aoi: Geographic bounding box (WGS 84 / EPSG:4326).
            bands: 1-indexed list of spectral bands to read (default: [1, 2, 3, 4]).

        Returns:
            StreamedRasterWindow dataclass containing pixel tensor, affine transform, and SHA-256.

        Raises:
            RasterIngestionException: If the URI is invalid, unreadable, or doesn't intersect AOI.
        """
        read_bands = list(bands) if bands is not None else self.default_bands

        try:
            with rasterio.open(raster_uri) as src:
                # 1. Transform AOI bounds (EPSG:4326) into raster CRS if necessary
                min_lon, min_lat, max_lon, max_lat = aoi.min_lon, aoi.min_lat, aoi.max_lon, aoi.max_lat

                if src.crs and not src.crs.is_geographic:
                    try:
                        left, bottom, right, top = transform_bounds(
                            "EPSG:4326", src.crs, min_lon, min_lat, max_lon, max_lat
                        )
                    except Exception as err:
                        raise RasterIngestionException(raster_uri, f"Coordinate transformation error: {err}")
                else:
                    left, bottom, right, top = min_lon, min_lat, max_lon, max_lat

                # 2. Compute pixel window
                raw_window = from_bounds(left, bottom, right, top, transform=src.transform)
                
                # Round to integer pixels
                int_window = raw_window.round_offsets().round_lengths()

                # Clamp window to raster bounds
                raster_bounds_win = Window(0, 0, src.width, src.height)
                try:
                    clamped_window = int_window.intersection(raster_bounds_win)
                except Exception:
                    raise RasterIngestionException(
                        raster_uri,
                        f"Requested AOI ({min_lat}, {min_lon}, {max_lat}, {max_lon}) does not intersect raster coverage."
                    )

                if clamped_window.width <= 0 or clamped_window.height <= 0:
                    raise RasterIngestionException(
                        raster_uri,
                        f"Requested AOI ({min_lat}, {min_lon}, {max_lat}, {max_lon}) does not intersect raster coverage."
                    )

                # 3. Read windowed pixel data (C, H, W)
                available_bands = [b for b in read_bands if b <= src.count]
                if not available_bands:
                    raise RasterIngestionException(
                        raster_uri,
                        f"Requested bands {read_bands} exceed available band count ({src.count})."
                    )

                data = src.read(available_bands, window=clamped_window)

                # 4. Compute affine transform for window
                win_transform_affine = window_transform(clamped_window, src.transform)

                # 5. Calculate GSD
                mean_lat = (aoi.min_lat + aoi.max_lat) / 2.0
                gsd = self._calculate_gsd(src, mean_lat=mean_lat)

                # 6. Compute SHA-256 hash of extracted window bytes
                byte_buffer = data.tobytes()
                sha256_hash = hashlib.sha256(byte_buffer).hexdigest()

                crs_str = str(src.crs) if src.crs else "EPSG:4326"

                return StreamedRasterWindow(
                    data=data,
                    transform=win_transform_affine,
                    crs=crs_str,
                    bounds=(min_lon, min_lat, max_lon, max_lat),
                    native_gsd=gsd,
                    sha256=sha256_hash,
                    byte_size=len(byte_buffer),
                    width=int(clamped_window.width),
                    height=int(clamped_window.height)
                )

        except RasterIngestionException:
            raise
        except Exception as exc:
            raise RasterIngestionException(raster_uri, f"I/O read failure: {exc}")
