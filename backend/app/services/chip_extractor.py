"""
GeoDelta High-Resolution Evidence Chip Extractor & Compositor (FR-EXP-001)
Extracts localized optical chips (Panel A: t1, Panel B: t2) and renders antialiased
tactical vector change overlays (Panel C: Change Overlay) for military dossiers.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


def ensure_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """
    Ensures input image array is 3-channel RGB in uint8 range [0, 255].
    Handles single-channel grayscale, multi-spectral (takes first 3 bands), and float normalization.
    """
    arr = np.asarray(img)

    # If channel-first (C, H, W)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[1]:
        arr = np.transpose(arr, (1, 2, 0))

    # Single-channel grayscale (H, W) or (H, W, 1)
    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
        if arr.ndim == 3:
            arr = arr.squeeze(-1)
        # Normalize to uint8
        if arr.dtype != np.uint8:
            min_val, max_val = float(arr.min()), float(arr.max())
            if max_val > min_val:
                arr = np.clip((arr - min_val) / (max_val - min_val) * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.zeros(arr.shape, dtype=np.uint8)
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        return arr

    # If RGB or RGBA
    if arr.ndim == 3:
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            min_val, max_val = float(arr.min()), float(arr.max())
            if max_val > min_val:
                arr = np.clip((arr - min_val) / (max_val - min_val) * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.zeros(arr.shape, dtype=np.uint8)
        return arr

    raise ValueError(f"Unsupported array shape for optical chip: {arr.shape}")


def extract_localized_chip(
    raster: np.ndarray,
    center_rc: Optional[Tuple[int, int]] = None,
    chip_size: int = 512,
) -> np.ndarray:
    """
    Extracts a square optical chip of size (chip_size x chip_size).
    Performs high-quality optical resizing so the imagery completely fills the chip panel
    without empty black borders or void padding.
    """
    rgb = ensure_uint8_rgb(raster)
    h, w, c = rgb.shape

    # If center_rc is provided and image is significantly larger than chip_size, crop and resize
    if center_rc is not None and (h > chip_size or w > chip_size):
        cr, cc = center_rc
        half = min(h, w) // 2
        r_min, r_max = max(0, cr - half), min(h, cr + half)
        c_min, c_max = max(0, cc - half), min(w, cc + half)
        crop = rgb[r_min:r_max, c_min:c_max]
        if crop.size > 0:
            return cv2.resize(crop, (chip_size, chip_size), interpolation=cv2.INTER_AREA)

    # Standard clean full optical scene resize to chip_size x chip_size
    return cv2.resize(rgb, (chip_size, chip_size), interpolation=cv2.INTER_AREA)


def render_vector_overlay_chip(
    base_raster_t2: np.ndarray,
    polygons: List[Dict[str, Any]],
    affine_transform: Optional[Any] = None,
    center_rc: Optional[Tuple[int, int]] = None,
    chip_size: int = 512,
    outline_color_bgr: Tuple[int, int, int] = (0, 68, 239),  # Crimson in BGR
    glow_color_bgr: Tuple[int, int, int] = (212, 182, 6),    # Cyan in BGR
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Renders vector change polygons on top of t2 base imagery with semi-transparent
    tactical fill and high-contrast antialiased borders.
    """
    chip_t2 = extract_localized_chip(base_raster_t2, center_rc=center_rc, chip_size=chip_size)
    overlay = chip_t2.copy()
    h_chip, w_chip = chip_t2.shape[:2]

    scale_x = w_chip / 512.0
    scale_y = h_chip / 512.0

    # Collect all polygon rings and geographic points
    all_rings = []
    all_geo_pts = []
    for feat in polygons:
        geom = feat.get("geometry_geojson", {})
        coords = geom.get("coordinates", [])
        gtype = geom.get("type", "Polygon")

        poly_rings = []
        if gtype == "Polygon" and coords:
            poly_rings = coords
        elif gtype == "MultiPolygon" and coords:
            for p in coords:
                poly_rings.extend(p)

        for ring in poly_rings:
            valid_ring = [pt for pt in ring if len(pt) >= 2]
            if len(valid_ring) >= 3:
                all_rings.append(valid_ring)
                for pt in valid_ring:
                    if abs(pt[0]) <= 180 and abs(pt[1]) <= 90:
                        all_geo_pts.append(pt)

    # Determine coordinate mapping mode
    auto_norm = False
    min_lon, max_lon, min_lat, max_lat = 0.0, 0.0, 0.0, 0.0
    if affine_transform is None and all_geo_pts:
        all_lons = [p[0] for p in all_geo_pts]
        all_lats = [p[1] for p in all_geo_pts]
        min_lon, max_lon = min(all_lons), max(all_lons)
        min_lat, max_lat = min(all_lats), max(all_lats)

        # Check if they fit within default nominal grid [75.04, 75.12] and [24.98, 25.06]
        in_default_grid = (75.040 <= min_lon <= 75.120 and 75.040 <= max_lon <= 75.120 and
                           24.980 <= min_lat <= 25.060 and 24.980 <= max_lat <= 25.060)
        if not in_default_grid:
            # Enable auto-normalization with 20% margin
            auto_norm = True
            pad_lon = max((max_lon - min_lon) * 0.20, 0.001)
            pad_lat = max((max_lat - min_lat) * 0.20, 0.001)
            min_lon -= pad_lon
            max_lon += pad_lon
            min_lat -= pad_lat
            max_lat += pad_lat

    for ring in all_rings:
        pts = []
        for pt in ring:
            if affine_transform is not None:
                lon, lat = pt[0], pt[1]
                inv_transform = ~affine_transform
                c_px, r_px = inv_transform * (lon, lat)
            elif auto_norm:
                c_px = ((pt[0] - min_lon) / (max_lon - min_lon)) * 512.0
                r_px = ((max_lat - pt[1]) / (max_lat - min_lat)) * 512.0
            elif abs(pt[0]) <= 180 and abs(pt[1]) <= 90:
                c_px = (pt[0] - 75.050) / 0.0001
                r_px = (25.045 - pt[1]) / 0.0001
            else:
                c_px, r_px = pt[0], pt[1]

            chip_c = int(round(c_px * scale_x))
            chip_r = int(round(r_px * scale_y))
            chip_c = max(0, min(w_chip - 1, chip_c))
            chip_r = max(0, min(h_chip - 1, chip_r))
            pts.append([chip_c, chip_r])

        if len(pts) >= 3:
            cnt = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            # Fill polygon on overlay with tactical crimson
            cv2.fillPoly(overlay, [cnt], (239, 68, 68))
            # Draw high-contrast outer boundary with antialiasing
            cv2.polylines(overlay, [cnt], isClosed=True, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.polylines(overlay, [cnt], isClosed=True, color=(239, 68, 68), thickness=1, lineType=cv2.LINE_AA)

    # Alpha blend overlay over base image
    blended = cv2.addWeighted(overlay, alpha, chip_t2, 1.0 - alpha, 0)
    return blended


def raster_to_png_bytes(raster: np.ndarray) -> bytes:
    """Encodes a uint8 RGB image array into in-memory PNG bytes."""
    rgb = ensure_uint8_rgb(raster)
    pil_img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def generate_dossier_chip_trio(
    raster_t1: np.ndarray,
    raster_t2: np.ndarray,
    polygons: List[Dict[str, Any]],
    affine_transform: Optional[Any] = None,
    center_rc: Optional[Tuple[int, int]] = None,
    chip_size: int = 512,
) -> Tuple[bytes, bytes, bytes]:
    """
    Extracts and returns in-memory PNG bytes for:
    - Panel A: Pre-Event (t1) Optical Chip
    - Panel B: Post-Event (t2) Optical Chip
    - Panel C: Vector Change Overlay Chip
    """
    chip_t1 = extract_localized_chip(raster_t1, center_rc=center_rc, chip_size=chip_size)
    chip_t2 = extract_localized_chip(raster_t2, center_rc=center_rc, chip_size=chip_size)
    chip_overlay = render_vector_overlay_chip(
        raster_t2,
        polygons=polygons,
        affine_transform=affine_transform,
        center_rc=center_rc,
        chip_size=chip_size,
    )

    bytes_t1 = raster_to_png_bytes(chip_t1)
    bytes_t2 = raster_to_png_bytes(chip_t2)
    bytes_overlay = raster_to_png_bytes(chip_overlay)

    return bytes_t1, bytes_t2, bytes_overlay
