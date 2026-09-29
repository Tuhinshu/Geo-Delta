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
    Extracts a square optical chip of size (chip_size x chip_size) centered at center_rc (row, col).
    If center_rc is None, uses center of raster. Handles boundary edge padding safely.
    """
    rgb = ensure_uint8_rgb(raster)
    h, w, c = rgb.shape

    if center_rc is None:
        cr, cc = h // 2, w // 2
    else:
        cr, cc = center_rc

    half = chip_size // 2
    r_min, r_max = cr - half, cr + half
    c_min, c_max = cc - half, cc + half

    # Initialize padded output with dark void fill
    chip = np.zeros((chip_size, chip_size, c), dtype=np.uint8)

    # Overlapping source coordinates
    src_r1 = max(0, r_min)
    src_r2 = min(h, r_max)
    src_c1 = max(0, c_min)
    src_c2 = min(w, c_max)

    # Destination coordinates inside chip
    dst_r1 = src_r1 - r_min
    dst_r2 = dst_r1 + (src_r2 - src_r1)
    dst_c1 = src_c1 - c_min
    dst_c2 = dst_c1 + (src_c2 - src_c1)

    if (src_r2 > src_r1) and (src_c2 > src_c1):
        chip[dst_r1:dst_r2, dst_c1:dst_c2] = rgb[src_r1:src_r2, src_c1:src_c2]

    return chip


def render_vector_overlay_chip(
    base_raster_t2: np.ndarray,
    polygons: List[Dict[str, Any]],
    affine_transform: Optional[Any] = None,
    center_rc: Optional[Tuple[int, int]] = None,
    chip_size: int = 512,
    outline_color_bgr: Tuple[int, int, int] = (0, 68, 239),  # Crimson in BGR
    glow_color_bgr: Tuple[int, int, int] = (212, 182, 6),    # Cyan in BGR
    alpha: float = 0.40,
) -> np.ndarray:
    """
    Renders vector change polygons on top of t2 base imagery with semi-transparent
    tactical fill and high-contrast antialiased borders.
    """
    chip_t2 = extract_localized_chip(base_raster_t2, center_rc=center_rc, chip_size=chip_size)
    overlay = chip_t2.copy()

    # Pre-calculate offset from raster to chip coordinates
    h_full, w_full = base_raster_t2.shape[:2]
    if center_rc is None:
        cr, cc = h_full // 2, w_full // 2
    else:
        cr, cc = center_rc

    half = chip_size // 2
    r_offset, c_offset = cr - half, cc - half

    for feat in polygons:
        geom = feat.get("geometry_geojson", {})
        coords = geom.get("coordinates", [])
        gtype = geom.get("type", "Polygon")

        # Parse polygon coordinate rings into pixel contours
        poly_rings = []
        if gtype == "Polygon" and coords:
            poly_rings = coords
        elif gtype == "MultiPolygon" and coords:
            for p in coords:
                poly_rings.extend(p)

        for ring in poly_rings:
            pts = []
            for pt in ring:
                if len(pt) < 2:
                    continue
                # If coordinates are in geographic WGS84 [lon, lat] and affine is provided
                if affine_transform is not None:
                    lon, lat = pt[0], pt[1]
                    inv_transform = ~affine_transform
                    c_px, r_px = inv_transform * (lon, lat)
                else:
                    # Assumed already in pixel coords [c, r] or [x, y]
                    c_px, r_px = pt[0], pt[1]

                # Map to chip local coordinates
                chip_c = int(round(c_px - c_offset))
                chip_r = int(round(r_px - r_offset))
                pts.append([chip_c, chip_r])

            if len(pts) >= 3:
                cnt = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                # Fill polygon on overlay with crimson
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
