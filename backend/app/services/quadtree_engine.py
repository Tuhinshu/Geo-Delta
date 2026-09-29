"""
GeoDelta Quadtree Tiling & GPU OOM Recovery Engine (NFR-SAFE-003)
Partitions high-resolution rasters into 4 overlapping quadrants with 10% spatial
overlap on CUDA OutOfMemoryError, processes sequentially, and blends seamless probability maps.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import torch


def create_blend_weights(h: int, w: int, overlap_y: int, overlap_x: int) -> np.ndarray:
    """
    Creates a 2D raised cosine (Hanning) blending weight mask to smoothly blend overlapping tiles.
    """
    wy = np.ones(h, dtype=np.float32)
    wx = np.ones(w, dtype=np.float32)

    # Blend vertical edges
    if overlap_y > 0 and 2 * overlap_y <= h:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, overlap_y, dtype=np.float32)))
        wy[:overlap_y] = ramp
        wy[-overlap_y:] = ramp[::-1]
    elif overlap_y > 0:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, h // 2, dtype=np.float32)))
        wy[: h // 2] = ramp
        wy[-(h // 2) :] = ramp[::-1]

    # Blend horizontal edges
    if overlap_x > 0 and 2 * overlap_x <= w:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, overlap_x, dtype=np.float32)))
        wx[:overlap_x] = ramp
        wx[-overlap_x:] = ramp[::-1]
    elif overlap_x > 0:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, w // 2, dtype=np.float32)))
        wx[: w // 2] = ramp
        wx[-(w // 2) :] = ramp[::-1]

    return np.outer(wy, wx)


def get_quadtree_tiles(
    h: int,
    w: int,
    overlap_ratio: float = 0.10,
) -> List[Tuple[int, int, int, int]]:
    """
    Partitions an image of size (H, W) into 4 overlapping quadrants.
    Returns list of slice bounds: [(r1, r2, c1, c2), ...]
    """
    mid_h = h // 2
    mid_w = w // 2
    ov_h = (round(mid_h * overlap_ratio))
    ov_w = (round(mid_w * overlap_ratio))

    tiles = [
        # Top-Left Quadrant
        (0, min(h, mid_h + ov_h), 0, min(w, mid_w + ov_w)),
        # Top-Right Quadrant
        (0, min(h, mid_h + ov_h), max(0, mid_w - ov_w), w),
        # Bottom-Left Quadrant
        (max(0, mid_h - ov_h), h, 0, min(w, mid_w + ov_w)),
        # Bottom-Right Quadrant
        (max(0, mid_h - ov_h), h, max(0, mid_w - ov_w), w),
    ]
    return tiles


def execute_quadtree_tiled_inference(
    model: Any,
    t1_tensor: torch.Tensor,
    t2_tensor: torch.Tensor,
    text_emb: torch.Tensor,
    overlap_ratio: float = 0.10,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Sequentially processes 4 overlapping quadrants and blends the resultant probability maps.
    
    Args:
        model: Siamese cross-attention change detection network.
        t1_tensor: Reference tensor of shape (1, C, H, W).
        t2_tensor: Target tensor of shape (1, C, H, W).
        text_emb: Conditioning text embedding of shape (1, 512).
        overlap_ratio: Fractional boundary overlap (default 10%).
        device: Computation device.
        
    Returns:
        torch.Tensor of shape (1, 1, H, W) containing blended change probabilities.
    """
    _, _, h, w = t1_tensor.shape
    tiles = get_quadtree_tiles(h, w, overlap_ratio=overlap_ratio)

    prob_accum = np.zeros((h, w), dtype=np.float32)
    weight_accum = np.zeros((h, w), dtype=np.float32)

    for r1, r2, c1, c2 in tiles:
        tile_h = r2 - r1
        tile_w = c2 - c1

        # Clear GPU cache before each tile pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            tile_t1 = t1_tensor[:, :, r1:r2, c1:c2].to(device)
            tile_t2 = t2_tensor[:, :, r1:r2, c1:c2].to(device)
            tile_emb = text_emb.to(device)

            out_prob = model(tile_t1, tile_t2, tile_emb)
            tile_prob = out_prob.squeeze().cpu().numpy()

        # Compute raised cosine weights for this quadrant
        ov_y = (round((tile_h / 2) * overlap_ratio))
        ov_x = (round((tile_w / 2) * overlap_ratio))
        tile_weights = create_blend_weights(tile_h, tile_w, ov_y, ov_x)

        prob_accum[r1:r2, c1:c2] += tile_prob * tile_weights
        weight_accum[r1:r2, c1:c2] += tile_weights

    # Normalize blended probabilities
    weight_accum = np.maximum(weight_accum, 1e-6)
    blended = np.clip(prob_accum / weight_accum, 0.0, 1.0)

    return torch.from_numpy(blended).unsqueeze(0).unsqueeze(0)
