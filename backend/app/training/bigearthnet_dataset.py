"""
BigEarthNet-S2 Multi-Spectral Training Dataset & Pair Generator for GeoDelta.
SIH Problem Statement SIH26227 - Ministry of Defence, Government of India.

Ingests real Sentinel-2 Level-2A surface reflectance tiles from BigEarthNet-S2 (tar.zst archive),
extracts multi-spectral bands (B02, B03, B04, B08), applies BOA radiometric reflectance normalization,
and constructs bitemporal change detection pairs (T1, T2, binary ground truth mask, VLM prompt).
"""

import os
import glob
import random
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset


TACTICAL_ALTERATION_PROMPTS = {
    "runway": "Newly paved runway extensions, airstrips, and tarmac",
    "bunker": "Fortified vehicle revetments and military bunkers",
    "trench": "Defensive trench excavation and perimeter berms",
    "facility": "Newly constructed military infrastructure and radar domes",
    "grading": "Terrain surface grading, soil clearing, and earthworks"
}


def load_sentinel2_bands(patch_path: str) -> Dict[str, np.ndarray]:
    """
    Reads Sentinel-2 10m bands (B02, B03, B04, B08) from a BigEarthNet patch directory.
    Returns raw 16-bit surface reflectance numpy arrays.
    """
    pname = os.path.basename(patch_path)
    bands = {}
    band_names = ["B02", "B03", "B04", "B08"]

    for bname in band_names:
        fpath = os.path.join(patch_path, f"{pname}_{bname}.tif")
        if os.path.isfile(fpath):
            img = Image.open(fpath)
            bands[bname] = np.array(img, dtype=np.float32)
        else:
            # Fallback if specific band file is missing
            bands[bname] = np.zeros((120, 120), dtype=np.float32)

    return bands


def composite_sentinel2_rgb(bands: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Converts 16-bit Sentinel-2 BOA surface reflectance bands (B04=Red, B03=Green, B02=Blue)
    into standard 8-bit RGB array via 2%-98% percentile radiometric stretching.
    """
    r = bands.get("B04", np.zeros((120, 120), dtype=np.float32))
    g = bands.get("B03", np.zeros((120, 120), dtype=np.float32))
    b = bands.get("B02", np.zeros((120, 120), dtype=np.float32))

    rgb = np.stack([r, g, b], axis=-1)
    p2, p98 = np.percentile(rgb, (2, 98))
    if p98 - p2 < 1e-4:
        rgb_norm = np.clip(rgb / 3000.0, 0.0, 1.0)
    else:
        rgb_norm = np.clip((rgb - p2) / (p98 - p2), 0.0, 1.0)

    return (rgb_norm * 255.0).astype(np.uint8)


def synthesize_sentinel2_change_pair(
    base_rgb: np.ndarray,
    alteration_type: Optional[str] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Synthesizes a realistic tactical ground alteration onto a real Sentinel-2 surface patch.
    Returns:
        t1_rgb: Baseline image (120x120x3)
        t2_rgb: Post-event image with alteration (120x120x3)
        mask: Ground-truth binary alteration mask (120x120, uint8 with 0 or 255)
        prompt: Natural language query corresponding to the alteration
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    h, w, c = base_rgb.shape
    t1_rgb = base_rgb.copy()
    t2_pil = Image.fromarray(base_rgb).convert("RGB")
    mask_pil = Image.new("L", (w, h), 0)

    draw_t2 = ImageDraw.Draw(t2_pil)
    draw_mask = ImageDraw.Draw(mask_pil)

    types = list(TACTICAL_ALTERATION_PROMPTS.keys())
    chosen_type = alteration_type if alteration_type in types else random.choice(types)
    prompt = TACTICAL_ALTERATION_PROMPTS[chosen_type]

    if chosen_type == "runway":
        # Runway extension: long linear paved strip
        rx1 = random.randint(10, 30)
        ry1 = random.randint(45, 65)
        rx2 = random.randint(90, 110)
        ry2 = ry1 + random.randint(12, 22)
        # Asphalt tarmac color (dark grey / high contrast pavement)
        tarmac_color = (48, 52, 58)
        draw_t2.rectangle([rx1, ry1, rx2, ry2], fill=tarmac_color)
        draw_mask.rectangle([rx1, ry1, rx2, ry2], fill=255)

    elif chosen_type == "bunker":
        # Hardened revetments / bunkers: rectangular / trapezoidal clusters
        bx = random.randint(30, 80)
        by = random.randint(30, 80)
        bw = random.randint(18, 28)
        bh = random.randint(14, 22)
        concrete_color = (130, 134, 138)
        draw_t2.rectangle([bx, by, bx + bw, by + bh], fill=concrete_color)
        draw_mask.rectangle([bx, by, bx + bw, by + bh], fill=255)

    elif chosen_type == "trench":
        # Defensive trench network: zigzagging polygonal line
        points = []
        cx = random.randint(20, 30)
        cy = random.randint(20, 40)
        points.append((cx, cy))
        for _ in range(4):
            cx += random.randint(15, 25)
            cy += random.randint(-12, 12)
            cy = max(10, min(h - 10, cy))
            points.append((cx, cy))
        trench_color = (35, 28, 22)  # Shadowed excavation dark earth
        draw_t2.line(points, fill=trench_color, width=4)
        draw_mask.line(points, fill=255, width=4)

    elif chosen_type == "facility":
        # Radar domes / military facility expansion
        fx = random.randint(40, 75)
        fy = random.randint(40, 75)
        fr = random.randint(10, 18)
        facility_color = (195, 200, 208)
        draw_t2.ellipse([fx - fr, fy - fr, fx + fr, fy + fr], fill=facility_color)
        draw_mask.ellipse([fx - fr, fy - fr, fx + fr, fy + fr], fill=255)

    else:  # grading
        # Soil clearing / graded surface
        gx = random.randint(25, 70)
        gy = random.randint(25, 70)
        gw = random.randint(25, 45)
        gh = random.randint(25, 45)
        graded_soil = (155, 128, 96)
        draw_t2.rectangle([gx, gy, gx + gw, gy + gh], fill=graded_soil)
        draw_mask.rectangle([gx, gy, gx + gw, gy + gh], fill=255)

    t2_rgb = np.array(t2_pil, dtype=np.uint8)
    mask_arr = np.array(mask_pil, dtype=np.uint8)

    return t1_rgb, t2_rgb, mask_arr, prompt


class BigEarthNetS2Dataset(Dataset):
    """
    PyTorch Dataset operating on Sentinel-2 Level-2A imagery from BigEarthNet-S2.
    Produces paired bitemporal samples formatted for SiameseCrossAttentionEngine.
    """

    def __init__(
        self,
        patches_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        num_pairs_per_patch: int = 3,
        seed: int = 42
    ):
        self.patches_dir = patches_dir
        self.image_size = image_size
        self.num_pairs_per_patch = num_pairs_per_patch
        self.seed = seed

        # Discover all patch subdirectories
        self.patch_dirs: List[str] = []
        if os.path.isdir(patches_dir):
            for root, dirs, files in os.walk(patches_dir):
                if any(f.endswith("_B04.tif") for f in files):
                    self.patch_dirs.append(root)

        print(f"[BigEarthNetDataset] Initialized with {len(self.patch_dirs)} Sentinel-2 patches in '{patches_dir}'")

        # Cache baseline RGB composites
        self.base_images: List[np.ndarray] = []
        for pdir in self.patch_dirs:
            bands = load_sentinel2_bands(pdir)
            rgb = composite_sentinel2_rgb(bands)
            self.base_images.append(rgb)

    def __len__(self) -> int:
        return len(self.base_images) * self.num_pairs_per_patch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        patch_idx = idx % len(self.base_images)
        pair_variant = idx // len(self.base_images)
        base_rgb = self.base_images[patch_idx]

        alteration_keys = list(TACTICAL_ALTERATION_PROMPTS.keys())
        chosen_type = alteration_keys[pair_variant % len(alteration_keys)]

        # Synthesize ground truth pair
        t1_np, t2_np, mask_np, prompt = synthesize_sentinel2_change_pair(
            base_rgb,
            alteration_type=chosen_type,
            seed=self.seed + idx
        )

        # Resize to requested training resolution (256x256)
        t1_pil = Image.fromarray(t1_np).resize(self.image_size, Image.BILINEAR)
        t2_pil = Image.fromarray(t2_np).resize(self.image_size, Image.BILINEAR)
        mask_pil = Image.fromarray(mask_np).resize(self.image_size, Image.NEAREST)

        # Convert to PyTorch tensors
        # Channels-first: (3, H, W) normalized [0.0, 1.0]
        t1_arr = np.array(t1_pil, dtype=np.float32) / 255.0
        t2_arr = np.array(t2_pil, dtype=np.float32) / 255.0
        mask_arr = (np.array(mask_pil, dtype=np.float32) > 127).astype(np.float32)

        t1_tensor = torch.from_numpy(t1_arr).permute(2, 0, 1).contiguous()
        t2_tensor = torch.from_numpy(t2_arr).permute(2, 0, 1).contiguous()
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).contiguous()

        return t1_tensor, t2_tensor, mask_tensor, prompt
