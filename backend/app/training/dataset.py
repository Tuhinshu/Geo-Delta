"""
Bitemporal Satellite Change Detection Dataset for GeoDelta.
Supports:
1. Standard Benchmark Datasets (LEVIR-CD, WHU-CD, OSCD) with directory layout:
   - A/ (Pre-event T1)
   - B/ (Post-event T2)
   - label/ (Ground truth binary change masks)
2. Tactical Synthetic Pairs (Airbases, revetments, trenches, grading)
"""

import os
from glob import glob
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class BitemporalChangeDataset(Dataset):
    """
    PyTorch Dataset for multi-temporal satellite change detection.
    Loads paired T1, T2 rasters and corresponding binary change masks.
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        image_size: Tuple[int, int] = (256, 256),
        default_prompt: str = "Identify newly constructed infrastructure and surface alterations",
        pairs: Optional[List[Dict[str, Any]]] = None
    ):
        self.image_size = image_size
        self.default_prompt = default_prompt
        self.samples: List[Dict[str, str]] = []

        if pairs is not None:
            self.samples = pairs
        elif root_dir and os.path.isdir(root_dir):
            self._load_from_directory(root_dir)

    def _load_from_directory(self, root_dir: str) -> None:
        """Discovers paired images from standard LEVIR-CD directory hierarchy."""
        dir_t1 = os.path.join(root_dir, "A")
        dir_t2 = os.path.join(root_dir, "B")
        dir_label = os.path.join(root_dir, "label")

        if os.path.isdir(dir_t1) and os.path.isdir(dir_t2):
            for t1_path in glob(os.path.join(dir_t1, "*.*")):
                fname = os.path.basename(t1_path)
                t2_path = os.path.join(dir_t2, fname)
                label_path = os.path.join(dir_label, fname) if os.path.isdir(dir_label) else None

                if os.path.isfile(t2_path):
                    self.samples.append({
                        "t1": t1_path,
                        "t2": t2_path,
                        "label": label_path if (label_path and os.path.isfile(label_path)) else None,
                        "prompt": self.default_prompt
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        item = self.samples[idx]

        # 1. Load and resize T1 image
        img_t1 = Image.open(item["t1"]).convert("RGB")
        img_t1 = img_t1.resize(self.image_size, Image.BILINEAR)
        t1_arr = np.array(img_t1, dtype=np.float32) / 255.0
        # Shape: (3, H, W)
        t1_tensor = torch.from_numpy(t1_arr).permute(2, 0, 1).contiguous()

        # 2. Load and resize T2 image
        img_t2 = Image.open(item["t2"]).convert("RGB")
        img_t2 = img_t2.resize(self.image_size, Image.BILINEAR)
        t2_arr = np.array(img_t2, dtype=np.float32) / 255.0
        # Shape: (3, H, W)
        t2_tensor = torch.from_numpy(t2_arr).permute(2, 0, 1).contiguous()

        # 3. Load or synthesize change mask
        if item.get("label") and os.path.isfile(item["label"]):
            mask_img = Image.open(item["label"]).convert("L")
            mask_img = mask_img.resize(self.image_size, Image.NEAREST)
            mask_arr = np.array(mask_img, dtype=np.float32)
            # Threshold to binary [0.0, 1.0]
            mask_arr = (mask_arr > 127).astype(np.float32)
        else:
            # Fallback: Absolute pixel luminance difference proxy
            diff = np.abs(t2_arr.mean(axis=-1) - t1_arr.mean(axis=-1))
            mask_arr = (diff > 0.15).astype(np.float32)

        # Shape: (1, H, W)
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).contiguous()
        prompt = item.get("prompt", self.default_prompt)

        return t1_tensor, t2_tensor, mask_tensor, prompt


def create_tactical_sample_pairs(output_dir: str = "storage/samples/tactical_train") -> str:
    """
    Generates a calibrated set of bitemporal training pairs with realistic military targets
    (runways, hangars, trenches, fortified berms) for defense-grade fine-tuning.
    """
    dir_a = os.path.join(output_dir, "A")
    dir_b = os.path.join(output_dir, "B")
    dir_lbl = os.path.join(output_dir, "label")
    os.makedirs(dir_a, exist_ok=True)
    os.makedirs(dir_b, exist_ok=True)
    os.makedirs(dir_lbl, exist_ok=True)

    np.random.seed(42)

    scenarios = [
        ("runway_ext", "Newly paved tarmac and airstrip extension", "tarmac"),
        ("trench_system", "Excavated zigzag defense trench line", "trench"),
        ("fortified_bunker", "Reinforced concrete command bunker", "bunker"),
        ("perimeter_berm", "Defensive earthen berm and security perimeter", "berm"),
        ("vehicle_depot", "Vehicle motor pool and equipment staging revetments", "depot"),
        ("hangar_facility", "Newly erected aircraft maintenance hangar", "hangar"),
        ("grading_work", "Heavy terrain surface grading and soil clearing", "grading"),
        ("helipad_zone", "Reinforced concrete forward landing helipad", "helipad"),
    ]

    for name, prompt, target_type in scenarios:
        # Base background texture: arid terrain
        h, w = 256, 256
        x = np.linspace(0, 8 * np.pi, w)
        y = np.linspace(0, 8 * np.pi, h)
        xx, yy = np.meshgrid(x, y)
        elevation = (np.sin(xx * 0.4) * np.cos(yy * 0.4) * 0.3 + 0.5)

        noise1 = np.random.normal(0, 15, (h, w))
        t1_r = np.clip(160 + elevation * 40 + noise1, 50, 240).astype(np.uint8)
        t1_g = np.clip(140 + elevation * 35 + noise1, 40, 220).astype(np.uint8)
        t1_b = np.clip(110 + elevation * 30 + noise1, 30, 200).astype(np.uint8)
        t1_rgb = np.stack([t1_r, t1_g, t1_b], axis=-1)

        t2_rgb = t1_rgb.copy()
        mask = np.zeros((h, w), dtype=np.uint8)

        # Inject specific targets
        if target_type == "tarmac":
            # Linear dark asphalt strip
            t2_rgb[110:140, 30:220] = [60, 65, 70]
            mask[110:140, 30:220] = 255
        elif target_type == "trench":
            # Zig-zag excavated trench
            for i in range(20, 230):
                offset = int(np.sin(i / 15.0) * 12)
                t2_rgb[120 + offset:126 + offset, i] = [40, 35, 30]
                mask[120 + offset:126 + offset, i] = 255
        elif target_type == "bunker":
            # Rectangular bunker with blast wall
            t2_rgb[90:160, 100:160] = [180, 185, 190]
            mask[90:160, 100:160] = 255
        elif target_type == "berm":
            # Curving soil mound
            for i in range(30, 220):
                t2_rgb[i, 70:82] = [190, 120, 70]
                mask[i, 70:82] = 255
        elif target_type == "hangar":
            # Large reflective roof
            t2_rgb[80:150, 70:180] = [210, 215, 220]
            mask[80:150, 70:180] = 255
        elif target_type == "helipad":
            # Circular helipad with H marking
            yy, xx = np.ogrid[:h, :w]
            dist = np.sqrt((xx - 128)**2 + (yy - 128)**2)
            circle = dist <= 40
            t2_rgb[circle] = [90, 95, 100]
            mask[circle] = 255
            # H marking in center
            t2_rgb[108:148, 116:122] = [255, 255, 255]
            t2_rgb[108:148, 134:140] = [255, 255, 255]
            t2_rgb[125:131, 122:134] = [255, 255, 255]
        else:
            # Graded soil clearing
            t2_rgb[70:180, 80:190] = [170, 105, 60]
            mask[70:180, 80:190] = 255

        # Save files
        Image.fromarray(t1_rgb).save(os.path.join(dir_a, f"{name}.png"))
        Image.fromarray(t2_rgb).save(os.path.join(dir_b, f"{name}.png"))
        Image.fromarray(mask).save(os.path.join(dir_lbl, f"{name}.png"))

    return output_dir
