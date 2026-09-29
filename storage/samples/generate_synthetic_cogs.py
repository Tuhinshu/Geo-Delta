"""
Synthetic Multi-Temporal 4-Band GeoTIFF Generator for GeoDelta
Generates co-registered 16-bit/8-bit multi-temporal rasters (t1, t2) with simulated tactical infrastructure.
Bands: Band 1 (Red), Band 2 (Green), Band 3 (Blue), Band 4 (NIR)
Coordinate Reference System: EPSG:4326 (WGS 84)
Native GSD: 10 meters / pixel
"""

import os
import math
import numpy as np

def create_synthetic_pair(
    output_dir: str = "./storage/samples",
    height: int = 512,
    width: int = 512,
    min_lat: float = 34.1200,
    max_lat: float = 34.1660,
    min_lon: float = 74.5600,
    max_lon: float = 74.6156
):
    os.makedirs(output_dir, exist_ok=True)
    t1_path = os.path.join(output_dir, "sentinel2_t1_20250115.tif")
    t2_path = os.path.join(output_dir, "sentinel2_t2_20250610.tif")
    gt_path = os.path.join(output_dir, "ground_truth_mask.tif")

    print(f"Generating synthetic bitemporal satellite pair ({width}x{height} pixels, 4 bands)...")

    # 1. Base terrain elevation & natural textures (common across t1 and t2)
    x = np.linspace(0, 4 * np.pi, width)
    y = np.linspace(0, 4 * np.pi, height)
    xx, yy = np.meshgrid(x, y)
    elevation = (np.sin(xx) * np.cos(yy) * 0.5 + 0.5)

    # Base surface reflectance in uint16 (typical Sentinel-2 L2A ranges: 500 to 4000)
    noise = np.random.normal(0, 50, (height, width))
    base_red = np.clip(1000 + elevation * 1200 + noise, 200, 10000).astype(np.uint16)
    base_green = np.clip(1100 + elevation * 1300 + noise, 200, 10000).astype(np.uint16)
    base_blue = np.clip(900 + elevation * 1000 + noise, 200, 10000).astype(np.uint16)
    base_nir = np.clip(2500 + elevation * 1800 + noise, 200, 10000).astype(np.uint16)

    # Assemble t1 array: shape (4, H, W)
    t1_data = np.stack([base_red, base_green, base_blue, base_nir], axis=0)

    # 2. t2 terrain has seasonal vegetation drying (NIR drops slightly) + tactical breach
    t2_data = t1_data.copy()
    # Seasonal NIR decrease across landscape (simulating winter -> dry summer)
    t2_data[3] = np.clip(t2_data[3].astype(np.int32) - 300, 200, 10000).astype(np.uint16)

    # Ground truth change mask: shape (H, W), initialized to 0
    gt_mask = np.zeros((height, width), dtype=np.uint8)

    # Inject Tactical Construction 1: Newly Paved Runway Extension (length 180px, width 18px)
    # Coordinates: row 160 to 178, col 120 to 300
    r_start, r_end = 160, 178
    c_start, c_end = 120, 300
    gt_mask[r_start:r_end, c_start:c_end] = 1

    # Asphalt / compacted tarmac signature in t2: High Red/Blue, Low NIR
    t2_data[0, r_start:r_end, c_start:c_end] = 3200  # High Red reflectance
    t2_data[1, r_start:r_end, c_start:c_end] = 3100  # Green
    t2_data[2, r_start:r_end, c_start:c_end] = 3000  # Blue
    t2_data[3, r_start:r_end, c_start:c_end] = 900   # Low NIR (asphalt absorption)

    # Inject Tactical Construction 2: Perimeter Bunker Revetments (30x30 box)
    b_r, b_c = 220, 340
    gt_mask[b_r:b_r+30, b_c:b_c+30] = 1
    t2_data[0, b_r:b_r+30, b_c:b_c+30] = 2800  # Disturbed soil / concrete
    t2_data[1, b_r:b_r+30, b_c:b_c+30] = 2700
    t2_data[2, b_r:b_r+30, b_c:b_c+30] = 2500
    t2_data[3, b_r:b_r+30, b_c:b_c+30] = 1200

    # Write files using rasterio if available, otherwise raw TIFF
    try:
        import rasterio
        from rasterio.transform import from_bounds

        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)
        profile = {
            "driver": "GTiff",
            "dtype": "uint16",
            "nodata": 0,
            "width": width,
            "height": height,
            "count": 4,
            "crs": "EPSG:4326",
            "transform": transform,
            "compress": "lzw"
        }

        with rasterio.open(t1_path, "w", **profile) as dst:
            dst.write(t1_data)
            dst.set_band_description(1, "B04_RED")
            dst.set_band_description(2, "B03_GREEN")
            dst.set_band_description(3, "B02_BLUE")
            dst.set_band_description(4, "B08_NIR")

        with rasterio.open(t2_path, "w", **profile) as dst:
            dst.write(t2_data)
            dst.set_band_description(1, "B04_RED")
            dst.set_band_description(2, "B03_GREEN")
            dst.set_band_description(3, "B02_BLUE")
            dst.set_band_description(4, "B08_NIR")

        mask_profile = profile.copy()
        mask_profile.update({"count": 1, "dtype": "uint8"})
        with rasterio.open(gt_path, "w", **mask_profile) as dst:
            dst.write(gt_mask, 1)

        print(f"Successfully generated GeoTIFF rasters with EPSG:4326 bounds:")
        print(f"  - Pre-Event (t1):  {t1_path}")
        print(f"  - Post-Event (t2): {t2_path}")
        print(f"  - Ground Truth:    {gt_path}")

    except ImportError:
        # Fallback using PIL or raw numpy
        from PIL import Image
        # Save as 4-channel TIFF
        im1 = Image.fromarray(t1_data[0:3].transpose(1, 2, 0).astype(np.uint8))
        im1.save(t1_path)
        im2 = Image.fromarray(t2_data[0:3].transpose(1, 2, 0).astype(np.uint8))
        im2.save(t2_path)
        Image.fromarray(gt_mask * 255).save(gt_path)
        print("Generated standard TIFF files (rasterio not present in host environment).")

if __name__ == "__main__":
    create_synthetic_pair()
