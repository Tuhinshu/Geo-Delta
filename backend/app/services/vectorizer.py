"""
High-Throughput Vectorization & Morphological Noise Suppression Engine for GeoDelta
Implements FR-VEC-001, FR-VEC-002, and Law 5:
    1. Morphological noise suppression: B_clean = (B - K_3x3) + K_3x3
    2. Sub-pixel 8-connectivity polygon extraction via rasterio.features.shapes
    3. Affine transformation from raster pixel space to geographic WGS 84 (EPSG:4326)
    4. Deterministic ellipsoidal area & MGRS footprint calculation via footprint_calculator
    5. Sub-tactical clutter filter (strictly dropping polygons < 50 m²)
"""

import uuid
from typing import List, Dict, Any, Union, Optional
import numpy as np
import cv2
import rasterio.features
import affine
import torch

from app.core.schemas import DetectedPolygonFeature
from app.services.footprint_calculator import (
    calculate_geodesic_footprint,
    SUB_TACTICAL_CLUTTER_MIN_AREA_SQ_M
)


def apply_morphological_noise_suppression(
    prob_map: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.70,
    kernel_size: int = 3
) -> np.ndarray:
    """
    FR-VEC-001: Binarizes continuous probability map and suppresses high-frequency
    single-pixel noise and isolated edge artifacts using morphological opening:
        B_clean = (B - K_3x3) + K_3x3
        
    Args:
        prob_map: 2D probability map array in [0.0, 1.0], shape (H, W).
                  Can also be (1, H, W) or (1, 1, H, W) torch.Tensor or np.ndarray.
        threshold: Decision boundary tau (default 0.70).
        kernel_size: Morphological structuring element dimension (default 3x3).
        
    Returns:
        Cleaned binary mask as uint8 ndarray of shape (H, W) with values {0, 1}.
    """
    if isinstance(prob_map, torch.Tensor):
        prob_arr = prob_map.detach().cpu().squeeze().numpy()
    else:
        prob_arr = np.squeeze(prob_map)

    if prob_arr.ndim != 2:
        raise ValueError(f"Expected 2D probability map, got shape {prob_arr.shape}")

    # Binarize against threshold tau
    binary = (prob_arr >= threshold).astype(np.uint8)

    # 3x3 structuring element
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    # Morphological Opening: Erosion followed by Dilation
    clean_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return clean_binary


def extract_vector_features(
    prob_map: Union[np.ndarray, torch.Tensor],
    transform: affine.Affine,
    threshold: float = 0.70,
    tactical_class: str = "Tactical Change Detection",
    min_area_sq_m: float = SUB_TACTICAL_CLUTTER_MIN_AREA_SQ_M
) -> List[DetectedPolygonFeature]:
    """
    FR-VEC-002: High-throughput polygon extraction converting raster probability maps
    into georeferenced GeoJSON MultiPolygons with ellipsoidal metrics and MGRS tags.
    
    Args:
        prob_map: Continuous probability map (H, W) or tensor.
        transform: Affine transform mapping pixel (col, row) to WGS84 coordinates.
        threshold: Confidence threshold tau (default 0.70).
        tactical_class: Entity category label.
        min_area_sq_m: Sub-tactical clutter cutoff (FR-VEC-004, default 50.0 m²).
        
    Returns:
        List of DetectedPolygonFeature objects passing the clutter filter.
    """
    if isinstance(prob_map, torch.Tensor):
        prob_arr = prob_map.detach().cpu().squeeze().numpy().astype(np.float32)
    else:
        prob_arr = np.squeeze(prob_map).astype(np.float32)

    # 1. Apply morphological opening
    clean_binary = apply_morphological_noise_suppression(prob_arr, threshold=threshold)

    # Quick exit if no change pixels detected
    if not np.any(clean_binary):
        return []

    # 2. Label 8-connected components to compute per-polygon confidence in O(1)
    num_labels, labels = cv2.connectedComponents(clean_binary, connectivity=8)
    if num_labels <= 1:
        return []

    # Precompute per-component mean confidence in O(1) vectorized pass
    flat_labels = labels.ravel().astype(np.int32)  # cv2 returns int32; cast ensures Pyright sees integer dtype
    counts = np.bincount(flat_labels, minlength=num_labels)
    sums = np.bincount(flat_labels, weights=prob_arr.ravel(), minlength=num_labels)
    label_means = np.divide(
        sums,
        counts,
        out=np.full(num_labels, threshold, dtype=np.float32),
        where=counts > 0
    )

    # 3. Vector extraction via rasterio.features.shapes over labeled raster
    # connectivity=8 matches standard military geospatial topology
    raw_shapes = rasterio.features.shapes(
        labels.astype(np.int32),
        mask=(labels > 0),
        connectivity=8,
        transform=transform
    )

    detected_features: List[DetectedPolygonFeature] = []

    for geom, label_val in raw_shapes:
        label_id = int(label_val)
        if label_id == 0:
            continue

        # Convert Polygon to MultiPolygon structure conforming to schema
        if geom["type"] == "Polygon":
            multipolygon_geom: Dict[str, Any] = {
                "type": "MultiPolygon",
                "coordinates": [geom["coordinates"]]
            }
        elif geom["type"] == "MultiPolygon":
            multipolygon_geom = geom
        else:
            continue

        # Calculate ellipsoidal ground footprint, centroid, and MGRS (Law 5)
        area_sq_m, area_ha, centroid_wgs84, mgrs_str = calculate_geodesic_footprint(multipolygon_geom)

        # FR-VEC-004 Clutter Filter: Discard geometries < 50 m²
        if area_sq_m < min_area_sq_m:
            continue

        # Retrieve precomputed mean confidence in O(1)
        mean_conf = float(label_means[label_id]) if label_id < num_labels else float(threshold)
        mean_conf = max(0.0, min(1.0, round(mean_conf, 4)))

        feature = DetectedPolygonFeature(
            feature_id=str(uuid.uuid4()),
            tactical_class=tactical_class,
            confidence=mean_conf,
            area_sq_meters=round(area_sq_m, 2),
            area_hectares=max(0.005, area_ha),
            centroid_wgs84=centroid_wgs84,
            centroid_mgrs=mgrs_str,
            geometry_geojson=multipolygon_geom
        )
        detected_features.append(feature)

    return detected_features


def export_feature_collection(features: List[DetectedPolygonFeature]) -> Dict[str, Any]:
    """
    Serializes a list of DetectedPolygonFeatures into a standard GeoJSON FeatureCollection.
    Ready for MapLibre GL and deck.gl client-side ingestion.
    """
    geojson_features = []
    for f in features:
        geojson_features.append({
            "type": "Feature",
            "id": f.feature_id,
            "geometry": f.geometry_geojson,
            "properties": {
                "feature_id": f.feature_id,
                "tactical_class": f.tactical_class,
                "confidence": f.confidence,
                "area_sq_meters": f.area_sq_meters,
                "area_hectares": f.area_hectares,
                "centroid_lat": f.centroid_wgs84[0],
                "centroid_lon": f.centroid_wgs84[1],
                "centroid_mgrs": f.centroid_mgrs
            }
        })

    return {
        "type": "FeatureCollection",
        "features": geojson_features
    }


class VectorResult:
    """Lightweight result container returned by VectorizerService.extract_polygons()."""

    def __init__(self, features: List[DetectedPolygonFeature]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)


class VectorizerService:
    """
    Object-oriented wrapper around extract_vector_features() and export_feature_collection().
    Provides the interface expected by FallbackDifferencingEngine and integration tests.
    """

    def extract_polygons(
        self,
        prob_map: Any,
        transform: Optional[Any] = None,
        threshold: float = 0.70,
        tactical_class: str = "Tactical Change Detection",
        min_area_sq_m: float = SUB_TACTICAL_CLUTTER_MIN_AREA_SQ_M,
    ) -> VectorResult:
        """
        FR-VEC-002: Converts a probability map into georeferenced polygon features.

        Args:
            prob_map: Continuous probability map (H, W) or torch.Tensor.
            transform: Affine transform mapping pixel space → WGS84.
                       If None, a default 10m/px identity transform is used.
            threshold: Confidence threshold tau (default 0.70).
            tactical_class: Entity category label for detected features.
            min_area_sq_m: Sub-tactical clutter cutoff in m² (default 50.0).

        Returns:
            VectorResult with .features list of DetectedPolygonFeature.
        """
        import affine as _affine
        if transform is None:
            # Default: 10m GSD identity-ish transform centred on (0, 0)
            transform = _affine.Affine(1e-4, 0.0, 0.0, 0.0, -1e-4, 0.0)

        features = extract_vector_features(
            prob_map=prob_map,
            transform=transform,
            threshold=threshold,
            tactical_class=tactical_class,
            min_area_sq_m=min_area_sq_m,
        )
        return VectorResult(features)

    def to_geojson(self, features: List[DetectedPolygonFeature]) -> Dict[str, Any]:
        """Serializes features to a GeoJSON FeatureCollection."""
        return export_feature_collection(features)
