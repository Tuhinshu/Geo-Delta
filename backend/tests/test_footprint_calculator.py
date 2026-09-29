"""
Unit & Geodesic Verification Tests for Footprint Calculator & Clutter Engine (Law 5, FR-VEC-003, FR-VEC-004).
"""

import pytest
import shapely.geometry  # type: ignore
from app.services.footprint_calculator import (
    calculate_geodesic_footprint,
    filter_sub_tactical_clutter,
    is_tactical_feature,
    format_mgrs_string,
    SUB_TACTICAL_CLUTTER_MIN_AREA_SQ_M
)
from app.core.schemas import DetectedPolygonFeature


def test_geodesic_area_karney_ellipsoid_accuracy():
    """
    Law 5 & Section 3.4: Tests geodesic area on WGS 84 ellipsoid.
    A nominal 100m x 100m square (10,000 m² nominal) must match expected
    geodesic surface area within 0.5% error margin.
    """
    # Location: Sector 4 (approx Lat 25.0, Lon 75.0)
    lat0, lon0 = 25.0, 75.0
    # WGS 84 degree meters at Lat 25.0:
    # 1 deg latitude ~ 110,787 meters, 1 deg longitude ~ 100,865 meters
    dlat = 100.0 / 110787.0
    dlon = 100.0 / 100865.0

    poly = shapely.geometry.box(lon0, lat0, lon0 + dlon, lat0 + dlat)
    area_sq_m, area_ha, centroid_wgs84, mgrs_str = calculate_geodesic_footprint(poly)

    nominal_area = 10000.0
    error_ratio = abs(area_sq_m - nominal_area) / nominal_area
    assert error_ratio < 0.005, f"Error {error_ratio * 100:.3f}% exceeds 0.5% tolerance"

    # Verify hectares conversion
    assert abs(area_ha - (area_sq_m / 10000.0)) < 0.001

    # Verify centroid coordinates
    assert abs(centroid_wgs84[0] - (lat0 + dlat / 2.0)) < 1e-4
    assert abs(centroid_wgs84[1] - (lon0 + dlon / 2.0)) < 1e-4

    # Verify 10-figure MGRS formatting (e.g. 43R EH 00000 64947)
    assert len(mgrs_str.replace(" ", "")) == 15
    assert mgrs_str.startswith("43R")


def test_sub_tactical_clutter_filter():
    """
    FR-VEC-004 & Section 3.4:
    Verifies that geometries with ellipsoidal area < 50 m² are strictly discarded.
    """
    # 1. Sub-tactical clutter feature: 5m x 5m = 25 m² (< 50 m²)
    dlat_small = 5.0 / 110787.0
    dlon_small = 5.0 / 100865.0
    poly_small = shapely.geometry.box(75.0, 25.0, 75.0 + dlon_small, 25.0 + dlat_small)
    area_small, ha_small, cent_small, mgrs_small = calculate_geodesic_footprint(poly_small)
    assert area_small < SUB_TACTICAL_CLUTTER_MIN_AREA_SQ_M
    assert not is_tactical_feature(area_small)

    # 2. Significant tactical feature: 20m x 20m = 400 m² (>= 50 m²)
    dlat_large = 20.0 / 110787.0
    dlon_large = 20.0 / 100865.0
    poly_large = shapely.geometry.box(75.0, 25.0, 75.0 + dlon_large, 25.0 + dlat_large)
    area_large, ha_large, cent_large, mgrs_large = calculate_geodesic_footprint(poly_large)
    assert area_large >= SUB_TACTICAL_CLUTTER_MIN_AREA_SQ_M
    assert is_tactical_feature(area_large)

    # Pydantic Schema Invariant: DetectedPolygonFeature strictly rejects area < 50 m²
    with pytest.raises(Exception):
        DetectedPolygonFeature(
            feature_id="small-1",
            tactical_class="Sensor Glint",
            confidence=0.72,
            area_sq_meters=round(area_small, 2),  # 25.02 m² < 50.0 m²
            area_hectares=0.005,
            centroid_wgs84=cent_small,
            centroid_mgrs=mgrs_small,
            geometry_geojson={"type": "MultiPolygon", "coordinates": []}
        )

    # Valid tactical features >= 50 m²
    feature_medium = DetectedPolygonFeature(
        feature_id="med-1",
        tactical_class="Small Bunker",
        confidence=0.78,
        area_sq_meters=75.0,
        area_hectares=0.0075,
        centroid_wgs84=cent_large,
        centroid_mgrs=mgrs_large,
        geometry_geojson={"type": "MultiPolygon", "coordinates": []}
    )
    feature_large = DetectedPolygonFeature(
        feature_id="large-1",
        tactical_class="Vehicle Revetment",
        confidence=0.88,
        area_sq_meters=round(area_large, 2),
        area_hectares=round(ha_large, 4),
        centroid_wgs84=cent_large,
        centroid_mgrs=mgrs_large,
        geometry_geojson={"type": "MultiPolygon", "coordinates": []}
    )

    # Test filtering with custom min_area_sq_m
    filtered = filter_sub_tactical_clutter([feature_medium, feature_large], min_area_sq_m=100.0)
    assert len(filtered) == 1
    assert filtered[0].feature_id == "large-1"



def test_mgrs_formatting_edge_cases():
    """Verifies NATO MGRS string syntax normalization."""
    # 15 char MGRS
    raw_15 = "43RBK1234567890"
    formatted = format_mgrs_string(raw_15)
    assert formatted == "43R BK 12345 67890"

    # Already formatted or whitespace
    raw_spaced = "43R BK 12345 67890"
    assert format_mgrs_string(raw_spaced) == "43R BK 12345 67890"
