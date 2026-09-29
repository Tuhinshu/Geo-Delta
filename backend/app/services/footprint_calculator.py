"""
Deterministic Geodesic Footprint & Tactical Surface Clutter Engine for GeoDelta
Implements Law 5, FR-VEC-003, and FR-VEC-004:
    1. Geodesic area calculation on the WGS 84 ellipsoid (pyproj.Geod & PostGIS ST_Area)
    2. Exact centroid determination (WGS 84 EPSG:4326)
    3. NATO 10-figure Military Grid Reference System (MGRS) coordinate conversion
    4. Sub-tactical surface clutter filter (strictly dropping polygons < 50 m²)
"""

from typing import Tuple, Dict, Any, List, Optional, Union
import pyproj
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.validation import make_valid
import mgrs

from app.core.schemas import DetectedPolygonFeature

# Initialize global WGS 84 Geodesic calculator and MGRS converter
_WGS84_GEOD = pyproj.Geod(ellps="WGS84")
_MGRS_CONVERTER = mgrs.MGRS()

# Tactical clutter threshold in square meters (FR-VEC-004)
SUB_TACTICAL_CLUTTER_MIN_AREA_SQ_M: float = 50.0


def format_mgrs_string(mgrs_raw: str) -> str:
    """
    Formats a raw MGRS string into standard NATO military syntax:
    Example: '43RBK1234567890' -> '43R BK 12345 67890'
    """
    cleaned = mgrs_raw.strip().replace(" ", "")
    if len(cleaned) == 15:
        # 3 chars GZD (e.g. 43R), 2 chars 100km ID (e.g. BK), 5 chars easting, 5 chars northing
        return f"{cleaned[:3]} {cleaned[3:5]} {cleaned[5:10]} {cleaned[10:15]}"
    elif len(cleaned) == 14:
        # 2 chars GZD (e.g. 4R), 2 chars 100km ID, 5 easting, 5 northing
        return f"{cleaned[:2]} {cleaned[2:4]} {cleaned[4:9]} {cleaned[9:14]}"
    return cleaned


def calculate_geodesic_footprint(
    geom_input: Any
) -> Tuple[float, float, Tuple[float, float], str]:
    """
    Calculates exact ellipsoidal ground area, hectares, WGS84 centroid, and 10-figure MGRS.
    
    Law 5 Compliance:
        Separates spatial arithmetic from language models. Uses Karney's ellipsoidal
        geodesic area algorithm via pyproj.Geod(ellps="WGS84"), matching PostGIS ST_Area(geom::geography).
        
    Args:
        geom_input: GeoJSON geometry dictionary or Shapely Polygon/MultiPolygon.
        
    Returns:
        Tuple containing:
            - area_sq_meters: Geodesic area in square meters
            - area_hectares: Geodesic area in hectares (area_sq_m / 10000.0)
            - centroid_wgs84: (latitude, longitude) rounded to 6 decimal places
            - centroid_mgrs: 10-figure MGRS string (e.g. '43R BK 12345 67890')
    """
    # 1. Parse into valid Shapely geometry
    if isinstance(geom_input, dict):
        shapely_geom = shape(geom_input)
    else:
        shapely_geom = geom_input

    if not shapely_geom.is_valid:
        shapely_geom = make_valid(shapely_geom)

    # 2. Compute ellipsoidal surface area on WGS84
    raw_area, _ = _WGS84_GEOD.geometry_area_perimeter(shapely_geom)
    area_sq_m = abs(float(raw_area))
    area_hectares = round(area_sq_m / 10000.0, 4)

    # 3. Compute centroid
    centroid = shapely_geom.centroid
    lon = float(centroid.x)
    lat = float(centroid.y)
    centroid_wgs84 = (round(lat, 6), round(lon, 6))

    # 4. Convert WGS 84 coordinates to 10-figure MGRS
    try:
        raw_mgrs = _MGRS_CONVERTER.toMGRS(lat, lon, MGRSPrecision=5)
        mgrs_str = format_mgrs_string(raw_mgrs)
    except Exception:
        # Fallback if coordinates near poles or out of MGRS domain
        mgrs_str = f"GRID-{round(lat, 4)}N-{round(lon, 4)}E"

    return area_sq_m, area_hectares, centroid_wgs84, mgrs_str


def is_tactical_feature(area_sq_meters: float, min_area_sq_m: float = SUB_TACTICAL_CLUTTER_MIN_AREA_SQ_M) -> bool:
    """
    FR-VEC-004 Sub-tactical clutter filter.
    Returns True if the feature has significant ground footprint (>= 50 m²), False otherwise.
    """
    return area_sq_meters >= min_area_sq_m


def filter_sub_tactical_clutter(
    features: List[DetectedPolygonFeature],
    min_area_sq_m: float = SUB_TACTICAL_CLUTTER_MIN_AREA_SQ_M
) -> List[DetectedPolygonFeature]:
    """
    Filters a list of detected polygon features, strictly dropping any feature
    whose ellipsoidal ground area is below min_area_sq_m (default 50.0 m²).
    
    Args:
        features: List of candidate DetectedPolygonFeature objects.
        min_area_sq_m: Minimum area cutoff in square meters.
        
    Returns:
        Filtered list of DetectedPolygonFeature objects.
    """
    return [f for f in features if f.area_sq_meters >= min_area_sq_m]


class GeodesicFootprintCalculator:
    """
    Object-oriented wrapper around the module-level geodesic footprint functions.
    Provides the interface expected by FallbackDifferencingEngine and integration tests.
    """

    def filter_clutter_and_calculate_footprints(
        self,
        polygons: List[DetectedPolygonFeature],
        min_area_sq_m: float = SUB_TACTICAL_CLUTTER_MIN_AREA_SQ_M,
    ) -> List[DetectedPolygonFeature]:
        """
        Filters sub-tactical clutter and returns only features meeting the minimum
        area threshold (FR-VEC-004). Delegates to filter_sub_tactical_clutter().

        Args:
            polygons: Candidate DetectedPolygonFeature list from VectorizerService.
            min_area_sq_m: Minimum geodesic area cutoff in square meters (default 50 m²).

        Returns:
            Filtered list of tactically significant DetectedPolygonFeature objects.
        """
        return filter_sub_tactical_clutter(polygons, min_area_sq_m)

    def calculate_footprint(self, geom_input: Any) -> Tuple[float, float, Tuple[float, float], str]:
        """
        Convenience wrapper for calculate_geodesic_footprint().
        Returns (area_sq_m, area_ha, centroid_wgs84, mgrs_str).
        """
        return calculate_geodesic_footprint(geom_input)
