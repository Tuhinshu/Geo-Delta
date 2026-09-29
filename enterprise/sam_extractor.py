"""
SAM-Geo Primitive Extraction & Spatial Proximity Network Engine (Solution D).
Performs offline instance segmentation on static and multi-temporal satellite rasters,
extracts structural vector primitives (nodes V), and constructs spatial topology (edges E)
via Delaunay Triangulation for border-scale scene graph generation.
"""

from __future__ import annotations

import argparse
import json
import math
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field
from scipy.spatial import Delaunay


class TacticalObjectClass(str, Enum):
    """Semantic tactical classification categories for extracted terrain primitives."""
    STRUCTURE = "Structure"
    ROAD = "Road"
    AIRSTRIP = "Airstrip"
    TOWER = "Tower"
    GRADED_AREA = "GradedArea"
    REVETMENT = "Revetment"
    BERM = "Berm"
    PERIMETER_FENCE = "PerimeterFence"
    WATER_BODY = "WaterBody"
    MILITARY_POST = "MilitaryPost"
    UNKNOWN = "Unknown"


class SceneGraphNode(BaseModel):
    """
    Topological Scene Graph Node v_i in G = (V, E).
    Represents a discrete segmented spatial entity extracted offline.
    """
    node_id: str = Field(..., description="Unique alphanumeric identifier for the spatial entity")
    class_type: TacticalObjectClass = Field(default=TacticalObjectClass.STRUCTURE, description="Tactical entity class")
    centroid: Tuple[float, float] = Field(..., description="Centroid coordinates in WGS 84 (lat, lon)")
    area_m2: float = Field(..., ge=0.0, description="Planar/ellipsoidal surface footprint in square meters")
    perimeter_m: float = Field(..., ge=0.0, description="Perimeter boundary length in meters")
    aspect_ratio: float = Field(default=1.0, ge=1.0, description="Bounding rectangle aspect ratio (length / width)")
    bbox: Tuple[float, float, float, float] = Field(
        ..., description="Spatial bounding box in WGS 84 (min_lat, min_lon, max_lat, max_lon)"
    )
    epoch: str = Field(..., description="Temporal acquisition epoch (ISO 8601 date, e.g. '2025-01-15')")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Foundation model segmentation confidence")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Supplementary GIS/tactical metadata")


class SceneGraphEdge(BaseModel):
    """
    Topological Scene Graph Edge e_ij in G = (V, E).
    Represents spatial adjacency, road connectivity, or proximity network relationships.
    """
    edge_id: str = Field(..., description="Unique identifier for the directed/undirected spatial edge")
    source_id: str = Field(..., description="Node ID of the origin spatial entity")
    target_id: str = Field(..., description="Node ID of the destination spatial entity")
    relation_type: str = Field(
        default="NEAR",
        description="Relationship type: 'ADJACENT_TO', 'CONNECTED_BY_ROAD', 'NEAR', 'DEFENDS', 'ACCESSES'"
    )
    distance_m: float = Field(..., ge=0.0, description="Spatial geodesic or Euclidean distance in meters")
    epoch: str = Field(..., description="Temporal acquisition epoch")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Supplementary link properties")


# Alias for backward-compatible naming
SpatialPrimitive = SceneGraphNode


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes exact spherical geodesic distance in meters between two WGS 84 points.
    Mean Earth radius R = 6,371,000 meters.
    """
    r_earth = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r_earth * c


def latlon_to_cartesian_meters(
    lat: float, lon: float, lat0: float, lon0: float
) -> Tuple[float, float]:
    """
    Projects WGS 84 coordinates to local tangent plane Cartesian coordinates (x, y) in meters
    relative to a central origin (lat0, lon0).
    """
    r_earth = 6371000.0
    x = (math.radians(lon - lon0)) * r_earth * math.cos(math.radians(lat0))
    y = (math.radians(lat - lat0)) * r_earth
    return x, y


class SAMPrimitiveExtractor:
    """
    Offline batch worker engine that processes raw imagery or segmentation masks,
    extracts geometric vector primitives, and computes proximity topology via Delaunay Triangulation.
    """

    def __init__(
        self,
        default_epoch: str = "2025-01-15",
        min_area_m2: float = 25.0,
        max_proximity_m: float = 1200.0,
    ) -> None:
        self.default_epoch = default_epoch
        self.min_area_m2 = min_area_m2
        self.max_proximity_m = max_proximity_m

    def classify_geometry(
        self,
        area_m2: float,
        aspect_ratio: float,
        perimeter_m: float,
        hint: Optional[str] = None,
    ) -> TacticalObjectClass:
        """
        Infers tactical object class from geometric morphological properties
        (area, aspect ratio, compactness) when explicit semantic label is unassigned.
        """
        if hint:
            hint_clean = hint.strip().lower()
            for cls_enum in TacticalObjectClass:
                if cls_enum.value.lower() in hint_clean:
                    return cls_enum

        # Circularity measure: 4 * pi * Area / Perimeter^2 (1.0 = perfect circle)
        circularity = (4.0 * math.pi * area_m2) / max(perimeter_m ** 2, 1e-4)

        if aspect_ratio >= 8.0 and area_m2 >= 5000.0:
            return TacticalObjectClass.AIRSTRIP
        elif aspect_ratio >= 4.5:
            return TacticalObjectClass.ROAD
        elif area_m2 < 150.0 and circularity >= 0.70:
            return TacticalObjectClass.TOWER
        elif area_m2 >= 15000.0 and aspect_ratio < 2.5:
            return TacticalObjectClass.GRADED_AREA
        elif 300.0 <= area_m2 <= 4000.0 and circularity < 0.40:
            return TacticalObjectClass.REVETMENT
        elif area_m2 >= 50.0:
            return TacticalObjectClass.STRUCTURE
        else:
            return TacticalObjectClass.UNKNOWN

    def extract_primitives_from_geojson(
        self,
        geojson_data: Union[dict, str, Path],
        epoch: Optional[str] = None,
    ) -> List[SceneGraphNode]:
        """
        Parses GeoJSON FeatureCollection into validated SceneGraphNode primitives.
        Computes bounding box, centroid, area, perimeter, and aspect ratio for each feature.
        """
        active_epoch = epoch or self.default_epoch

        if isinstance(geojson_data, (str, Path)):
            with open(geojson_data, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = geojson_data

        features = data.get("features", [])
        nodes: List[SceneGraphNode] = []

        for idx, feat in enumerate(features):
            geom = feat.get("geometry", {})
            props = feat.get("properties", {})
            g_type = geom.get("type", "")
            coords = geom.get("coordinates", [])

            if not coords:
                continue

            # Flatten coordinates to extract vertices (lon, lat)
            flat_pts: List[Tuple[float, float]] = []
            if g_type == "Polygon":
                flat_pts = [(p[0], p[1]) for ring in coords for p in ring if len(p) >= 2]
            elif g_type == "MultiPolygon":
                flat_pts = [
                    (p[0], p[1])
                    for poly in coords
                    for ring in poly
                    for p in ring
                    if len(p) >= 2
                ]
            elif g_type == "Point":
                flat_pts = [(coords[0], coords[1])]
            elif g_type == "LineString":
                flat_pts = [(p[0], p[1]) for p in coords if len(p) >= 2]

            if not flat_pts:
                continue

            lons = [pt[0] for pt in flat_pts]
            lats = [pt[1] for pt in flat_pts]
            min_lon, max_lon = min(lons), max(lons)
            min_lat, max_lat = min(lats), max(lats)
            centroid = (float((min_lat + max_lat) / 2.0), float((min_lon + max_lon) / 2.0))

            # Area & Perimeter computation
            area_m2 = float(props.get("area_sq_m") or props.get("area_m2") or 0.0)
            if area_m2 <= 0.0:
                # Estimate planar area via approximate meter conversion
                lat_m = (max_lat - min_lat) * 111320.0
                lon_m = (max_lon - min_lon) * 111320.0 * math.cos(math.radians(centroid[0]))
                area_m2 = max(lat_m * lon_m * 0.7, 50.0)

            if area_m2 < self.min_area_m2:
                continue

            # Perimeter estimation
            perimeter_m = float(props.get("perimeter_m") or 0.0)
            if perimeter_m <= 0.0:
                lat_span = (max_lat - min_lat) * 111320.0
                lon_span = (max_lon - min_lon) * 111320.0 * math.cos(math.radians(centroid[0]))
                perimeter_m = 2.0 * (lat_span + lon_span)

            # Aspect ratio estimation
            lat_span = max((max_lat - min_lat) * 111320.0, 1.0)
            lon_span = max((max_lon - min_lon) * 111320.0 * math.cos(math.radians(centroid[0])), 1.0)
            aspect_ratio = max(lat_span / lon_span, lon_span / lat_span)

            # Tactical class resolution
            raw_class = props.get("tactical_class") or props.get("class") or props.get("type")
            class_type = self.classify_geometry(area_m2, aspect_ratio, perimeter_m, raw_class)

            node_id = str(props.get("feature_id") or props.get("id") or f"node_{active_epoch}_{idx:05d}")
            confidence = float(props.get("confidence", 0.95))

            node = SceneGraphNode(
                node_id=node_id,
                class_type=class_type,
                centroid=centroid,
                area_m2=round(area_m2, 2),
                perimeter_m=round(perimeter_m, 2),
                aspect_ratio=round(aspect_ratio, 2),
                bbox=(round(min_lat, 6), round(min_lon, 6), round(max_lat, 6), round(max_lon, 6)),
                epoch=active_epoch,
                confidence=confidence,
                properties=props,
            )
            nodes.append(node)

        return nodes

    def extract_primitives_from_mask(
        self,
        binary_mask: np.ndarray,
        affine_transform: Any,
        epoch: Optional[str] = None,
        gsd_m: float = 10.0,
    ) -> List[SceneGraphNode]:
        """
        Extracts structural primitives from a 2D binary segmentation mask array
        using connected component labeling and geometric moment analysis.
        """
        import cv2

        active_epoch = epoch or self.default_epoch
        mask_u8 = (binary_mask.astype(np.uint8) > 0).astype(np.uint8) * 255

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        nodes: List[SceneGraphNode] = []

        # Coordinate transformation helper
        def px_to_geo(c: float, r: float) -> Tuple[float, float]:
            if hasattr(affine_transform, "__mul__"):
                lon, lat = affine_transform * (c, r)
            elif hasattr(affine_transform, "__matmul__"):
                lon, lat = affine_transform @ (c, r)
            else:
                lon, lat = c * 0.0001, r * -0.0001
            return lat, lon

        for lbl in range(1, num_labels):
            area_px = stats[lbl, cv2.CC_STAT_AREA]
            area_m2 = float(area_px) * (gsd_m ** 2)

            if area_m2 < self.min_area_m2:
                continue

            left = stats[lbl, cv2.CC_STAT_LEFT]
            top = stats[lbl, cv2.CC_STAT_TOP]
            w = stats[lbl, cv2.CC_STAT_WIDTH]
            h = stats[lbl, cv2.CC_STAT_HEIGHT]

            aspect_ratio = max(float(w) / max(float(h), 1.0), float(h) / max(float(w), 1.0))
            perimeter_px = 2.0 * (w + h)
            perimeter_m = perimeter_px * gsd_m

            # Geometric bbox corners
            lat_top_left, lon_top_left = px_to_geo(left, top)
            lat_bot_right, lon_bot_right = px_to_geo(left + w, top + h)
            min_lat = min(lat_top_left, lat_bot_right)
            max_lat = max(lat_top_left, lat_bot_right)
            min_lon = min(lon_top_left, lon_bot_right)
            max_lon = max(lon_top_left, lon_bot_right)

            cx_px, cy_px = centroids[lbl]
            c_lat, c_lon = px_to_geo(cx_px, cy_px)

            class_type = self.classify_geometry(area_m2, aspect_ratio, perimeter_m)
            node_id = f"prim_{active_epoch}_{lbl:05d}"

            node = SceneGraphNode(
                node_id=node_id,
                class_type=class_type,
                centroid=(round(c_lat, 6), round(c_lon, 6)),
                area_m2=round(area_m2, 2),
                perimeter_m=round(perimeter_m, 2),
                aspect_ratio=round(aspect_ratio, 2),
                bbox=(round(min_lat, 6), round(min_lon, 6), round(max_lat, 6), round(max_lon, 6)),
                epoch=active_epoch,
                confidence=0.92,
                properties={"pixel_count": int(area_px)},
            )
            nodes.append(node)

        return nodes

    def build_proximity_network(
        self,
        nodes: List[SceneGraphNode],
        max_distance_m: Optional[float] = None,
        connect_roads: bool = True,
    ) -> List[SceneGraphEdge]:
        """
        Builds spatial proximity network (edges E) across all nodes via Delaunay Triangulation.
        Complexity: O(N log N) spatial neighbor graph construction, avoiding O(N^2) all-pairs scaling.
        Prunes edges where distance > max_distance_m.
        """
        max_dist = max_distance_m or self.max_proximity_m
        n = len(nodes)
        if n < 2:
            return []

        # Anchor origin to first node centroid
        lat0, lon0 = nodes[0].centroid
        coords_2d = np.zeros((n, 2), dtype=np.float64)

        for i, node in enumerate(nodes):
            x, y = latlon_to_cartesian_meters(node.centroid[0], node.centroid[1], lat0, lon0)
            coords_2d[i, 0] = x
            coords_2d[i, 1] = y

        candidate_pairs: Set[Tuple[int, int]] = set()

        if n >= 3:
            # Check if points are collinear or identical
            try:
                tri = Delaunay(coords_2d)
                # Extract simplex edges from Delaunay triangulation
                for simplex in tri.simplices:
                    for i in range(3):
                        u = int(simplex[i])
                        v = int(simplex[(i + 1) % 3])
                        if u != v:
                            pair = (min(u, v), max(u, v))
                            candidate_pairs.add(pair)
            except Exception:
                # Fallback to nearest neighbors if Delaunay degenerates (e.g. collinear points)
                for u in range(n):
                    for v in range(u + 1, min(u + 6, n)):
                        candidate_pairs.add((u, v))
        else:
            # Exactly 2 nodes
            candidate_pairs.add((0, 1))

        edges: List[SceneGraphEdge] = []
        edge_counter = 0

        for u, v in candidate_pairs:
            node_u = nodes[u]
            node_v = nodes[v]

            dist = haversine_distance_m(
                node_u.centroid[0], node_u.centroid[1],
                node_v.centroid[0], node_v.centroid[1],
            )

            if dist > max_dist:
                continue

            # Determine tactical relationship type
            is_road_connection = (
                connect_roads and
                (node_u.class_type in (TacticalObjectClass.ROAD, TacticalObjectClass.AIRSTRIP) or
                 node_v.class_type in (TacticalObjectClass.ROAD, TacticalObjectClass.AIRSTRIP))
            )

            if is_road_connection:
                rel = "CONNECTED_BY_ROAD"
            elif dist <= 150.0:
                rel = "ADJACENT_TO"
            elif node_u.class_type == TacticalObjectClass.REVETMENT or node_v.class_type == TacticalObjectClass.REVETMENT:
                rel = "DEFENDS"
            else:
                rel = "NEAR"

            edge_counter += 1
            epoch = node_u.epoch
            edge_id = f"edge_{epoch}_{edge_counter:06d}"

            edge = SceneGraphEdge(
                edge_id=edge_id,
                source_id=node_u.node_id,
                target_id=node_v.node_id,
                relation_type=rel,
                distance_m=round(dist, 2),
                epoch=epoch,
                properties={"simplex_delaunay": True},
            )
            edges.append(edge)

        return edges

    def extract_scene_graph(
        self,
        geojson_or_mask: Union[dict, str, Path, np.ndarray],
        epoch: str,
        affine_transform: Optional[Any] = None,
        max_proximity_m: Optional[float] = None,
    ) -> Tuple[List[SceneGraphNode], List[SceneGraphEdge]]:
        """
        Unified pipeline: extracts all nodes V and derives Delaunay proximity edges E.
        Returns (nodes, edges).
        """
        if isinstance(geojson_or_mask, np.ndarray):
            if affine_transform is None:
                raise ValueError("affine_transform must be provided when passing a numpy raster mask")
            nodes = self.extract_primitives_from_mask(geojson_or_mask, affine_transform, epoch=epoch)
        else:
            nodes = self.extract_primitives_from_geojson(geojson_or_mask, epoch=epoch)

        edges = self.build_proximity_network(nodes, max_distance_m=max_proximity_m)
        return nodes, edges


def main() -> None:
    """CLI entrypoint for running SAM-Geo offline primitive extraction."""
    parser = argparse.ArgumentParser(description="GeoDelta Solution D Offline SAM-Geo Extractor")
    parser.add_argument("--input-geojson", type=str, help="Path to input GeoJSON feature collection")
    parser.add_argument("--epoch", type=str, default="2025-01-15", help="Acquisition epoch (YYYY-MM-DD)")
    parser.add_argument("--output-json", type=str, help="Path to write compiled scene graph JSON")
    parser.add_argument("--max-dist", type=float, default=1200.0, help="Maximum proximity distance in meters")

    args = parser.parse_args()
    if not args.input_geojson:
        parser.print_help()
        return

    extractor = SAMPrimitiveExtractor(default_epoch=args.epoch, max_proximity_m=args.max_dist)
    nodes, edges = extractor.extract_scene_graph(args.input_geojson, epoch=args.epoch)

    output = {
        "epoch": args.epoch,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": [n.model_dump() for n in nodes],
        "edges": [e.model_dump() for e in edges],
    }

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Scene graph exported to {args.output_json} ({len(nodes)} nodes, {len(edges)} edges)")
    else:
        print(f"Extracted {len(nodes)} nodes, {len(edges)} edges for epoch {args.epoch}")


if __name__ == "__main__":
    main()
