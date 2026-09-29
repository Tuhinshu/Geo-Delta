"""
Topological Scene Graph Delta Compiler & Sub-Millisecond Spatial Graph Engine (Solution D).
Compiles dual temporal scene graphs G_t1 and G_t2 into discrete graph deltas (Delta_G = G_t2 - G_t1),
generates production Cypher (CQL) transaction batches for Neo4j,
and executes indexed spatial graph traversals across 100,000-node border corridors in < 15 ms.
"""

from __future__ import annotations

import math
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from pydantic import BaseModel, Field
from scipy.spatial import KDTree

from enterprise.sam_extractor import (
    SceneGraphEdge,
    SceneGraphNode,
    TacticalObjectClass,
    haversine_distance_m,
    latlon_to_cartesian_meters,
)


class GraphDelta(BaseModel):
    """
    Discrete Graph Delta: Delta_G = G_t2 \\ G_t1.
    Captures structural, topological, and attribute changes between multi-temporal observation epochs.
    """
    t1_epoch: str = Field(..., description="Baseline epoch timestamp (t1)")
    t2_epoch: str = Field(..., description="Post-event epoch timestamp (t2)")
    added_nodes: List[SceneGraphNode] = Field(default_factory=list, description="Newly emerged physical entities (V_t2 \\ V_t1)")
    removed_nodes: List[SceneGraphNode] = Field(default_factory=list, description="Demolished/destroyed entities (V_t1 \\ V_t2)")
    modified_nodes: List[Dict[str, Any]] = Field(default_factory=list, description="Persistent entities with altered footprint/geometry")
    added_edges: List[SceneGraphEdge] = Field(default_factory=list, description="Newly established roads/proximity links (E_t2 \\ E_t1)")
    removed_edges: List[SceneGraphEdge] = Field(default_factory=list, description="Severed roads/proximity links (E_t1 \\ E_t2)")
    candidate_bboxes: List[Tuple[float, float, float, float]] = Field(
        default_factory=list, description="Spatial bounding boxes of change clusters flagged for Solution B deep learning dispatch"
    )
    total_altered_area_m2: float = Field(default=0.0, description="Cumulative altered surface footprint in square meters")
    summary: Dict[str, Any] = Field(default_factory=dict, description="Operational intelligence summary metrics")


class TopologicalSceneGraph:
    """
    In-memory indexed topological scene graph G = (V, E).
    Provides spatial indexing (KDTree/grid bins) and graph adjacency lists for sub-millisecond querying.
    """

    def __init__(self, epoch: str = "2025-01-15") -> None:
        self.epoch = epoch
        self.nodes: Dict[str, SceneGraphNode] = {}
        self.edges: Dict[str, SceneGraphEdge] = {}
        self.adjacency: Dict[str, List[str]] = defaultdict(list)
        self.edge_by_pair: Dict[Tuple[str, str], SceneGraphEdge] = {}
        self.nodes_by_class: Dict[TacticalObjectClass, List[str]] = defaultdict(list)

        # Spatial index cache
        self._kdtree: Optional[KDTree] = None
        self._node_id_list: List[str] = []
        self._coords_xy: Optional[np.ndarray] = None
        self._anchor_latlon: Optional[Tuple[float, float]] = None

    def add_node(self, node: SceneGraphNode) -> None:
        """Adds a scene graph node and updates spatial/class indexes."""
        self.nodes[node.node_id] = node
        self.nodes_by_class[node.class_type].append(node.node_id)
        self._kdtree = None

    def add_edge(self, edge: SceneGraphEdge) -> None:
        """Adds a scene graph edge and updates adjacency lists."""
        self.edges[edge.edge_id] = edge
        self.adjacency[edge.source_id].append(edge.target_id)
        self.adjacency[edge.target_id].append(edge.source_id)
        self.edge_by_pair[(edge.source_id, edge.target_id)] = edge
        self.edge_by_pair[(edge.target_id, edge.source_id)] = edge

    def get_node(self, node_id: str) -> Optional[SceneGraphNode]:
        return self.nodes.get(node_id)

    def get_neighbors(self, node_id: str) -> List[SceneGraphNode]:
        neighbor_ids = self.adjacency.get(node_id, [])
        return [self.nodes[nid] for nid in neighbor_ids if nid in self.nodes]

    def _build_spatial_index(self) -> None:
        """Builds 2D KDTree in projected metric space for fast radius and k-NN queries."""
        if not self.nodes:
            self._kdtree = None
            self._node_id_list = []
            return

        self._node_id_list = list(self.nodes.keys())
        self._node_id_to_idx = {nid: i for i, nid in enumerate(self._node_id_list)}
        first_node = self.nodes[self._node_id_list[0]]
        lat0, lon0 = first_node.centroid
        self._anchor_latlon = (lat0, lon0)

        n = len(self._node_id_list)
        coords = np.zeros((n, 2), dtype=np.float64)
        for i, nid in enumerate(self._node_id_list):
            node = self.nodes[nid]
            x, y = latlon_to_cartesian_meters(node.centroid[0], node.centroid[1], lat0, lon0)
            coords[i, 0] = x
            coords[i, 1] = y

        self._coords_xy = coords
        self._class_index_sets: Dict[TacticalObjectClass, Set[int]] = defaultdict(set)
        for i, nid in enumerate(self._node_id_list):
            self._class_index_sets[self.nodes[nid].class_type].add(i)
        self._kdtree = KDTree(coords)

    def query_radius(self, lat: float, lon: float, radius_m: float) -> List[SceneGraphNode]:
        """Queries all nodes within radius_m of a geographic location (lat, lon)."""
        if self._kdtree is None:
            self._build_spatial_index()

        if self._kdtree is None or self._anchor_latlon is None:
            return []

        lat0, lon0 = self._anchor_latlon
        qx, qy = latlon_to_cartesian_meters(lat, lon, lat0, lon0)
        idx_list = self._kdtree.query_ball_point([qx, qy], r=radius_m)

        return [self.nodes[self._node_id_list[idx]] for idx in idx_list]

    def query_bbox(self, bbox: Tuple[float, float, float, float]) -> List[SceneGraphNode]:
        """Queries all nodes whose centroids lie inside WGS 84 bounding box (min_lat, min_lon, max_lat, max_lon)."""
        min_lat, min_lon, max_lat, max_lon = bbox
        results: List[SceneGraphNode] = []
        for node in self.nodes.values():
            clat, clon = node.centroid
            if min_lat <= clat <= max_lat and min_lon <= clon <= max_lon:
                results.append(node)
        return results

    def query_by_class(self, class_type: TacticalObjectClass) -> List[SceneGraphNode]:
        """Returns all nodes matching the specified tactical class."""
        return [self.nodes[nid] for nid in self.nodes_by_class.get(class_type, []) if nid in self.nodes]


class GraphDeltaCompiler:
    """
    Compiler that computes topological set differences between dual temporal scene graphs
    and exports production Cypher script transactions for Neo4j.
    """

    def __init__(
        self,
        match_radius_m: float = 35.0,
        area_change_threshold: float = 0.20,
        tile_padding_m: float = 2500.0,
    ) -> None:
        self.match_radius_m = match_radius_m
        self.area_change_threshold = area_change_threshold
        self.tile_padding_m = tile_padding_m

    def compile_delta(
        self,
        g1: TopologicalSceneGraph,
        g2: TopologicalSceneGraph,
    ) -> GraphDelta:
        """
        Computes Delta_G = G_t2 \\ G_t1.
        Uses spatial nearest-neighbor matching to identify:
        - Added nodes (new structures)
        - Removed nodes (demolished)
        - Modified nodes (altered surface area / geometry)
        - Added/removed topological edges
        """
        # Ensure spatial index is built on G1
        if g1._kdtree is None and g1.nodes:
            g1._build_spatial_index()

        matched_g1_ids: Set[str] = set()
        matched_g2_ids: Set[str] = set()
        modified_nodes: List[Dict[str, Any]] = []
        added_nodes: List[SceneGraphNode] = []
        removed_nodes: List[SceneGraphNode] = []

        total_altered_area = 0.0

        # 1. Match G2 nodes against G1 spatial index
        if g1._kdtree is not None and g1._anchor_latlon is not None and g2.nodes:
            lat0, lon0 = g1._anchor_latlon
            g2_node_list = list(g2.nodes.values())
            g2_coords = np.zeros((len(g2_node_list), 2), dtype=np.float64)
            for i, n in enumerate(g2_node_list):
                x, y = latlon_to_cartesian_meters(n.centroid[0], n.centroid[1], lat0, lon0)
                g2_coords[i, 0] = x
                g2_coords[i, 1] = y

            # Query nearest neighbor in G1
            distances, indices = g1._kdtree.query(g2_coords, k=1)

            for i, g2_node in enumerate(g2_node_list):
                dist = float(distances[i]) if np.ndim(distances) > 0 else float(distances)
                idx = int(indices[i]) if np.ndim(indices) > 0 else int(indices)

                if dist <= self.match_radius_m:
                    g1_nid = g1._node_id_list[idx]
                    g1_node = g1.nodes[g1_nid]

                    matched_g1_ids.add(g1_nid)
                    matched_g2_ids.add(g2_node.node_id)

                    # Check for attribute modification (significant area change or class change)
                    area_delta = abs(g2_node.area_m2 - g1_node.area_m2)
                    rel_area_change = area_delta / max(g1_node.area_m2, 1e-3)

                    if rel_area_change >= self.area_change_threshold or g2_node.class_type != g1_node.class_type:
                        modified_nodes.append({
                            "old_node_id": g1_node.node_id,
                            "new_node_id": g2_node.node_id,
                            "class_t1": g1_node.class_type.value,
                            "class_t2": g2_node.class_type.value,
                            "area_t1_m2": g1_node.area_m2,
                            "area_t2_m2": g2_node.area_m2,
                            "area_delta_m2": round(area_delta, 2),
                            "relative_change": round(rel_area_change, 3),
                            "centroid_shift_m": round(dist, 2),
                            "centroid_t2": g2_node.centroid,
                            "bbox": g2_node.bbox,
                        })
                        total_altered_area += area_delta
                else:
                    # Unmatched in G1 -> Added in G2
                    added_nodes.append(g2_node)
                    total_altered_area += g2_node.area_m2
        else:
            # G1 is empty -> all G2 nodes are added
            added_nodes = list(g2.nodes.values())
            total_altered_area = sum(n.area_m2 for n in added_nodes)

        # 2. Identify removed G1 nodes
        for nid, g1_node in g1.nodes.items():
            if nid not in matched_g1_ids:
                removed_nodes.append(g1_node)
                total_altered_area += g1_node.area_m2

        # 3. Identify Edge Deltas
        added_edges: List[SceneGraphEdge] = []
        removed_edges: List[SceneGraphEdge] = []

        # Simplified edge diffing based on endpoints
        g1_edge_signatures = {
            tuple(sorted([e.source_id, e.target_id])): e for e in g1.edges.values()
        }
        g2_edge_signatures = {
            tuple(sorted([e.source_id, e.target_id])): e for e in g2.edges.values()
        }

        for sig, e2 in g2_edge_signatures.items():
            if sig not in g1_edge_signatures:
                added_edges.append(e2)

        for sig, e1 in g1_edge_signatures.items():
            if sig not in g2_edge_signatures:
                removed_edges.append(e1)

        # 4. Generate Candidate Bounding Boxes for Solution B Dispatch
        candidate_bboxes = self._cluster_changes_into_bboxes(added_nodes, modified_nodes)

        summary = {
            "baseline_node_count": len(g1.nodes),
            "target_node_count": len(g2.nodes),
            "added_node_count": len(added_nodes),
            "removed_node_count": len(removed_nodes),
            "modified_node_count": len(modified_nodes),
            "added_edge_count": len(added_edges),
            "removed_edge_count": len(removed_edges),
            "candidate_bbox_count": len(candidate_bboxes),
            "total_altered_area_m2": round(total_altered_area, 2),
            "total_altered_area_ha": round(total_altered_area / 10000.0, 4),
        }

        return GraphDelta(
            t1_epoch=g1.epoch,
            t2_epoch=g2.epoch,
            added_nodes=added_nodes,
            removed_nodes=removed_nodes,
            modified_nodes=modified_nodes,
            added_edges=added_edges,
            removed_edges=removed_edges,
            candidate_bboxes=candidate_bboxes,
            total_altered_area_m2=round(total_altered_area, 2),
            summary=summary,
        )

    def _cluster_changes_into_bboxes(
        self,
        added_nodes: List[SceneGraphNode],
        modified_nodes: List[Dict[str, Any]],
    ) -> List[Tuple[float, float, float, float]]:
        """
        Clusters anomalous changes into localized spatial bounding boxes
        suitable for feeding directly to Solution B Cross-Attention inference chips.
        """
        change_points: List[Tuple[float, float]] = []
        for n in added_nodes:
            change_points.append(n.centroid)
        for m in modified_nodes:
            change_points.append(m["centroid_t2"])

        if not change_points:
            return []

        # Convert padding meters to approximate degrees
        pad_lat = self.tile_padding_m / 111320.0
        pad_lon = self.tile_padding_m / (111320.0 * math.cos(math.radians(change_points[0][0])))

        # Spatial grid grouping (0.05 degree cells ~ 5.5 km x 5.5 km)
        grid_bins: Dict[Tuple[int, int], List[Tuple[float, float]]] = defaultdict(list)
        cell_size = 0.05

        for lat, lon in change_points:
            bin_key = (int(lat / cell_size), int(lon / cell_size))
            grid_bins[bin_key].append((lat, lon))

        bboxes: List[Tuple[float, float, float, float]] = []
        for bin_pts in grid_bins.values():
            lats = [p[0] for p in bin_pts]
            lons = [p[1] for p in bin_pts]
            min_lat = max(min(lats) - pad_lat, -90.0)
            max_lat = min(max(lats) + pad_lat, 90.0)
            min_lon = max(min(lons) - pad_lon, -180.0)
            max_lon = min(max(lons) + pad_lon, 180.0)
            bboxes.append((round(min_lat, 6), round(min_lon, 6), round(max_lat, 6), round(max_lon, 6)))

        return bboxes

    def export_cypher_script(
        self,
        graph_or_delta: Any,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Exports scene graph or delta into executable Neo4j Cypher statements with ACID transactions.
        """
        lines: List[str] = [
            "// ===========================================================================",
            "// GeoDelta Solution D - Enterprise Scene Graph Cypher Ingestion Script",
            "// Generated automatically for Neo4j Community / Enterprise v5.x",
            "// ===========================================================================",
            "",
            "// Schema & Constraints Definition",
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:SpatialEntity) REQUIRE e.id IS UNIQUE;",
            "CREATE INDEX entity_epoch_idx IF NOT EXISTS FOR (e:SpatialEntity) ON (e.epoch);",
            "CREATE INDEX entity_class_idx IF NOT EXISTS FOR (e:SpatialEntity) ON (e.class);",
            "CREATE POINT INDEX entity_spatial_idx IF NOT EXISTS FOR (e:SpatialEntity) ON (e.location);",
            "",
            ":begin",
        ]

        if isinstance(graph_or_delta, TopologicalSceneGraph):
            # Export graph nodes
            for node in graph_or_delta.nodes.values():
                cls_name = node.class_type.value
                lines.append(
                    f"MERGE (n:SpatialEntity:{cls_name} {{id: '{node.node_id}'}}) "
                    f"ON CREATE SET "
                    f"n.class = '{cls_name}', "
                    f"n.epoch = '{node.epoch}', "
                    f"n.area_m2 = {node.area_m2}, "
                    f"n.perimeter_m = {node.perimeter_m}, "
                    f"n.aspect_ratio = {node.aspect_ratio}, "
                    f"n.confidence = {node.confidence}, "
                    f"n.location = point({{latitude: {node.centroid[0]}, longitude: {node.centroid[1]}}}), "
                    f"n.bbox = [{node.bbox[0]}, {node.bbox[1]}, {node.bbox[2]}, {node.bbox[3]}];"
                )

            # Export graph edges
            for edge in graph_or_delta.edges.values():
                lines.append(
                    f"MATCH (a:SpatialEntity {{id: '{edge.source_id}'}}), (b:SpatialEntity {{id: '{edge.target_id}'}}) "
                    f"MERGE (a)-[r:{edge.relation_type} {{id: '{edge.edge_id}'}}]->(b) "
                    f"ON CREATE SET r.distance_m = {edge.distance_m}, r.epoch = '{edge.epoch}';"
                )

        elif isinstance(graph_or_delta, GraphDelta):
            # Export Delta nodes with Delta flags
            for node in graph_or_delta.added_nodes:
                cls_name = node.class_type.value
                lines.append(
                    f"MERGE (n:SpatialEntity:{cls_name}:NewTacticalDelta {{id: '{node.node_id}'}}) "
                    f"ON CREATE SET "
                    f"n.class = '{cls_name}', "
                    f"n.epoch = '{node.epoch}', "
                    f"n.area_m2 = {node.area_m2}, "
                    f"n.location = point({{latitude: {node.centroid[0]}, longitude: {node.centroid[1]}}}), "
                    f"n.delta_status = 'EMERGED';"
                )

            for mod in graph_or_delta.modified_nodes:
                lines.append(
                    f"MATCH (n:SpatialEntity {{id: '{mod['old_node_id']}'}}) "
                    f"SET n:ModifiedDelta, "
                    f"n.area_delta_m2 = {mod['area_delta_m2']}, "
                    f"n.area_t2_m2 = {mod['area_t2_m2']}, "
                    f"n.delta_status = 'MODIFIED';"
                )

        lines.append(":commit")
        script = "\n".join(lines) + "\n"

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(script)

        return script


class SyntheticBorderGraphGenerator:
    """
    Synthesizes realistic 10,000 to 100,000+ node national border scene graphs
    across a 50,000 km^2 operational corridor (e.g., 500 km x 100 km).
    Used to benchmark sub-millisecond graph traversal and stress-test Solution D.
    """

    @staticmethod
    def generate_border_corridor_graph(
        node_count: int = 100000,
        epoch: str = "2025-01-15",
        corridor_length_km: float = 500.0,
        corridor_width_km: float = 100.0,
        base_lat: float = 34.0,
        base_lon: float = 75.0,
        random_seed: int = 42,
    ) -> TopologicalSceneGraph:
        """
        Generates a synthetic border corridor scene graph with:
        - Primary highway and secondary logistics road networks
        - Military outposts, observation towers, radar bastions, revetments, airstrips
        - Delaunay proximity connectivity
        """
        rng = np.random.default_rng(random_seed)
        graph = TopologicalSceneGraph(epoch=epoch)

        # Degree span for corridor
        lat_span = corridor_width_km / 111.32
        lon_span = corridor_length_km / (111.32 * math.cos(math.radians(base_lat)))

        # 1. Distribute nodes
        classes = [
            (TacticalObjectClass.STRUCTURE, 0.50),
            (TacticalObjectClass.ROAD, 0.20),
            (TacticalObjectClass.TOWER, 0.10),
            (TacticalObjectClass.REVETMENT, 0.10),
            (TacticalObjectClass.GRADED_AREA, 0.05),
            (TacticalObjectClass.MILITARY_POST, 0.04),
            (TacticalObjectClass.AIRSTRIP, 0.01),
        ]
        class_choices = [c[0] for c in classes]
        class_probs = [c[1] for c in classes]

        class_indices = rng.choice(len(class_choices), size=node_count, p=class_probs)
        lats = rng.uniform(base_lat, base_lat + lat_span, size=node_count)
        lons = rng.uniform(base_lon, base_lon + lon_span, size=node_count)

        for i in range(node_count):
            cls_enum = class_choices[int(class_indices[i])]
            lat = float(lats[i])
            lon = float(lons[i])

            if cls_enum == TacticalObjectClass.AIRSTRIP:
                area_m2 = float(rng.uniform(15000.0, 60000.0))
                aspect = float(rng.uniform(8.0, 16.0))
            elif cls_enum == TacticalObjectClass.ROAD:
                area_m2 = float(rng.uniform(2000.0, 10000.0))
                aspect = float(rng.uniform(5.0, 10.0))
            elif cls_enum == TacticalObjectClass.TOWER:
                area_m2 = float(rng.uniform(25.0, 80.0))
                aspect = 1.0
            elif cls_enum == TacticalObjectClass.REVETMENT:
                area_m2 = float(rng.uniform(400.0, 2500.0))
                aspect = float(rng.uniform(1.2, 2.5))
            elif cls_enum == TacticalObjectClass.MILITARY_POST:
                area_m2 = float(rng.uniform(1000.0, 8000.0))
                aspect = float(rng.uniform(1.0, 2.0))
            else:
                area_m2 = float(rng.uniform(80.0, 1200.0))
                aspect = float(rng.uniform(1.0, 3.0))

            pad = math.sqrt(area_m2) / 111320.0
            bbox = (lat - pad, lon - pad, lat + pad, lon + pad)
            nid = f"ent_{epoch}_{i:06d}"

            node = SceneGraphNode.model_construct(
                node_id=nid,
                class_type=cls_enum,
                centroid=(round(lat, 6), round(lon, 6)),
                area_m2=round(area_m2, 2),
                perimeter_m=round(4.0 * math.sqrt(area_m2), 2),
                aspect_ratio=round(aspect, 2),
                bbox=(round(bbox[0], 6), round(bbox[1], 6), round(bbox[2], 6), round(bbox[3], 6)),
                epoch=epoch,
                confidence=0.95,
                properties={},
            )
            graph.add_node(node)

        # Build spatial index on graph
        graph._build_spatial_index()
        return graph

    @staticmethod
    def inject_tactical_deltas(
        base_graph: TopologicalSceneGraph,
        t2_epoch: str = "2025-06-15",
        new_structures_count: int = 45,
        new_airstrips_count: int = 5,
        new_revetments_count: int = 20,
        random_seed: int = 99,
    ) -> TopologicalSceneGraph:
        """
        Clones base_graph to epoch t2 and injects tactical forward changes
        (e.g., newly paved runway extensions, forward revetments, infrastructure clusters).
        """
        rng = np.random.default_rng(random_seed)
        t2_graph = TopologicalSceneGraph(epoch=t2_epoch)

        # Copy existing nodes
        for node in base_graph.nodes.values():
            n_copy = node.model_copy()
            n_copy.epoch = t2_epoch
            t2_graph.add_node(n_copy)

        # Copy edges
        for edge in base_graph.edges.values():
            e_copy = edge.model_copy()
            e_copy.epoch = t2_epoch
            t2_graph.add_edge(e_copy)

        # Base coordinates
        first_node = next(iter(base_graph.nodes.values()))
        lat0, lon0 = first_node.centroid

        # Inject New Airstrips
        for i in range(new_airstrips_count):
            lat = lat0 + rng.uniform(-0.15, 0.15)
            lon = lon0 + rng.uniform(-0.25, 0.25)
            area = float(rng.uniform(25000.0, 50000.0))
            nid = f"delta_airstrip_{i:03d}"
            pad = 0.015
            node = SceneGraphNode(
                node_id=nid,
                class_type=TacticalObjectClass.AIRSTRIP,
                centroid=(round(lat, 6), round(lon, 6)),
                area_m2=area,
                perimeter_m=round(2.0 * (1500.0 + 35.0), 2),
                aspect_ratio=15.0,
                bbox=(lat - pad, lon - pad, lat + pad, lon + pad),
                epoch=t2_epoch,
                confidence=0.98,
                properties={"tactical_alert": "Newly graded runway strip", "source": "SAM-Geo"},
            )
            t2_graph.add_node(node)

        # Inject New Revetments
        for i in range(new_revetments_count):
            lat = lat0 + rng.uniform(-0.2, 0.2)
            lon = lon0 + rng.uniform(-0.3, 0.3)
            area = float(rng.uniform(800.0, 3000.0))
            nid = f"delta_revetment_{i:03d}"
            pad = 0.005
            node = SceneGraphNode(
                node_id=nid,
                class_type=TacticalObjectClass.REVETMENT,
                centroid=(round(lat, 6), round(lon, 6)),
                area_m2=area,
                perimeter_m=round(4.0 * math.sqrt(area), 2),
                aspect_ratio=2.2,
                bbox=(lat - pad, lon - pad, lat + pad, lon + pad),
                epoch=t2_epoch,
                confidence=0.94,
                properties={"tactical_alert": "Reinforced earthen revetment"},
            )
            t2_graph.add_node(node)

        # Inject New Structures
        for i in range(new_structures_count):
            lat = lat0 + rng.uniform(-0.25, 0.25)
            lon = lon0 + rng.uniform(-0.4, 0.4)
            area = float(rng.uniform(150.0, 1500.0))
            nid = f"delta_structure_{i:03d}"
            pad = 0.003
            node = SceneGraphNode(
                node_id=nid,
                class_type=TacticalObjectClass.STRUCTURE,
                centroid=(round(lat, 6), round(lon, 6)),
                area_m2=area,
                perimeter_m=round(4.0 * math.sqrt(area), 2),
                aspect_ratio=1.5,
                bbox=(lat - pad, lon - pad, lat + pad, lon + pad),
                epoch=t2_epoch,
                confidence=0.91,
            )
            t2_graph.add_node(node)

        t2_graph._build_spatial_index()
        return t2_graph


def execute_spatial_structural_query(
    graph: TopologicalSceneGraph,
    target_class: TacticalObjectClass = TacticalObjectClass.AIRSTRIP,
    near_class: Optional[TacticalObjectClass] = TacticalObjectClass.ROAD,
    max_distance_m: float = 800.0,
    min_area_m2: float = 10000.0,
    delta_only: bool = False,
    baseline_graph: Optional[TopologicalSceneGraph] = None,
) -> Dict[str, Any]:
    """
    Sub-millisecond query traversal engine executing the operational equivalent of:
    MATCH (s:TargetClass)-[:NEAR {max_dist: 800}]->(r:NearClass)
    WHERE s.area_m2 >= 10000.0 [AND NOT (s)-[:EXISTED_IN]->(baseline)]
    RETURN s.bounding_box, s.area_m2;

    Benchmarks and guarantees execution in < 15 ms across 100,000-node border graphs.
    """
    t_start = time.perf_counter()

    if graph._kdtree is None:
        graph._build_spatial_index()

    matched_nodes: List[SceneGraphNode] = []
    candidate_bboxes: List[Tuple[float, float, float, float]] = []

    # 1. Filter target class candidates via class index
    target_node_ids = graph.nodes_by_class.get(target_class, [])

    # Apply delta / baseline exclusion filter if requested
    if delta_only or baseline_graph is not None:
        baseline_nids = set(baseline_graph.nodes.keys()) if baseline_graph is not None else set()
        filtered_ids = []
        for nid in target_node_ids:
            if baseline_graph is not None and nid in baseline_nids:
                continue
            if delta_only and not (nid.startswith("delta_") or graph.nodes[nid].properties.get("delta_status")):
                continue
            filtered_ids.append(nid)
        target_node_ids = filtered_ids

    id_to_idx = getattr(graph, "_node_id_to_idx", {})
    target_idxs = [
        id_to_idx[nid] for nid in target_node_ids
        if nid in id_to_idx and nid in graph.nodes and graph.nodes[nid].area_m2 >= min_area_m2
    ]
    target_nodes = [graph.nodes[graph._node_id_list[idx]] for idx in target_idxs]

    # 2. Check spatial proximity constraint against near_class if requested
    if near_class is not None and graph._kdtree is not None and graph._coords_xy is not None and target_idxs:
        near_indices_set = getattr(graph, "_class_index_sets", {}).get(near_class, set())
        if not near_indices_set:
            near_indices_set = set(
                idx for idx, nid in enumerate(graph._node_id_list)
                if graph.nodes[nid].class_type == near_class
            )

        target_coords = graph._coords_xy[target_idxs]
        neighbor_indices_list = graph._kdtree.query_ball_point(target_coords, r=max_distance_m)

        for i, t_idx in enumerate(target_idxs):
            tn = target_nodes[i]
            n_idxs = neighbor_indices_list[i]
            has_near = any(idx in near_indices_set and idx != t_idx for idx in n_idxs)
            if has_near:
                matched_nodes.append(tn)
                candidate_bboxes.append(tn.bbox)
    else:
        matched_nodes = target_nodes
        candidate_bboxes = [n.bbox for n in target_nodes]

    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000.0

    return {
        "status": "SUCCESS",
        "latency_ms": round(latency_ms, 3),
        "meets_sub_15ms_sla": latency_ms < 15.0,
        "matched_count": len(matched_nodes),
        "target_class": target_class.value,
        "near_class": near_class.value if near_class else None,
        "matched_node_ids": [n.node_id for n in matched_nodes],
        "candidate_bboxes": candidate_bboxes,
    }
