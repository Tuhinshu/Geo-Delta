"""
Test Suite for GeoDelta Enterprise Scaling Architecture (Solution D - Phase 7).
Verifies:
1. SAM-Geo instance vector primitive extraction (GeoJSON and binary mask).
2. Delaunay Triangulation proximity network generation.
3. Topological Scene Graph Delta compilation (Delta_G = G_t2 \\ G_t1).
4. Sub-millisecond indexed spatial graph traversals (< 15 ms across 100,000 nodes).
5. Cypher transaction generation for Neo4j.
"""

import time
import numpy as np
import pytest
from affine import Affine

from enterprise.sam_extractor import (
    SAMPrimitiveExtractor,
    SceneGraphEdge,
    SceneGraphNode,
    TacticalObjectClass,
    haversine_distance_m,
    latlon_to_cartesian_meters,
)
from enterprise.graph_compiler import (
    GraphDelta,
    GraphDeltaCompiler,
    SyntheticBorderGraphGenerator,
    TopologicalSceneGraph,
    execute_spatial_structural_query,
)


@pytest.fixture
def sample_geojson():
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": "feat_airstrip_01",
                    "tactical_class": "Airstrip",
                    "confidence": 0.98,
                    "area_sq_m": 45000.0,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [75.000, 34.000],
                        [75.030, 34.000],
                        [75.030, 34.002],
                        [75.000, 34.002],
                        [75.000, 34.000],
                    ]],
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "id": "feat_road_01",
                    "tactical_class": "Road",
                    "confidence": 0.94,
                    "area_sq_m": 8000.0,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [75.005, 34.003],
                        [75.025, 34.003],
                        [75.025, 34.004],
                        [75.005, 34.004],
                        [75.005, 34.003],
                    ]],
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "id": "feat_revetment_01",
                    "tactical_class": "Revetment",
                    "confidence": 0.91,
                    "area_sq_m": 1200.0,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [75.010, 34.006],
                        [75.013, 34.006],
                        [75.013, 34.008],
                        [75.010, 34.008],
                        [75.010, 34.006],
                    ]],
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "id": "feat_small_clutter",
                    "tactical_class": "Structure",
                    "confidence": 0.50,
                    "area_sq_m": 15.0,  # Below min_area_m2 filter (25.0)
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [75.050, 34.050],
                        [75.0501, 34.050],
                        [75.0501, 34.0501],
                        [75.050, 34.0501],
                        [75.050, 34.050],
                    ]],
                },
            },
        ],
    }


def test_sam_primitive_extraction_from_geojson(sample_geojson):
    extractor = SAMPrimitiveExtractor(default_epoch="2025-01-15", min_area_m2=25.0)
    nodes = extractor.extract_primitives_from_geojson(sample_geojson, epoch="2025-01-15")

    # Clutter feature (<25 m^2) should be excluded
    assert len(nodes) == 3

    node_dict = {n.node_id: n for n in nodes}
    assert "feat_airstrip_01" in node_dict
    assert "feat_road_01" in node_dict
    assert "feat_revetment_01" in node_dict

    airstrip = node_dict["feat_airstrip_01"]
    assert airstrip.class_type == TacticalObjectClass.AIRSTRIP
    assert airstrip.area_m2 == 45000.0
    assert airstrip.centroid[0] == pytest.approx(34.001, abs=0.001)
    assert airstrip.centroid[1] == pytest.approx(75.015, abs=0.001)
    assert airstrip.aspect_ratio >= 1.0


def test_sam_primitive_extraction_from_mask():
    extractor = SAMPrimitiveExtractor(default_epoch="2025-01-15", min_area_m2=50.0)

    # Synthetic binary mask: 100x100 raster with two distinct blobs
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:30] = 1  # 20x20 = 400 pixels (40,000 m^2 at 10m GSD)
    mask[60:70, 50:90] = 1  # 10x40 = 400 pixels, high aspect ratio

    # Transform: 10m pixel size at 34.0 N, 75.0 E
    transform = Affine.translation(75.0, 34.0) * Affine.scale(0.0001, -0.0001)

    nodes = extractor.extract_primitives_from_mask(mask, transform, epoch="2025-01-15", gsd_m=10.0)
    assert len(nodes) == 2

    for node in nodes:
        assert node.area_m2 == pytest.approx(40000.0, rel=0.05)
        assert node.epoch == "2025-01-15"
        assert len(node.bbox) == 4


def test_delaunay_proximity_network_construction():
    extractor = SAMPrimitiveExtractor(max_proximity_m=2000.0)

    # Create 5 synthetic nodes in a spatial cluster
    nodes = [
        SceneGraphNode(
            node_id="n1",
            class_type=TacticalObjectClass.ROAD,
            centroid=(34.000, 75.000),
            area_m2=5000.0,
            perimeter_m=1000.0,
            aspect_ratio=6.0,
            bbox=(33.999, 74.999, 34.001, 75.001),
            epoch="2025-01-15",
        ),
        SceneGraphNode(
            node_id="n2",
            class_type=TacticalObjectClass.STRUCTURE,
            centroid=(34.003, 75.002),
            area_m2=800.0,
            perimeter_m=120.0,
            aspect_ratio=1.2,
            bbox=(34.002, 75.001, 34.004, 75.003),
            epoch="2025-01-15",
        ),
        SceneGraphNode(
            node_id="n3",
            class_type=TacticalObjectClass.REVETMENT,
            centroid=(34.005, 75.004),
            area_m2=1500.0,
            perimeter_m=180.0,
            aspect_ratio=2.0,
            bbox=(34.004, 75.003, 34.006, 75.005),
            epoch="2025-01-15",
        ),
        SceneGraphNode(
            node_id="n4",
            class_type=TacticalObjectClass.TOWER,
            centroid=(34.002, 75.005),
            area_m2=50.0,
            perimeter_m=28.0,
            aspect_ratio=1.0,
            bbox=(34.001, 75.004, 34.003, 75.006),
            epoch="2025-01-15",
        ),
        SceneGraphNode(
            node_id="n5_distant",
            class_type=TacticalObjectClass.STRUCTURE,
            centroid=(34.050, 75.050),  # ~7 km away, exceeding max_proximity_m
            area_m2=1000.0,
            perimeter_m=130.0,
            aspect_ratio=1.5,
            bbox=(34.049, 75.049, 34.051, 75.051),
            epoch="2025-01-15",
        ),
    ]

    edges = extractor.build_proximity_network(nodes, max_distance_m=1500.0)

    # Distant node should not have edges connected to n1-n4
    for e in edges:
        assert not (e.source_id == "n5_distant" or e.target_id == "n5_distant")

    # Clustered nodes should have edges
    assert len(edges) >= 3

    # Check that road relationship was properly assigned
    road_edges = [e for e in edges if e.source_id == "n1" or e.target_id == "n1"]
    assert any(e.relation_type == "CONNECTED_BY_ROAD" for e in road_edges)


def test_topological_scene_graph_spatial_index():
    graph = TopologicalSceneGraph(epoch="2025-01-15")

    for i in range(20):
        lat = 34.0 + (i * 0.005)
        lon = 75.0 + (i * 0.005)
        node = SceneGraphNode(
            node_id=f"node_{i:02d}",
            class_type=TacticalObjectClass.STRUCTURE if i % 2 == 0 else TacticalObjectClass.TOWER,
            centroid=(lat, lon),
            area_m2=500.0 + (i * 50.0),
            perimeter_m=100.0,
            aspect_ratio=1.0,
            bbox=(lat - 0.001, lon - 0.001, lat + 0.001, lon + 0.001),
            epoch="2025-01-15",
        )
        graph.add_node(node)

    # Query within 1 km of node_00
    results = graph.query_radius(34.0, 75.0, radius_m=1000.0)
    assert len(results) >= 1
    assert any(n.node_id == "node_00" for n in results)

    # Query by bounding box
    bbox_nodes = graph.query_bbox((33.99, 74.99, 34.015, 75.015))
    assert len(bbox_nodes) >= 3

    # Query by class
    towers = graph.query_by_class(TacticalObjectClass.TOWER)
    assert len(towers) == 10


def test_graph_delta_compilation():
    compiler = GraphDeltaCompiler(match_radius_m=35.0, area_change_threshold=0.20)

    # Baseline G_t1
    g1 = TopologicalSceneGraph(epoch="2025-01-15")
    n1 = SceneGraphNode(
        node_id="base_s1",
        class_type=TacticalObjectClass.STRUCTURE,
        centroid=(34.010, 75.010),
        area_m2=1000.0,
        perimeter_m=130.0,
        aspect_ratio=1.2,
        bbox=(34.009, 75.009, 34.011, 75.011),
        epoch="2025-01-15",
    )
    n2 = SceneGraphNode(
        node_id="base_s2_to_be_demolished",
        class_type=TacticalObjectClass.STRUCTURE,
        centroid=(34.020, 75.020),
        area_m2=800.0,
        perimeter_m=115.0,
        aspect_ratio=1.0,
        bbox=(34.019, 75.019, 34.021, 75.021),
        epoch="2025-01-15",
    )
    g1.add_node(n1)
    g1.add_node(n2)

    # Post-event G_t2
    g2 = TopologicalSceneGraph(epoch="2025-06-15")
    # n1 expanded by 60%
    n1_expanded = SceneGraphNode(
        node_id="target_s1",
        class_type=TacticalObjectClass.STRUCTURE,
        centroid=(34.01002, 75.01002),  # ~2 meters shift
        area_m2=1600.0,  # 60% increase
        perimeter_m=170.0,
        aspect_ratio=1.3,
        bbox=(34.008, 75.008, 34.012, 75.012),
        epoch="2025-06-15",
    )
    # New airstrip
    n_new_airstrip = SceneGraphNode(
        node_id="target_airstrip_new",
        class_type=TacticalObjectClass.AIRSTRIP,
        centroid=(34.030, 75.030),
        area_m2=35000.0,
        perimeter_m=3200.0,
        aspect_ratio=12.0,
        bbox=(34.025, 75.025, 34.035, 75.035),
        epoch="2025-06-15",
    )
    g2.add_node(n1_expanded)
    g2.add_node(n_new_airstrip)

    delta = compiler.compile_delta(g1, g2)

    assert delta.t1_epoch == "2025-01-15"
    assert delta.t2_epoch == "2025-06-15"

    # Added node check
    assert len(delta.added_nodes) == 1
    assert delta.added_nodes[0].node_id == "target_airstrip_new"

    # Removed node check
    assert len(delta.removed_nodes) == 1
    assert delta.removed_nodes[0].node_id == "base_s2_to_be_demolished"

    # Modified node check
    assert len(delta.modified_nodes) == 1
    assert delta.modified_nodes[0]["old_node_id"] == "base_s1"
    assert delta.modified_nodes[0]["area_delta_m2"] == pytest.approx(600.0, abs=1.0)

    # Candidate bounding boxes generated for Solution B dispatch
    assert len(delta.candidate_bboxes) >= 1


def test_export_cypher_transactions():
    compiler = GraphDeltaCompiler()
    graph = TopologicalSceneGraph(epoch="2025-01-15")

    n1 = SceneGraphNode(
        node_id="node_cypher_01",
        class_type=TacticalObjectClass.AIRSTRIP,
        centroid=(34.0, 75.0),
        area_m2=20000.0,
        perimeter_m=2000.0,
        aspect_ratio=10.0,
        bbox=(33.99, 74.99, 34.01, 75.01),
        epoch="2025-01-15",
    )
    graph.add_node(n1)

    cql = compiler.export_cypher_script(graph)

    # Check key Cypher statements
    assert "CREATE CONSTRAINT" in cql
    assert "CREATE POINT INDEX" in cql
    assert "MERGE (n:SpatialEntity:Airstrip" in cql
    assert ":begin" in cql
    assert ":commit" in cql


def test_sub_15ms_100k_node_border_graph_traversal():
    """
    CRITICAL BENCHMARK (Phase 7 SLA):
    Generates a 100,000-node synthetic border corridor scene graph across 50,000 km^2
    and executes indexed spatial-structural query traversal.
    Target SLA: Latency strictly < 15.0 ms!
    """
    print("\nSynthesizing 100,000-node border corridor scene graph...")
    t0 = time.perf_counter()
    graph = SyntheticBorderGraphGenerator.generate_border_corridor_graph(
        node_count=100000,
        epoch="2025-01-15",
        corridor_length_km=500.0,
        corridor_width_km=100.0,
        random_seed=42,
    )
    t_gen = (time.perf_counter() - t0) * 1000.0
    print(f"Generated 100,000 nodes in {t_gen:.2f} ms")

    assert len(graph.nodes) == 100000

    # Inject 5 new airstrips near roads
    print("Injecting tactical delta targets...")
    t2_graph = SyntheticBorderGraphGenerator.inject_tactical_deltas(
        graph,
        t2_epoch="2025-06-15",
        new_structures_count=30,
        new_airstrips_count=5,
        new_revetments_count=15,
        random_seed=77,
    )

    # Execute operational Cypher-equivalent query
    print("Executing sub-millisecond query across 100,000+ nodes...")
    res = execute_spatial_structural_query(
        t2_graph,
        target_class=TacticalObjectClass.AIRSTRIP,
        near_class=TacticalObjectClass.ROAD,
        max_distance_m=1500.0,
        min_area_m2=20000.0,
        delta_only=True,
        baseline_graph=graph,
    )

    print(f"Query Result: Latency = {res['latency_ms']} ms | Matched = {res['matched_count']}")

    # Verification of SLA (< 15.0 ms invariant)
    assert res["status"] == "SUCCESS"
    assert res["meets_sub_15ms_sla"] is True
    assert res["latency_ms"] < 15.0
    assert res["matched_count"] > 0
    assert len(res["candidate_bboxes"]) == res["matched_count"]
