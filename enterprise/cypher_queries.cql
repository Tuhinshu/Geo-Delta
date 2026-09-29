// ==============================================================================
// GeoDelta Enterprise Scaling Architecture (Solution D - Production)
// Curated Cypher Query Library for Sub-Millisecond Border Reconnaissance
// Target Database: Neo4j Enterprise / Community 5.x + Neo4j Spatial / APOC / GDS
// Standard Compliance: SIH26227 GEOINT Specifications
// ==============================================================================

// ------------------------------------------------------------------------------
// SECTION 0: GRAPH SCHEMA CONSTRAINTS & SPATIAL INDEXES
// ------------------------------------------------------------------------------

// Unique identification constraints
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (e:SpatialEntity) REQUIRE e.id IS UNIQUE;

CREATE CONSTRAINT epoch_id_unique IF NOT EXISTS
FOR (ep:Epoch) REQUIRE ep.date IS UNIQUE;

// Core B-tree & Spatial Point indexes for sub-millisecond lookups
CREATE INDEX entity_epoch_idx IF NOT EXISTS
FOR (e:SpatialEntity) ON (e.epoch);

CREATE INDEX entity_class_idx IF NOT EXISTS
FOR (e:SpatialEntity) ON (e.class);

CREATE INDEX entity_area_idx IF NOT EXISTS
FOR (e:SpatialEntity) ON (e.area_m2);

CREATE POINT INDEX entity_spatial_idx IF NOT EXISTS
FOR (e:SpatialEntity) ON (e.location);


// ------------------------------------------------------------------------------
// QUERY 1: NEWLY PAVED RUNWAY EXTENSION / AIRSTRIP NEAR BORDER ROAD
// Natural Language: "Identify newly paved airstrips or runway extensions within 800m of roads"
// Operational Goal: Detect tactical air mobility expansions without scanning 50,000 km^2
// ------------------------------------------------------------------------------
MATCH (a:Airstrip:SpatialEntity)
WHERE a.epoch = $target_epoch
  AND NOT (a)-[:EXISTED_IN]->(:Epoch {date: $baseline_epoch})
  AND a.area_m2 >= 15000.0
MATCH (r:Road:SpatialEntity {epoch: $target_epoch})
WHERE point.distance(a.location, r.location) <= 800.0
RETURN 
    a.id AS airstrip_id,
    a.area_m2 AS runway_area_m2,
    a.location.latitude AS lat,
    a.location.longitude AS lon,
    a.bbox AS candidate_bbox,
    r.id AS nearest_road_id,
    point.distance(a.location, r.location) AS road_proximity_m
ORDER BY a.area_m2 DESC;


// ------------------------------------------------------------------------------
// QUERY 2: EMERGING FORWARD STRUCTURAL CLUSTER WITHIN 500M OF INFRASTRUCTURE
// Natural Language: "Detect clusters of newly emerged buildings near logistical roads"
// Operational Goal: Uncover clandestine base construction before full operational deployment
// ------------------------------------------------------------------------------
MATCH (s:Structure:SpatialEntity)
WHERE s.epoch = $target_epoch
  AND NOT (s)-[:EXISTED_IN]->(:Epoch {date: $baseline_epoch})
MATCH (r:Road:SpatialEntity {epoch: $target_epoch})
WHERE point.distance(s.location, r.location) <= 500.0
WITH s, r, point.distance(s.location, r.location) AS dist_to_road
MATCH (neighbor:Structure:SpatialEntity {epoch: $target_epoch})
WHERE s <> neighbor 
  AND point.distance(s.location, neighbor.location) <= 300.0
WITH s, dist_to_road, count(neighbor) AS cluster_density
WHERE cluster_density >= 3
RETURN 
    s.id AS cluster_core_id,
    s.location.latitude AS lat,
    s.location.longitude AS lon,
    s.bbox AS candidate_bbox,
    cluster_density,
    dist_to_road
ORDER BY cluster_density DESC;


// ------------------------------------------------------------------------------
// QUERY 3: FORTIFIED REVETMENT & ARTILLERY BERM EMERGENCE
// Natural Language: "Find newly excavated earthen revetments and defensive berms"
// Operational Goal: Isolate forward hardening & ammunition storage revetments
// ------------------------------------------------------------------------------
MATCH (rev:Revetment:SpatialEntity)
WHERE rev.epoch = $target_epoch
  AND NOT (rev)-[:EXISTED_IN]->(:Epoch {date: $baseline_epoch})
OPTIONAL MATCH (rev)-[:DEFENDS]->(target:SpatialEntity)
RETURN 
    rev.id AS revetment_id,
    rev.area_m2 AS surface_area_m2,
    rev.aspect_ratio AS elongation,
    rev.location.latitude AS lat,
    rev.location.longitude AS lon,
    rev.bbox AS candidate_bbox,
    target.id AS defended_target_id,
    target.class AS defended_class;


// ------------------------------------------------------------------------------
// QUERY 4: ROAD NETWORK TOPOLOGY EXPANSION (BRIDGING TO FORWARD POSTS)
// Natural Language: "Identify newly built connector roads linking to forward posts"
// Operational Goal: Track logistical supply line progression toward border frontiers
// ------------------------------------------------------------------------------
MATCH (r:Road:SpatialEntity)-[:CONNECTED_BY_ROAD]->(p:MilitaryPost:SpatialEntity)
WHERE r.epoch = $target_epoch
  AND NOT (r)-[:EXISTED_IN]->(:Epoch {date: $baseline_epoch})
RETURN 
    r.id AS road_segment_id,
    r.area_m2 AS road_footprint_m2,
    p.id AS connected_post_id,
    p.location.latitude AS post_lat,
    p.location.longitude AS post_lon,
    r.bbox AS candidate_bbox;


// ------------------------------------------------------------------------------
// QUERY 5: REGIONAL CANDIDATE BOUNDING BOX EXTRACTION FOR SOLUTION B DISPATCH
// Operational Goal: Screen 50,000 km^2 border corridor in <15 ms and output isolated
// candidate bounding boxes (10km x 10km chips) for Solution B GPU deep learning verification
// ------------------------------------------------------------------------------
MATCH (delta:SpatialEntity)
WHERE delta.epoch = $target_epoch
  AND NOT (delta)-[:EXISTED_IN]->(:Epoch {date: $baseline_epoch})
  AND delta.class IN ['Airstrip', 'Revetment', 'Structure', 'GradedArea', 'MilitaryPost']
WITH delta.bbox AS bbox, delta.class AS tactical_class, delta.area_m2 AS area
RETURN 
    bbox[0] AS min_lat,
    bbox[1] AS min_lon,
    bbox[2] AS max_lat,
    bbox[3] AS max_lon,
    tactical_class,
    area
ORDER BY area DESC
LIMIT 50;


// ------------------------------------------------------------------------------
// QUERY 6: HIGH-RISK BUFFER ZONE VIOLATION QUERY
// Natural Language: "Identify all unauthorized structural activity inside the 5km buffer zone"
// ------------------------------------------------------------------------------
MATCH (zone:BufferZone {zone_name: 'BorderDemarcationSector4'})
MATCH (e:SpatialEntity)
WHERE e.epoch = $target_epoch
  AND NOT (e)-[:EXISTED_IN]->(:Epoch {date: $baseline_epoch})
  AND point.distance(e.location, zone.boundary_center) <= zone.radius_m
RETURN 
    e.id AS entity_id,
    e.class AS classification,
    e.area_m2 AS altered_footprint_m2,
    point.distance(e.location, zone.boundary_center) AS distance_to_border_m,
    e.bbox AS candidate_bbox
ORDER BY altered_footprint_m2 DESC;


// ------------------------------------------------------------------------------
// QUERY 7: TEMPORAL EPOCH GRAPH DELTA AGGREGATION GROUPED BY SECTOR
// Operational Goal: Provide theater commanders with macro tactical change summaries
// ------------------------------------------------------------------------------
MATCH (e:SpatialEntity {epoch: $target_epoch})
WHERE NOT (e)-[:EXISTED_IN]->(:Epoch {date: $baseline_epoch})
WITH e.class AS tactical_class,
     count(e) AS entity_count,
     sum(e.area_m2) AS total_altered_m2,
     avg(e.confidence) AS mean_confidence
RETURN 
    tactical_class,
    entity_count,
    round(total_altered_m2, 2) AS total_area_m2,
    round(total_altered_m2 / 10000.0, 4) AS total_area_hectares,
    round(mean_confidence, 3) AS avg_model_confidence
ORDER BY total_area_m2 DESC;
