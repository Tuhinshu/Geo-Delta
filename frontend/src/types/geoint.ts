/**
 * Frontend TypeScript Data Contracts for GeoDelta GEOINT Platform
 * Strictly aligned with Backend Pydantic Schemas (FR-UI-001 through FR-UI-004)
 */

export interface BoundingBoxAOI {
  min_lat: number;
  max_lat: number;
  min_lon: number;
  max_lon: number;
}

export interface DetectedPolygonFeature {
  feature_id: string;
  tactical_class: string;
  confidence: number;
  area_sq_meters: number;
  area_hectares: number;
  centroid_wgs84: [number, number]; // [Latitude, Longitude]
  centroid_mgrs: string;
  geometry_geojson: {
    type: 'Polygon' | 'MultiPolygon';
    coordinates: any;
  };
}

export interface ECCMetrics {
  ecc_score: number;
  warp_matrix: number[][];
  registration_converged: boolean;
}

export interface InferenceResponse {
  task_id: string;
  cache_key?: string;
  status: 'FEASIBLE' | 'COMPLETED' | 'REJECTED_SUB_NYQUIST' | 'REJECTED_CLOUD_COVER' | 'REJECTED_REGISTRATION_FAILURE' | 'FAILED';
  total_area_altered_sq_m: number;
  total_features_detected: number;
  polygons: DetectedPolygonFeature[];
  ecc_metrics?: ECCMetrics;
  stage_latencies_ms?: {
    cog_streaming_ms?: number;
    radiometric_normalization_ms?: number;
    ecc_coregistration_ms?: number;
    neural_inference_ms?: number;
    vectorization_ms?: number;
  };
  total_latency_ms?: number;
  message?: string;
  counter_factual?: boolean;
}

export interface PipelineStageInfo {
  stage: 'IDLE' | 'COG_STREAMING' | 'ECC_ALIGNMENT' | 'SIAMESE_INFERENCE' | 'VECTORIZATION' | 'COMPLETED';
  percent: number;
  currentTask?: string;
}
