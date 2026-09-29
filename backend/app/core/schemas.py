"""
Pydantic v2 Data Contracts for GeoDelta GEOINT Platform
Enforces strict typing, physical remote-sensing invariants, and zero-hallucination data exchange.
"""

from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class UserRole(str, Enum):
    ANALYST = "ROLE_ANALYST"
    COMMANDER = "ROLE_COMMANDER"
    ADMIN = "ROLE_ADMIN"


class FeasibilityStatus(str, Enum):
    FEASIBLE = "FEASIBLE"
    COMPLETED = "COMPLETED"
    REJECTED_SUB_NYQUIST = "REJECTED_SUB_NYQUIST"
    REJECTED_CLOUD_COVER = "REJECTED_CLOUD_COVER"
    REJECTED_REGISTRATION_FAILURE = "REJECTED_REGISTRATION_FAILURE"
    FAILED = "FAILED"


class BoundingBoxAOI(BaseModel):
    min_lat: float = Field(..., ge=-90.0, le=90.0, description="Southern latitude boundary")
    max_lat: float = Field(..., ge=-90.0, le=90.0, description="Northern latitude boundary")
    min_lon: float = Field(..., ge=-180.0, le=180.0, description="Western longitude boundary")
    max_lon: float = Field(..., ge=-180.0, le=180.0, description="Eastern longitude boundary")

    @field_validator("max_lat")
    def validate_latitude_order(cls, v: float, values: Any) -> float:
        if hasattr(values, "data") and "min_lat" in values.data and v <= values.data["min_lat"]:
            raise ValueError("max_lat must be strictly greater than min_lat")
        return v

    @field_validator("max_lon")
    def validate_longitude_order(cls, v: float, values: Any) -> float:
        if hasattr(values, "data") and "min_lon" in values.data and v <= values.data["min_lon"]:
            raise ValueError("max_lon must be strictly greater than min_lon")
        return v


class InferenceRequest(BaseModel):
    query_text: str = Field(..., min_length=3, max_length=256, description="Natural language target query")
    negative_query: Optional[str] = Field(None, max_length=256, description="Negative context for suppression")
    aoi: BoundingBoxAOI = Field(..., description="Geographic boundary of the target AOI")
    t1_date: datetime = Field(..., description="Pre-event satellite acquisition timestamp")
    t2_date: datetime = Field(..., description="Post-event satellite acquisition timestamp")
    confidence_threshold: float = Field(default=0.70, ge=0.30, le=0.95, description="Initial mask threshold tau")
    cloud_threshold_percent: float = Field(default=35.0, ge=5.0, le=50.0, description="Maximum permitted cloud ratio")


class CounterFactualRequest(BaseModel):
    task_id: str = Field(..., description="Original inference job ID referencing cached visual tensors")
    new_query_text: str = Field(..., min_length=3, max_length=256, description="New contrasting query prompt")
    new_negative_query: Optional[str] = Field(None, max_length=256, description="Updated negative context")
    confidence_threshold: float = Field(default=0.70, ge=0.30, le=0.95, description="Threshold tau")


class RegistrationMetrics(BaseModel):
    ecc_score: float = Field(..., ge=0.0, le=1.0, description="Enhanced Correlation Coefficient score")
    warp_matrix: List[List[float]] = Field(..., description="2x3 Affine transformation matrix")
    registration_converged: bool = Field(..., description="Flag indicating ECC convergence")


class DetectedPolygonFeature(BaseModel):
    feature_id: str = Field(..., description="Unique UUID for detected polygon")
    tactical_class: str = Field(..., description="Predicted tactical entity classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Mean confidence score within polygon boundary")
    area_sq_meters: float = Field(..., ge=50.0, description="Ellipsoidal ground footprint in square meters")
    area_hectares: float = Field(..., ge=0.005, description="Ground footprint in hectares")
    centroid_wgs84: Tuple[float, float] = Field(..., description="(Latitude, Longitude) of centroid")
    centroid_mgrs: str = Field(..., description="MGRS 10-figure grid coordinate")
    geometry_geojson: Dict[str, Any] = Field(..., description="Valid GeoJSON MultiPolygon dictionary")


class ExecutionTrace(BaseModel):
    task_id: str = Field(..., description="Unique Celery job ID")
    execution_time_ms: float = Field(..., description="Total end-to-end latency in milliseconds")
    stage_latencies_ms: Dict[str, float] = Field(..., description="Breakdown of timing per pipeline stage")
    device_used: str = Field(..., description="Execution device (e.g. 'NVIDIA GeForce RTX 4060' or 'CPU')")
    execution_mode: str = Field(..., description="'DEEP_LEARNING_SIAMESE' or 'DETERMINISTIC_FALLBACK'")
    t1_sha256: str = Field(..., description="SHA-256 cryptographic hash of t1 source GeoTIFF bytes")
    t2_sha256: str = Field(..., description="SHA-256 cryptographic hash of t2 source GeoTIFF bytes")


class InferenceResponse(BaseModel):
    task_id: str = Field(..., description="Job identifier")
    status: FeasibilityStatus = Field(..., description="Feasibility and completion status")
    total_area_altered_sq_m: float = Field(..., description="Aggregate altered ground footprint area")
    total_features_detected: int = Field(..., description="Count of discrete vector change polygons")
    polygons: List[DetectedPolygonFeature] = Field(default_factory=list, description="List of detected georeferenced polygons")
    trace: Optional[ExecutionTrace] = Field(None, description="Auditable telemetry and execution trace")
    message: Optional[str] = Field(None, description="Operational advisory message")


class DossierExportRequest(BaseModel):
    task_id: str = Field(..., description="Completed inference job ID")
    analyst_callsign: str = Field(default="ANALYST-ALPHA-7", max_length=64)
    classification_level: str = Field(default="RESTRICTED // GEOINT ASSESSMENT")
    notes: Optional[str] = Field(None, max_length=1000)


class HealthStatusResponse(BaseModel):
    status: str = Field("ONLINE", description="System health status")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = Field("1.0.0")
    air_gapped: bool = Field(True, description="Air-gapped operation flag")
    services: Dict[str, str] = Field(..., description="Sub-service health checks")
    hardware: Dict[str, Any] = Field(..., description="GPU and CPU telemetry")
