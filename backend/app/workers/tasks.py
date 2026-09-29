"""
Celery Task Definitions for GeoDelta GEOINT Platform
Coordinates asynchronous pipeline execution:
Stage 0: Nyquist-Shannon Resolution Screening (Law 1)
Stage 1: COG Windowed Byte-Range Retrieval (FR-GEO-001)
Stage 2: Radiometric Percentile Normalization & Sub-pixel ECC Coregistration (Law 2, Law 3)
Stage 3: RemoteCLIP + Siamese Cross-Attention Forward Pass & Feature Caching (FR-INF-001, FR-NLQ-004)
Stage 4: Morphological Noise Suppression & PostGIS Vectorization (Phase 3)
Stage 5: Result Compilation & Storage
"""

import time
import os
import logging
from typing import Dict, Any, Optional
import torch

from app.workers.celery_app import celery_app
from app.core.schemas import BoundingBoxAOI, FeasibilityStatus
from app.core.guardrails import validate_nyquist_resolution, validate_cloud_cover
from app.core.exceptions import (
    SubNyquistResolutionException,
    CloudCoverExceededException,
    RegistrationFailureException,
    RasterIngestionException
)
from app.services.cog_streamer import COGStreamer
from app.services.normalizer import RadiometricNormalizer
from app.services.coregistration import SubPixelCoregistration
from app.services.vlm_encoder import RemoteCLIPTextEncoder
from app.services.siamese_engine import SiameseCrossAttentionEngine
from app.services.feature_cache import feature_cache
from app.services.vectorizer import extract_vector_features

logger = logging.getLogger(__name__)

DEFAULT_T1_PATH = "./storage/samples/sentinel2_t1_20250115.tif"
DEFAULT_T2_PATH = "./storage/samples/sentinel2_t2_20250610.tif"

# Model singletons to avoid re-instantiation overhead across calls
_VLM_ENCODER = None
_SIAMESE_MODEL = None


def get_vlm_encoder() -> RemoteCLIPTextEncoder:
    global _VLM_ENCODER
    if _VLM_ENCODER is None:
        _VLM_ENCODER = RemoteCLIPTextEncoder()
    return _VLM_ENCODER


def get_siamese_model() -> SiameseCrossAttentionEngine:
    global _SIAMESE_MODEL
    if _SIAMESE_MODEL is None:
        _SIAMESE_MODEL = SiameseCrossAttentionEngine(in_channels=4, embed_dim=512)
        _SIAMESE_MODEL.eval()
    return _SIAMESE_MODEL


def _safe_update_state(task_self: Any, state: str, meta: Dict[str, Any]) -> None:
    """Safely updates Celery task state, swallowing connection errors in offline/test mode."""
    if task_self is not None and hasattr(task_self, "update_state"):
        try:
            task_self.update_state(state=state, meta=meta)
        except Exception:
            pass


@celery_app.task(bind=True, name="app.workers.tasks.execute_inference_pipeline")
def execute_inference_pipeline(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main asynchronous worker task coordinating the pipeline execution stages.
    """
    req = getattr(self, "request", None)
    task_id = (getattr(req, "id", None) if req else None) or "standalone-exec-001"
    start_total_time = time.perf_counter()
    latencies: Dict[str, float] = {}

    query_text = request_payload.get("query_text", "")
    negative_query = request_payload.get("negative_query")
    aoi_data = request_payload.get("aoi", {
        "min_lat": 34.1300,
        "max_lat": 34.1500,
        "min_lon": 74.5750,
        "max_lon": 74.5950
    })
    aoi = BoundingBoxAOI(**aoi_data)
    t1_uri = request_payload.get("t1_uri", DEFAULT_T1_PATH)
    t2_uri = request_payload.get("t2_uri", DEFAULT_T2_PATH)

    logger.info(f"[{task_id}] Executing pipeline for query='{query_text}'")

    # --------------------------------------------------------------------------
    # Stage 0: Law 1 Nyquist-Shannon GSD Resolution Invariant Screening
    # --------------------------------------------------------------------------
    t0_stage = time.perf_counter()
    _safe_update_state(self, state="PROGRESS", meta={"stage": "FEASIBILITY_SCREENING", "percent": 10})
    validate_nyquist_resolution(query_text, native_gsd=10.0)
    latencies["feasibility_screening_ms"] = round((time.perf_counter() - t0_stage) * 1000.0, 2)

    # --------------------------------------------------------------------------
    # Stage 1: Windowed COG Streaming & Cryptographic Hashing (FR-GEO-001)
    # --------------------------------------------------------------------------
    t1_stage = time.perf_counter()
    _safe_update_state(self, state="PROGRESS", meta={"stage": "COG_STREAMING", "percent": 25})
    streamer = COGStreamer(default_bands=(1, 2, 3, 4))

    window_t1 = streamer.stream_window(t1_uri, aoi=aoi)
    window_t2 = streamer.stream_window(t2_uri, aoi=aoi)

    latencies["cog_streaming_ms"] = round((time.perf_counter() - t1_stage) * 1000.0, 2)

    # --------------------------------------------------------------------------
    # Stage 2: Radiometric Normalization & Sub-Pixel ECC Coregistration (Law 2, 3)
    # --------------------------------------------------------------------------
    t2_stage = time.perf_counter()
    _safe_update_state(self, state="PROGRESS", meta={"stage": "ECC_COREGISTRATION", "percent": 45})

    normalizer = RadiometricNormalizer(p_low=2.0, p_high=98.0)
    norm_t1_float, _ = normalizer.normalize(window_t1.data)
    norm_t2_float, _ = normalizer.normalize(window_t2.data)

    coreg = SubPixelCoregistration()
    aligned_t2_float, reg_metrics = coreg.align(norm_t1_float, norm_t2_float)

    latencies["normalization_and_coregistration_ms"] = round((time.perf_counter() - t2_stage) * 1000.0, 2)

    # --------------------------------------------------------------------------
    # Stage 3: RemoteCLIP + Siamese Cross-Attention Forward Pass (Phase 2)
    # --------------------------------------------------------------------------
    t3_stage = time.perf_counter()
    _safe_update_state(self, state="PROGRESS", meta={"stage": "CROSS_ATTENTION_INFERENCE", "percent": 70})

    vlm = get_vlm_encoder()
    model = get_siamese_model()

    # 1. Compute negative suppression vector e*
    e_star = vlm.encode_conditioned_prompt(query_text, negative_query=negative_query, beta=0.65)

    # 2. Convert to PyTorch tensors (1, 4, H, W)
    t1_tensor = torch.from_numpy(norm_t1_float).unsqueeze(0).to(torch.float32)
    t2_tensor = torch.from_numpy(aligned_t2_float).unsqueeze(0).to(torch.float32)

    # 3. Extract feature pyramids
    with torch.no_grad():
        _, _, f_delta_pyramid = model.extract_feature_pyramids(t1_tensor, t2_tensor)

        # 4. Cache feature pyramid for counter-factual re-queries (FR-NLQ-004)
        cache_key = feature_cache.generate_cache_key(str(aoi.model_dump()), str(t1_uri), str(t2_uri))
        feature_cache.put(
            cache_key,
            f_delta_pyramid,
            target_size=(window_t1.height, window_t1.width),
            transform=window_t1.transform
        )

        # 5. Modulate bottleneck and decode to probability map
        prob_map = model.forward_from_features(
            f_delta_pyramid, e_star, target_size=(window_t1.height, window_t1.width)
        )

    latencies["neural_inference_ms"] = round((time.perf_counter() - t3_stage) * 1000.0, 2)

    # --------------------------------------------------------------------------
    # Stage 4: Morphological Noise Suppression & PostGIS Vectorization (Phase 3)
    # --------------------------------------------------------------------------
    t4_stage = time.perf_counter()
    _safe_update_state(self, state="PROGRESS", meta={"stage": "VECTORIZATION", "percent": 90})

    confidence_threshold = float(request_payload.get("confidence_threshold", 0.70))
    detected_polygons = extract_vector_features(
        prob_map=prob_map,
        transform=window_t1.transform,
        threshold=confidence_threshold,
        tactical_class=query_text,
        min_area_sq_m=50.0
    )

    total_area_sq_m = round(sum(p.area_sq_meters for p in detected_polygons), 2)
    total_features = len(detected_polygons)
    latencies["vectorization_ms"] = round((time.perf_counter() - t4_stage) * 1000.0, 2)

    total_latency_ms = round((time.perf_counter() - start_total_time) * 1000.0, 2)

    # Optional DB Persistence for Audit & PostGIS storage
    try:
        from app.db.session import SessionLocal, init_db
        from app.db.models import DetectedChange, AuditLog
        init_db()
        db = SessionLocal()
        try:
            for p in detected_polygons:
                db_change = DetectedChange(
                    id=p.feature_id,
                    task_id=task_id,
                    tactical_class=p.tactical_class,
                    confidence=p.confidence,
                    area_sq_m=p.area_sq_meters,
                    area_hectares=p.area_hectares,
                    centroid_lat=p.centroid_wgs84[0],
                    centroid_lon=p.centroid_wgs84[1],
                    centroid_mgrs=p.centroid_mgrs,
                    geom_geojson=p.geometry_geojson
                )
                db.add(db_change)

            audit = AuditLog(
                task_id=task_id,
                t1_sha256=window_t1.sha256,
                t2_sha256=window_t2.sha256,
                prompt=query_text,
                negative_suppression=negative_query,
                polygons_detected=total_features,
                total_area_sq_m=total_area_sq_m,
                execution_time_ms=total_latency_ms
            )
            db.add(audit)
            db.commit()
        except Exception as db_err:
            logger.warning(f"DB persistence warning: {db_err}")
            db.rollback()
        finally:
            db.close()
    except Exception as exc:
        logger.warning(f"DB session init skipped: {exc}")

    # Broadcast completion
    _safe_update_state(self, state="SUCCESS", meta={"stage": "COMPLETED", "percent": 100})

    mean_prob = float(prob_map.mean().item())
    max_prob = float(prob_map.max().item())

    return {
        "task_id": task_id,
        "cache_key": cache_key,
        "status": FeasibilityStatus.COMPLETED.value,
        "stage": "COMPLETED",
        "ecc_metrics": reg_metrics.model_dump(),
        "t1_sha256": window_t1.sha256,
        "t2_sha256": window_t2.sha256,
        "native_gsd": window_t1.native_gsd,
        "raster_dims": [window_t1.height, window_t1.width],
        "probability_stats": {
            "mean": round(mean_prob, 4),
            "max": round(max_prob, 4)
        },
        "total_area_altered_sq_m": total_area_sq_m,
        "total_features_detected": total_features,
        "polygons": [p.model_dump() for p in detected_polygons],
        "stage_latencies_ms": latencies,
        "total_latency_ms": total_latency_ms,
        "message": f"Detected {total_features} tactical change polygon(s) covering {total_area_sq_m} m²."
    }


@celery_app.task(bind=True, name="app.workers.tasks.execute_counterfactual_pipeline")
def execute_counterfactual_pipeline(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sub-150ms Counter-Factual Query Execution Task (FR-NLQ-004).
    Re-runs ONLY cross-attention modulation and UNet++ decoder over cached feature pyramids.
    """
    req = getattr(self, "request", None)
    task_id = (getattr(req, "id", None) if req else None) or "counterfactual-exec-001"
    start_time = time.perf_counter()

    new_query_text = request_payload.get("new_query_text", "")
    new_negative_query = request_payload.get("new_negative_query")
    cache_key = request_payload.get("cache_key") or request_payload.get("task_id", "")

    # Law 1 screening
    validate_nyquist_resolution(new_query_text, native_gsd=10.0)

    cached_entry = feature_cache.get(cache_key)
    if cached_entry is None:
        raise ValueError(f"Cache miss for key '{cache_key}'. Feature pyramid must be cached first.")

    cached_pyramid, target_size = cached_entry

    vlm = get_vlm_encoder()
    model = get_siamese_model()

    e_star_new = vlm.encode_conditioned_prompt(new_query_text, negative_query=new_negative_query, beta=0.65)

    with torch.no_grad():
        prob_map = model.forward_from_features(cached_pyramid, e_star_new, target_size=target_size)

    cached_transform = feature_cache.get_transform(cache_key)
    confidence_threshold = float(request_payload.get("confidence_threshold", 0.70))
    detected_polygons = []
    if cached_transform is not None:
        detected_polygons = extract_vector_features(
            prob_map=prob_map,
            transform=cached_transform,
            threshold=confidence_threshold,
            tactical_class=new_query_text,
            min_area_sq_m=50.0
        )
    total_area_sq_m = round(sum(p.area_sq_meters for p in detected_polygons), 2)
    total_features = len(detected_polygons)
    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    return {
        "task_id": task_id,
        "cache_key": cache_key,
        "status": FeasibilityStatus.COMPLETED.value,
        "counter_factual": True,
        "execution_time_ms": latency_ms,
        "total_area_altered_sq_m": total_area_sq_m,
        "total_features_detected": total_features,
        "polygons": [p.model_dump() for p in detected_polygons],
        "probability_stats": {
            "mean": round(float(prob_map.mean().item()), 4),
            "max": round(float(prob_map.max().item()), 4)
        },
        "message": f"Counter-factual re-query completed in {latency_ms:.2f}ms."
    }

