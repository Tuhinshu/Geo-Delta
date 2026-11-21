"""
Intelligence Dossier & Vector Export API Endpoints (FR-EXP-001, FR-EXP-002)
Generates publication-grade military intelligence dossiers in PDF format and standard GIS vector exports.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional, Tuple
from fastapi import APIRouter, HTTPException, Response, status
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from app.services.dossier_generator import DossierGenerator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/dossier", tags=["Dossier & Exports"])


def _decode_base64_raster(b64_str: Optional[str]) -> Optional[np.ndarray]:
    """Decodes a base64 string or data URL into a uint8 RGB numpy array."""
    if not b64_str or not isinstance(b64_str, str):
        return None
    try:
        # Strip data URL prefix if present
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        raw_bytes = base64.b64decode(b64_str.strip())
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return np.array(img, dtype=np.uint8)
    except Exception as exc:
        logger.warning(f"Failed to decode base64 image for dossier export: {exc}")
        return None


class DossierExportRequest(BaseModel):
    task_id: str = Field(..., description="Unique task or analysis identifier")
    query_prompt: str = Field(..., description="Natural language tactical query")
    negative_prompt: Optional[str] = Field(None, description="Negative semantic suppression context")
    acquisition_dates: Optional[Tuple[str, str]] = Field(
        default=("2025-01-15", "2025-06-10"),
        description="Pair of pre-event and post-event acquisition dates (t1, t2)"
    )
    aoi_bounds: Optional[Dict[str, float]] = Field(
        default=None,
        description="Bounding box coordinates: min_lat, max_lat, min_lon, max_lon"
    )
    total_area_altered_sq_m: float = Field(0.0, ge=0.0, description="Total altered surface area in square meters")
    total_area_altered_ha: float = Field(0.0, ge=0.0, description="Total altered surface area in hectares")
    polygons: List[Dict[str, Any]] = Field(default_factory=list, description="List of detected polygon features")
    ecc_metrics: Optional[Dict[str, Any]] = Field(default=None, description="Sub-pixel ECC coregistration metrics")
    analyst_callsign: str = Field("OFFICER-IN-CHARGE-GEODELTA", description="Authenticated analyst military call-sign")
    image_t1_base64: Optional[str] = Field(None, description="Base64 or data URL for Pre-Event optical imagery")
    image_t2_base64: Optional[str] = Field(None, description="Base64 or data URL for Post-Event optical imagery")


@router.post(
    "/export",
    summary="1-Click Publication-Grade Cryptographic Intelligence Dossier (PDF)",
    description="Compiles and returns a binary PDF stream containing classification ribbons, optical chips, quantitative metrics, and SHA-256 evidence chain-of-custody.",
    response_class=Response,
)
async def export_intelligence_dossier(request: DossierExportRequest) -> Response:
    try:
        raster_t1 = _decode_base64_raster(request.image_t1_base64)
        raster_t2 = _decode_base64_raster(request.image_t2_base64)

        generator = DossierGenerator()
        pdf_bytes = generator.generate_dossier(
            task_id=request.task_id,
            query_prompt=request.query_prompt,
            negative_prompt=request.negative_prompt,
            acquisition_dates=request.acquisition_dates,
            aoi_bounds=request.aoi_bounds,
            total_area_altered_sq_m=request.total_area_altered_sq_m,
            total_area_altered_ha=request.total_area_altered_ha,
            polygons=request.polygons,
            raster_t1=raster_t1,
            raster_t2=raster_t2,
            ecc_metrics=request.ecc_metrics,
            analyst_callsign=request.analyst_callsign,
        )

        filename = f"geodelta_dossier_{request.task_id}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "X-Dossier-Task-Id": request.task_id,
                "X-Dossier-Classification": "RESTRICTED",
            },
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dossier compilation failed: {str(exc)}",
        )


@router.post(
    "/export-geojson",
    summary="Export Detected Change Vectors as Standard GeoJSON",
    description="Returns compliant OGC GeoJSON FeatureCollection of all detected change polygons.",
    response_model=Dict[str, Any],
)
async def export_geojson_features(request: DossierExportRequest) -> Dict[str, Any]:
    generator = DossierGenerator()
    return generator.export_geojson(request.polygons, request.task_id)


@router.post(
    "/export-shapefile-zip",
    summary="Export Change Vectors as Compressed GIS Archive",
    description="Returns in-memory zip archive with GeoJSON and cryptographic chain-of-custody manifest.",
    response_class=Response,
)
async def export_shapefile_zip(request: DossierExportRequest) -> Response:
    generator = DossierGenerator()
    zip_bytes = generator.export_shapefile_zip(request.polygons, request.task_id)
    filename = f"geodelta_vectors_{request.task_id}.zip"
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
        },
    )
