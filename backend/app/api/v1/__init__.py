"""
API v1 Router Configuration for GeoDelta GEOINT Platform
"""

from fastapi import APIRouter
from app.api.v1.endpoints.inference import router as inference_router
from app.api.v1.endpoints.dossier import router as dossier_router

api_router = APIRouter()
api_router.include_router(inference_router)
api_router.include_router(dossier_router)

__all__ = ["api_router"]

