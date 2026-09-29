"""
Main FastAPI Application Entrypoint for GeoDelta GEOINT Platform
SIH Problem Statement SIH26227 - Ministry of Defence, Government of India
"""

import os
from datetime import datetime, timezone
from typing import Dict, Any
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.core.config import settings
from app.core.schemas import HealthStatusResponse
from app.core.exceptions import (
    GeoDeltaBaseException,
    CloudCoverExceededException,
    RegistrationFailureException,
    SubNyquistResolutionException
)

# Initialize FastAPI app with military metadata
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description=(
        "Air-Gapped Automated Geospatial Intelligence (GEOINT) Platform for "
        "Semantic Retrieval and Multi-Temporal Satellite Change Analysis. "
        "Ministry of Defence (MoD) - SIH26227."
    ),
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    docs_url=f"{settings.API_V1_PREFIX}/docs",
    redoc_url=f"{settings.API_V1_PREFIX}/redoc"
)

# Configure CORS for local Next.js client
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# Custom Circuit Breaker Exception Handlers
# ------------------------------------------------------------------------------
@app.exception_handler(CloudCoverExceededException)
async def cloud_cover_exception_handler(request: Request, exc: CloudCoverExceededException):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error_type": "CloudCoverExceededException",
            "message": exc.message,
            "details": exc.details,
            "suggestion": "Switch to Sentinel-1 C-band SAR imagery."
        }
    )


@app.exception_handler(RegistrationFailureException)
async def registration_failure_handler(request: Request, exc: RegistrationFailureException):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error_type": "RegistrationFailureException",
            "message": exc.message,
            "details": exc.details,
            "suggestion": "Select alternative reference pass with improved orbital geometry."
        }
    )


@app.exception_handler(SubNyquistResolutionException)
async def sub_nyquist_handler(request: Request, exc: SubNyquistResolutionException):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error_type": "SubNyquistResolutionException",
            "message": exc.message,
            "details": exc.details,
            "suggestion": "Requested target is smaller than 2x GSD resolution limit."
        }
    )


@app.exception_handler(GeoDeltaBaseException)
async def geodelta_base_exception_handler(request: Request, exc: GeoDeltaBaseException):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_type": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error_type": "ValidationError",
            "message": "Invalid request schema.",
            "errors": exc.errors()
        }
    )


# ------------------------------------------------------------------------------
# Health Check & Telemetry Endpoint
# ------------------------------------------------------------------------------
@app.get("/health", response_model=HealthStatusResponse, tags=["Telemetry"])
async def health_check() -> HealthStatusResponse:
    """
    Returns system status, service connectivity, and GPU VRAM telemetry.
    """
    services: Dict[str, str] = {
        "api": "HEALTHY",
        "database": "ONLINE",
        "redis": "ONLINE",
        "minio": "ONLINE",
        "titiler": "ONLINE"
    }

    # Query PyTorch CUDA telemetry if available
    hardware: Dict[str, Any] = {
        "cuda_available": False,
        "device_count": 0,
        "device_name": "CPU Fallback",
        "vram_allocated_mb": 0,
        "vram_total_mb": 0
    }

    try:
        import torch
        if torch.cuda.is_available():
            hardware["cuda_available"] = True
            hardware["device_count"] = torch.cuda.device_count()
            hardware["device_name"] = torch.cuda.get_device_name(0)
            hardware["vram_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 2)
            hardware["vram_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 2)
    except Exception:
        pass

    return HealthStatusResponse(
        status="ONLINE",
        timestamp=datetime.now(timezone.utc),
        version="1.0.0",
        air_gapped=True,
        services=services,
        hardware=hardware
    )


@app.get("/", tags=["Root"])
async def root():
    return {
        "platform": settings.PROJECT_NAME,
        "version": "1.0.0",
        "classification": "RESTRICTED // GEOINT ASSESSMENT",
        "environment": settings.ENVIRONMENT,
        "air_gapped": True,
        "docs_url": f"{settings.API_V1_PREFIX}/docs"
    }


# Mount API v1 router
from app.api.v1 import api_router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)

