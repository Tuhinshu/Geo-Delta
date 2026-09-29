"""
Inference & GEOINT Analysis Endpoints for GeoDelta API v1
Enforces Law 1 Nyquist validation, dispatches ingestion, coregistration, and Siamese inference.
Provides sub-150ms counter-factual query evaluation over cached feature pyramids.
"""

import os
import uuid
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status

from app.core.config import settings
from app.core.schemas import InferenceRequest, CounterFactualRequest, FeasibilityStatus
from app.core.guardrails import validate_nyquist_resolution
from app.workers.tasks import execute_inference_pipeline, execute_counterfactual_pipeline

router = APIRouter(prefix="/inference", tags=["GEOINT Analysis"])


@router.post(
    "/analyze",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Submit Natural Language GEOINT Analysis Query"
)
async def submit_analysis(request: InferenceRequest) -> Dict[str, Any]:
    """
    Submits a natural language query for multi-temporal change detection.
    Pre-screens query against Law 1 Nyquist-Shannon GSD Resolution Invariant.
    Dispatches COG streaming, radiometric normalization, ECC, and Siamese cross-attention inference.
    """
    # 1. Law 1 Guardrail Screening (pre-inference circuit breaker)
    validate_nyquist_resolution(request.query_text, native_gsd=10.0)

    # 2. Serialize payload
    payload = request.model_dump()
    payload["t1_date"] = request.t1_date.isoformat()
    payload["t2_date"] = request.t2_date.isoformat()

    # 3. Execute pipeline
    # In test mode or when Redis is absent, execute directly without blocking
    if os.environ.get("SYNC_INFERENCE", "false").lower() == "true":
        task_id = f"job-{uuid.uuid4().hex[:8]}"
        payload_with_id = payload.copy()
        payload_with_id["task_id"] = task_id
        return execute_inference_pipeline.run(payload_with_id)

    try:
        async_result = execute_inference_pipeline.apply_async(args=[payload], retry=False)
        return {
            "task_id": async_result.id,
            "status": "QUEUED",
            "message": "Analysis job dispatched to GPU worker queue."
        }
    except Exception:
        # Fallback to direct synchronous execution if broker is not connected
        task_id = f"job-{uuid.uuid4().hex[:8]}"
        payload_with_id = payload.copy()
        payload_with_id["task_id"] = task_id
        return execute_inference_pipeline.run(payload_with_id)


@router.post(
    "/counterfactual",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Submit Counter-Factual Query (<150ms Re-query)"
)
async def submit_counterfactual(request: CounterFactualRequest) -> Dict[str, Any]:
    """
    Evaluates a new semantic prompt on cached visual feature pyramids in < 150ms (FR-NLQ-004).
    """
    payload = request.model_dump()
    try:
        return execute_counterfactual_pipeline.run(payload)
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(err)
        )


@router.get(
    "/status/{task_id}",
    response_model=Dict[str, Any],
    summary="Check Analysis Task Status"
)
async def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Polls the status of an asynchronous inference task.
    """
    try:
        res = execute_inference_pipeline.AsyncResult(task_id)
        return {
            "task_id": task_id,
            "status": res.status,
            "result": res.result if res.ready() else None
        }
    except Exception as exc:
        return {
            "task_id": task_id,
            "status": "UNKNOWN",
            "message": str(exc)
        }
