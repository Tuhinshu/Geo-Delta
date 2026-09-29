"""
Integration test for execute_inference_pipeline task under Phase 1.
"""

import pytest
from app.workers.tasks import execute_inference_pipeline
from app.core.exceptions import SubNyquistResolutionException


def test_pipeline_task_success():
    """
    Test end-to-end execution of Phase 1 ingestion, normalization, and coregistration pipeline.
    """
    payload = {
        "query_text": "Identify newly paved runway extension",
        "aoi": {
            "min_lat": 34.1300,
            "max_lat": 34.1500,
            "min_lon": 74.5750,
            "max_lon": 74.5950
        }
    }

    result = execute_inference_pipeline.run(payload)
    assert result["status"] == "COMPLETED"
    assert "ecc_metrics" in result
    assert result["ecc_metrics"]["registration_converged"] is True
    assert len(result["t1_sha256"]) == 64
    assert len(result["t2_sha256"]) == 64
    assert result["total_latency_ms"] > 0.0


def test_pipeline_task_sub_nyquist_rejection():
    """
    Test that pipeline immediately aborts if query contains sub-Nyquist target (Law 1).
    """
    payload = {
        "query_text": "Detect individual vehicle on roadway",
        "aoi": {
            "min_lat": 34.1300,
            "max_lat": 34.1500,
            "min_lon": 74.5750,
            "max_lon": 74.5950
        }
    }

    with pytest.raises(SubNyquistResolutionException):
        execute_inference_pipeline.run(payload)
