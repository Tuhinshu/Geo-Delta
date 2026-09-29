import os
import pytest
from fastapi.testclient import TestClient

os.environ["SYNC_INFERENCE"] = "true"
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "GeoDelta" in data["platform"]
    assert data["air_gapped"] is True


def test_health_telemetry_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ONLINE"
    assert "services" in data
    assert "hardware" in data


def test_submit_analysis_sub_nyquist_rejection():
    """
    Law 1 Circuit Breaker: Sub-Nyquist queries must be rejected with HTTP 400.
    """
    payload = {
        "query_text": "Detect individual vehicle crossing road",
        "aoi": {
            "min_lat": 34.1300,
            "max_lat": 34.1500,
            "min_lon": 74.5750,
            "max_lon": 74.5950
        },
        "t1_date": "2025-01-15T00:00:00Z",
        "t2_date": "2025-06-10T00:00:00Z",
        "confidence_threshold": 0.70,
        "cloud_threshold_percent": 35.0
    }
    response = client.post("/api/v1/inference/analyze", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["error_type"] == "SubNyquistResolutionException"
    assert "below the Nyquist-Shannon limit" in data["message"]


def test_submit_analysis_feasible_query():
    """
    Law 1: Resolvable queries pass screening and execute ingestion/coregistration pipeline.
    """
    payload = {
        "query_text": "Identify newly paved runway extension",
        "aoi": {
            "min_lat": 34.1300,
            "max_lat": 34.1500,
            "min_lon": 74.5750,
            "max_lon": 74.5950
        },
        "t1_date": "2025-01-15T00:00:00Z",
        "t2_date": "2025-06-10T00:00:00Z",
        "confidence_threshold": 0.70,
        "cloud_threshold_percent": 35.0
    }
    response = client.post("/api/v1/inference/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] in ["COMPLETED", "QUEUED"]


def test_counterfactual_endpoint_flow():
    """
    FR-NLQ-004: Tests counter-factual endpoint re-querying over cached pyramids in <150ms.
    """
    init_payload = {
        "query_text": "Identify newly paved runway extension",
        "aoi": {
            "min_lat": 34.1300,
            "max_lat": 34.1500,
            "min_lon": 74.5750,
            "max_lon": 74.5950
        },
        "t1_date": "2025-01-15T00:00:00Z",
        "t2_date": "2025-06-10T00:00:00Z",
        "confidence_threshold": 0.70,
        "cloud_threshold_percent": 35.0
    }
    init_res = client.post("/api/v1/inference/analyze", json=init_payload)
    assert init_res.status_code == 200
    init_data = init_res.json()
    cache_key = init_data["cache_key"]

    cf_payload = {
        "task_id": cache_key,
        "new_query_text": "Show perimeter bunker fortification",
        "new_negative_query": "seasonal agricultural harvesting",
        "confidence_threshold": 0.70
    }
    cf_res = client.post("/api/v1/inference/counterfactual", json=cf_payload)
    assert cf_res.status_code == 200
    cf_data = cf_res.json()
    assert cf_data["counter_factual"] is True
    assert cf_data["execution_time_ms"] < 150.0

    # Non-existent cache key test -> 404
    bad_payload = {
        "task_id": "non_existent_key_12345",
        "new_query_text": "Show perimeter fortification"
    }
    bad_res = client.post("/api/v1/inference/counterfactual", json=bad_payload)
    assert bad_res.status_code == 404

