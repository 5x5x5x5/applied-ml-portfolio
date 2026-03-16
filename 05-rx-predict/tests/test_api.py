"""API endpoint tests for RxPredict."""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_root_endpoint(async_client: AsyncClient) -> None:
    """Test root endpoint returns API info."""
    response = await async_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "RxPredict"
    assert "version" in data
    assert "endpoints" in data
    assert data["latency_target_ms"] == 100


@pytest.mark.asyncio
async def test_health_endpoint(async_client: AsyncClient) -> None:
    """Test health check returns comprehensive status."""
    response = await async_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded", "unhealthy")
    assert "model_loaded" in data
    assert "redis_connected" in data
    assert "uptime_seconds" in data
    assert "performance" in data


@pytest.mark.asyncio
async def test_metrics_endpoint(async_client: AsyncClient) -> None:
    """Test Prometheus metrics endpoint."""
    response = await async_client.get("/metrics")
    assert response.status_code == 200
    assert "rxpredict_" in response.text


@pytest.mark.asyncio
async def test_predict_valid_request(
    async_client: AsyncClient, sample_prediction_request: dict[str, Any]
) -> None:
    """Test single prediction with valid input."""
    response = await async_client.post("/predict", json=sample_prediction_request)
    assert response.status_code == 200

    data = response.json()
    assert "request_id" in data
    assert "response_probability" in data
    assert 0 <= data["response_probability"] <= 1
    assert "confidence_lower" in data
    assert "confidence_upper" in data
    assert data["confidence_lower"] <= data["response_probability"]
    assert data["response_probability"] <= data["confidence_upper"]
    assert data["predicted_class"] in [
        "poor_response",
        "partial_response",
        "good_response",
        "excellent_response",
    ]
    assert data["risk_level"] in ["high_risk", "moderate_risk", "low_risk", "minimal_risk"]
    assert "inference_time_ms" in data
    assert "model_version" in data
    assert "X-Request-ID" in response.headers
    assert "X-Response-Time" in response.headers


@pytest.mark.asyncio
async def test_predict_poor_metabolizer(
    async_client: AsyncClient, poor_metabolizer_patient: dict[str, Any]
) -> None:
    """Test prediction for poor metabolizer patient."""
    response = await async_client.post("/predict", json=poor_metabolizer_patient)
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["response_probability"] <= 1
    assert data["predicted_class"] in [
        "poor_response",
        "partial_response",
        "good_response",
        "excellent_response",
    ]


@pytest.mark.asyncio
async def test_predict_minimal_input(async_client: AsyncClient) -> None:
    """Test prediction with minimal required fields."""
    minimal = {
        "demographics": {
            "age": 30,
            "weight_kg": 70.0,
            "height_cm": 175.0,
        },
        "drug": {
            "name": "Aspirin",
            "dosage_mg": 100.0,
        },
    }
    response = await async_client.post("/predict", json=minimal)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data


@pytest.mark.asyncio
async def test_predict_invalid_age(async_client: AsyncClient) -> None:
    """Test validation rejects invalid age."""
    payload = {
        "demographics": {
            "age": 200,  # Invalid
            "weight_kg": 70.0,
            "height_cm": 175.0,
        },
        "drug": {
            "name": "Aspirin",
            "dosage_mg": 100.0,
        },
    }
    response = await async_client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_missing_drug(async_client: AsyncClient) -> None:
    """Test validation requires drug info."""
    payload = {
        "demographics": {
            "age": 30,
            "weight_kg": 70.0,
            "height_cm": 175.0,
        },
    }
    response = await async_client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_batch_predict(
    async_client: AsyncClient, sample_prediction_request: dict[str, Any]
) -> None:
    """Test batch prediction endpoint."""
    payload = {
        "patients": [sample_prediction_request, sample_prediction_request],
    }
    response = await async_client.post("/batch-predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["batch_size"] == 2
    assert len(data["predictions"]) == 2
    assert "total_inference_time_ms" in data
    assert "avg_inference_time_ms" in data

    for pred in data["predictions"]:
        assert "response_probability" in pred
        assert "predicted_class" in pred
        assert "risk_level" in pred


@pytest.mark.asyncio
async def test_batch_predict_empty(async_client: AsyncClient) -> None:
    """Test batch prediction rejects empty list."""
    payload = {"patients": []}
    response = await async_client.post("/batch-predict", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_request_id_propagation(async_client: AsyncClient) -> None:
    """Test that X-Request-ID is propagated."""
    custom_id = "test-request-12345"
    payload = {
        "demographics": {"age": 30, "weight_kg": 70.0, "height_cm": 175.0},
        "drug": {"name": "Test", "dosage_mg": 50.0},
    }
    response = await async_client.post(
        "/predict",
        json=payload,
        headers={"X-Request-ID": custom_id},
    )
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_id


@pytest.mark.asyncio
async def test_response_time_header(
    async_client: AsyncClient, sample_prediction_request: dict[str, Any]
) -> None:
    """Test that response time header is present and reasonable."""
    response = await async_client.post("/predict", json=sample_prediction_request)
    assert response.status_code == 200
    response_time = response.headers.get("X-Response-Time", "")
    assert response_time.endswith("ms")
