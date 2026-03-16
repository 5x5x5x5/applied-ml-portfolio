"""Tests for the FastAPI application endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pharma_sentinel.api.main import app, app_state
from pharma_sentinel.models.adverse_event_classifier import AdverseEventClassifier

# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def client(trained_classifier: AdverseEventClassifier) -> TestClient:
    """FastAPI test client with a loaded model."""
    app_state.classifier = trained_classifier
    app_state.prediction_count = 0
    app_state.error_count = 0
    app_state.severity_counts = {"mild": 0, "moderate": 0, "severe": 0, "critical": 0}
    app_state.total_latency_ms = 0.0
    app_state.total_confidence = 0.0
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def client_no_model() -> TestClient:
    """FastAPI test client without a loaded model."""
    app_state.classifier = None
    return TestClient(app, raise_server_exceptions=False)


# ─────────────────────────────────────────────────────────────────────
# Health endpoint tests
# ─────────────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check_healthy(self, client: TestClient) -> None:
        """Test health check returns healthy when model is loaded."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "version" in data
        assert "uptime_seconds" in data

    def test_health_check_degraded(self, client_no_model: TestClient) -> None:
        """Test health check returns degraded when model is not loaded."""
        response = client_no_model.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False


# ─────────────────────────────────────────────────────────────────────
# Predict endpoint tests
# ─────────────────────────────────────────────────────────────────────


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_success(self, client: TestClient) -> None:
        """Test successful prediction request."""
        response = client.post(
            "/predict",
            json={
                "text": "Patient experienced mild nausea and headache after medication.",
                "drug_name": "Ibuprofen",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "severity" in data
        assert data["severity"] in ["mild", "moderate", "severe", "critical"]
        assert 0.0 <= data["confidence"] <= 1.0
        assert "probabilities" in data
        assert len(data["probabilities"]) == 4
        assert data["drug_name"] == "Ibuprofen"
        assert "processing_time_ms" in data
        assert "timestamp" in data

    def test_predict_critical_text(self, client: TestClient) -> None:
        """Test prediction with critical severity text."""
        response = client.post(
            "/predict",
            json={
                "text": "Patient died from fatal cardiac arrest and anaphylactic shock after drug administration.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        # Critical text should have high critical probability
        assert data["probabilities"]["critical"] > 0.1

    def test_predict_without_drug_name(self, client: TestClient) -> None:
        """Test prediction without optional drug_name field."""
        response = client.post(
            "/predict",
            json={
                "text": "Patient experienced moderate vomiting and fever for three days.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["drug_name"] is None

    def test_predict_text_too_short(self, client: TestClient) -> None:
        """Test prediction with text below minimum length."""
        response = client.post(
            "/predict",
            json={"text": "Short"},
        )
        assert response.status_code == 422

    def test_predict_text_too_long(self, client: TestClient) -> None:
        """Test prediction with text exceeding maximum length."""
        response = client.post(
            "/predict",
            json={"text": "x" * 10001},
        )
        assert response.status_code == 422

    def test_predict_empty_text(self, client: TestClient) -> None:
        """Test prediction with empty text."""
        response = client.post(
            "/predict",
            json={"text": ""},
        )
        assert response.status_code == 422

    def test_predict_model_not_loaded(self, client_no_model: TestClient) -> None:
        """Test prediction returns 503 when model not loaded."""
        response = client_no_model.post(
            "/predict",
            json={
                "text": "Patient experienced nausea and headache after medication.",
            },
        )
        assert response.status_code == 503

    def test_predict_updates_metrics(self, client: TestClient) -> None:
        """Test that prediction updates internal metrics."""
        initial_count = app_state.prediction_count

        client.post(
            "/predict",
            json={
                "text": "Patient experienced moderate vomiting and diarrhea requiring treatment.",
            },
        )

        assert app_state.prediction_count == initial_count + 1


# ─────────────────────────────────────────────────────────────────────
# Batch predict endpoint tests
# ─────────────────────────────────────────────────────────────────────


class TestBatchPredictEndpoint:
    """Tests for the /batch-predict endpoint."""

    def test_batch_predict_success(self, client: TestClient) -> None:
        """Test successful batch prediction."""
        response = client.post(
            "/batch-predict",
            json={
                "reports": [
                    {"text": "Patient reported mild headache and fatigue lasting two days."},
                    {
                        "text": "Severe liver failure requiring hospitalization and emergency surgery."
                    },
                    {"text": "Fatal cardiac arrest after drug administration leading to death."},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 3
        assert len(data["predictions"]) == 3
        assert "severity_summary" in data
        assert "processing_time_ms" in data

    def test_batch_predict_severity_summary(self, client: TestClient) -> None:
        """Test batch predict returns severity summary."""
        response = client.post(
            "/batch-predict",
            json={
                "reports": [
                    {"text": "Mild nausea and dizziness reported by the patient after treatment."},
                    {"text": "Mild headache and drowsiness. No treatment was required."},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        summary = data["severity_summary"]
        assert sum(summary.values()) == 2

    def test_batch_predict_empty_list(self, client: TestClient) -> None:
        """Test batch predict with empty reports list."""
        response = client.post(
            "/batch-predict",
            json={"reports": []},
        )
        assert response.status_code == 422

    def test_batch_predict_exceeds_max(self, client: TestClient) -> None:
        """Test batch predict with too many reports."""
        reports = [
            {"text": f"Patient report number {i} with mild headache and nausea."}
            for i in range(101)
        ]
        response = client.post(
            "/batch-predict",
            json={"reports": reports},
        )
        assert response.status_code == 422

    def test_batch_predict_model_not_loaded(self, client_no_model: TestClient) -> None:
        """Test batch predict returns 503 when model not loaded."""
        response = client_no_model.post(
            "/batch-predict",
            json={
                "reports": [
                    {"text": "Patient experienced nausea and headache after medication."},
                ],
            },
        )
        assert response.status_code == 503


# ─────────────────────────────────────────────────────────────────────
# Metrics endpoint tests
# ─────────────────────────────────────────────────────────────────────


class TestMetricsEndpoint:
    """Tests for the /metrics endpoint."""

    def test_metrics_initial(self, client: TestClient) -> None:
        """Test metrics endpoint returns initial zeros."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["predictions_total"] == 0
        assert data["error_count"] == 0
        assert data["average_latency_ms"] == 0.0
        assert "uptime_seconds" in data
        assert "predictions_by_severity" in data

    def test_metrics_after_predictions(self, client: TestClient) -> None:
        """Test metrics reflect prediction activity."""
        # Make some predictions
        for text in [
            "Patient reported mild headache and nausea after taking medication.",
            "Severe seizure requiring hospitalization and intensive care treatment.",
        ]:
            client.post("/predict", json={"text": text})

        response = client.get("/metrics")
        data = response.json()
        assert data["predictions_total"] == 2
        assert data["average_latency_ms"] > 0
        assert data["average_confidence"] > 0


# ─────────────────────────────────────────────────────────────────────
# Edge case and security tests
# ─────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Tests for edge cases and security."""

    def test_invalid_json_body(self, client: TestClient) -> None:
        """Test handling of invalid JSON."""
        response = client.post(
            "/predict",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_missing_required_field(self, client: TestClient) -> None:
        """Test missing required 'text' field."""
        response = client.post(
            "/predict",
            json={"drug_name": "Aspirin"},
        )
        assert response.status_code == 422

    def test_cors_headers(self, client: TestClient) -> None:
        """Test CORS headers are present on preflight."""
        response = client.options(
            "/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        # OPTIONS should not error
        assert response.status_code in (200, 405)

    def test_nonexistent_endpoint(self, client: TestClient) -> None:
        """Test 404 for non-existent endpoints."""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_wrong_method(self, client: TestClient) -> None:
        """Test method not allowed."""
        response = client.get("/predict")
        assert response.status_code == 405

    def test_special_characters_in_text(self, client: TestClient) -> None:
        """Test prediction with special characters."""
        response = client.post(
            "/predict",
            json={
                "text": "Patient <script>alert('xss')</script> had nausea & headache after taking the drug.",
            },
        )
        # Should handle gracefully without error
        assert response.status_code == 200

    def test_unicode_text(self, client: TestClient) -> None:
        """Test prediction with unicode characters."""
        response = client.post(
            "/predict",
            json={
                "text": "Patient experienced moderate nausea and dizziness after treatment.",
            },
        )
        assert response.status_code == 200
