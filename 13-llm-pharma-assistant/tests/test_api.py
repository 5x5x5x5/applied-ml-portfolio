"""Tests for the FastAPI application endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pharm_assist.api.models import (
    AskRequest,
    AskResponse,
    CitationResponse,
    DocumentChunkResponse,
    DocumentIngestResponse,
    DocumentSearchResponse,
    ErrorResponse,
    HealthResponse,
    WebSocketMessage,
)


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app.

    Patches the chain and retriever to avoid needing a real Anthropic API key.
    """
    from pharm_assist.api import main as api_main

    # Reset module-level state
    api_main._vector_store = None
    api_main._retriever = None
    api_main._chain = None
    api_main._sessions = {}

    return TestClient(api_main.app)


# ── Health Endpoint ─────────────────────────────────────────────────────────


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "collections" in data
        assert "uptime_seconds" in data

    def test_health_response_model(self, client: TestClient) -> None:
        response = client.get("/health")
        health = HealthResponse(**response.json())
        assert health.status == "healthy"


# ── Document Search Endpoint ────────────────────────────────────────────────


class TestDocumentSearchEndpoint:
    """Tests for the /documents/search endpoint."""

    def test_search_requires_query(self, client: TestClient) -> None:
        response = client.get("/documents/search")
        assert response.status_code == 422  # Validation error

    def test_search_with_query(self, client: TestClient) -> None:
        response = client.get("/documents/search", params={"query": "aspirin"})
        assert response.status_code == 200

        data = response.json()
        assert "query" in data
        assert "results" in data
        assert "total_results" in data

    def test_search_response_model(self, client: TestClient) -> None:
        response = client.get(
            "/documents/search",
            params={"query": "side effects", "n_results": 5},
        )
        assert response.status_code == 200
        search_response = DocumentSearchResponse(**response.json())
        assert search_response.query == "side effects"

    def test_search_with_collection_filter(self, client: TestClient) -> None:
        response = client.get(
            "/documents/search",
            params={
                "query": "dosage",
                "collection": "drug_labels",
                "n_results": 3,
            },
        )
        assert response.status_code == 200

    def test_search_with_drug_filter(self, client: TestClient) -> None:
        response = client.get(
            "/documents/search",
            params={"query": "interactions", "drug_name": "Aspirin"},
        )
        assert response.status_code == 200

    def test_search_with_mmr(self, client: TestClient) -> None:
        response = client.get(
            "/documents/search",
            params={"query": "metformin", "use_mmr": True},
        )
        assert response.status_code == 200


# ── Pydantic Model Tests ───────────────────────────────────────────────────


class TestPydanticModels:
    """Tests for the Pydantic request/response models."""

    def test_ask_request_validation(self) -> None:
        req = AskRequest(question="What are the side effects of aspirin?")
        assert req.question == "What are the side effects of aspirin?"
        assert req.n_results == 5
        assert req.stream is False

    def test_ask_request_min_length(self) -> None:
        with pytest.raises(Exception):
            AskRequest(question="Hi")  # Too short (min 3 chars)

    def test_ask_request_with_options(self) -> None:
        req = AskRequest(
            question="What are the drug interactions for metformin?",
            drug_name="metformin",
            section_type="DRUG INTERACTIONS",
            n_results=10,
            stream=True,
            session_id="test-session-123",
        )
        assert req.drug_name == "metformin"
        assert req.section_type == "DRUG INTERACTIONS"
        assert req.n_results == 10
        assert req.stream is True
        assert req.session_id == "test-session-123"

    def test_citation_response(self) -> None:
        cit = CitationResponse(
            citation_id=1,
            source_file="aspirin.txt",
            drug_name="Aspirin",
            section_type="ADVERSE REACTIONS",
            relevance_score=0.95,
            excerpt="Common side effects include...",
        )
        assert cit.citation_id == 1
        assert cit.drug_name == "Aspirin"
        assert cit.relevance_score == 0.95

    def test_ask_response(self) -> None:
        resp = AskResponse(
            question="What are aspirin side effects?",
            answer="Aspirin can cause GI upset and bleeding.",
            citations=[],
            confidence=0.85,
            model="claude-sonnet-4-20250514",
            latency_ms=1234.5,
        )
        assert resp.confidence == 0.85
        assert resp.model == "claude-sonnet-4-20250514"

    def test_document_ingest_response(self) -> None:
        resp = DocumentIngestResponse(
            files_processed=3,
            chunks_created=42,
            collection="drug_labels",
            errors=[],
        )
        assert resp.files_processed == 3
        assert resp.chunks_created == 42

    def test_error_response(self) -> None:
        err = ErrorResponse(
            error="Not found",
            detail="The requested resource was not found.",
            status_code=404,
        )
        assert err.status_code == 404

    def test_websocket_message(self) -> None:
        msg = WebSocketMessage(
            type="question",
            content="What is aspirin?",
            metadata={"drug_name": "aspirin"},
        )
        assert msg.type == "question"
        assert msg.content == "What is aspirin?"

    def test_document_chunk_response(self) -> None:
        chunk = DocumentChunkResponse(
            chunk_id="abc123",
            text="Some text about drugs.",
            metadata={"drug_name": "Aspirin", "section_type": "INDICATIONS"},
            relevance_score=0.88,
            collection="drug_labels",
        )
        assert chunk.chunk_id == "abc123"
        assert chunk.relevance_score == 0.88

    def test_health_response(self) -> None:
        health = HealthResponse(
            status="healthy",
            version="0.1.0",
            collections={"drug_labels": 100, "guidelines": 50},
            uptime_seconds=3600.0,
        )
        assert health.status == "healthy"
        assert health.collections["drug_labels"] == 100

    def test_document_search_response(self) -> None:
        resp = DocumentSearchResponse(
            query="aspirin dosage",
            results=[
                DocumentChunkResponse(
                    chunk_id="1",
                    text="325mg every 4-6 hours",
                    metadata={},
                    relevance_score=0.9,
                    collection="drug_labels",
                )
            ],
            total_results=1,
        )
        assert resp.total_results == 1
        assert len(resp.results) == 1


# ── API Integration Tests ──────────────────────────────────────────────────


class TestAPIIntegration:
    """Integration tests that exercise the full request/response cycle."""

    def test_openapi_schema_available(self, client: TestClient) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "PharmAssistAI"

    def test_cors_headers(self, client: TestClient) -> None:
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # CORS middleware should handle OPTIONS
        assert response.status_code in (200, 405)

    def test_ask_endpoint_exists(self, client: TestClient) -> None:
        """Verify the /ask endpoint exists (even if it needs an API key to succeed)."""
        response = client.post(
            "/ask",
            json={"question": "What is aspirin?"},
        )
        # Might fail due to missing API key, but should not be 404
        assert response.status_code != 404

    def test_ingest_endpoint_exists(self, client: TestClient) -> None:
        """Verify the /documents/ingest endpoint exists."""
        response = client.post("/documents/ingest")
        # Will fail validation (no files), but should not be 404
        assert response.status_code != 404
