"""Tests for the CloudGenomics FastAPI application."""

from __future__ import annotations

import io

import pytest
from httpx import ASGITransport, AsyncClient

from cloud_genomics.api.main import app
from cloud_genomics.pipeline.vcf_processor import create_sample_vcf


@pytest.fixture
async def client() -> AsyncClient:
    """Create an async test client for the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestHealthEndpoint:
    """Tests for GET /health."""

    @pytest.mark.asyncio
    async def test_health_check(self, client: AsyncClient) -> None:
        """Health endpoint should return 200 with status info."""
        response = await client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "version" in data
        assert data["version"] == "1.0.0"
        assert "model_loaded" in data
        assert "dependencies" in data
        assert "uptime_seconds" in data
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_health_model_status(self, client: AsyncClient) -> None:
        """Health should report model loaded after startup."""
        response = await client.get("/health")
        data = response.json()

        assert data["model_loaded"] is True
        assert data["status"] == "healthy"
        assert data["model_accuracy"] is not None
        assert data["model_accuracy"] > 0.0


class TestClassifyVariantEndpoint:
    """Tests for POST /classify-variant."""

    @pytest.mark.asyncio
    async def test_classify_basic_variant(self, client: AsyncClient) -> None:
        """Should classify a basic variant request."""
        payload = {
            "chrom": "chr17",
            "pos": 7675088,
            "ref": "G",
            "alt": "A",
            "sift_score": 0.01,
            "polyphen2_score": 0.99,
            "cadd_phred": 35.0,
            "revel_score": 0.9,
            "gnomad_af": 0.0,
            "consequence": "missense",
        }
        response = await client.post("/classify-variant", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "variant_id" in data
        assert "classification" in data
        assert data["classification"] in [
            "benign",
            "likely_benign",
            "VUS",
            "likely_pathogenic",
            "pathogenic",
        ]
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1
        assert "class_probabilities" in data
        assert len(data["class_probabilities"]) == 5
        assert "explanation" in data
        assert len(data["explanation"]) > 0
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_classify_with_chrom_normalization(self, client: AsyncClient) -> None:
        """Chromosome should be normalized to 'chr' prefix."""
        payload = {
            "chrom": "17",
            "pos": 100,
            "ref": "A",
            "alt": "G",
        }
        response = await client.post("/classify-variant", json=payload)
        assert response.status_code == 200
        assert response.json()["chrom"] == "chr17"

    @pytest.mark.asyncio
    async def test_classify_invalid_allele(self, client: AsyncClient) -> None:
        """Invalid allele characters should return 422."""
        payload = {
            "chrom": "chr1",
            "pos": 100,
            "ref": "Z",  # invalid
            "alt": "A",
        }
        response = await client.post("/classify-variant", json=payload)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_classify_invalid_chromosome(self, client: AsyncClient) -> None:
        """Invalid chromosome should return 422."""
        payload = {
            "chrom": "chr99",
            "pos": 100,
            "ref": "A",
            "alt": "G",
        }
        response = await client.post("/classify-variant", json=payload)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_classify_invalid_position(self, client: AsyncClient) -> None:
        """Position <= 0 should return 422."""
        payload = {
            "chrom": "chr1",
            "pos": 0,
            "ref": "A",
            "alt": "G",
        }
        response = await client.post("/classify-variant", json=payload)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_classify_score_ranges(self, client: AsyncClient) -> None:
        """Scores outside valid ranges should return 422."""
        payload = {
            "chrom": "chr1",
            "pos": 100,
            "ref": "A",
            "alt": "G",
            "gnomad_af": 5.0,  # must be <= 1
        }
        response = await client.post("/classify-variant", json=payload)
        assert response.status_code == 422


class TestGetVariantEndpoint:
    """Tests for GET /variant/{variant_id}."""

    @pytest.mark.asyncio
    async def test_get_classified_variant(self, client: AsyncClient) -> None:
        """Should retrieve a previously classified variant."""
        # First classify a variant
        payload = {
            "chrom": "chr1",
            "pos": 12345,
            "ref": "A",
            "alt": "G",
        }
        classify_response = await client.post("/classify-variant", json=payload)
        assert classify_response.status_code == 200
        variant_id = classify_response.json()["variant_id"]

        # Then retrieve it
        get_response = await client.get(f"/variant/{variant_id}")
        assert get_response.status_code == 200
        data = get_response.json()
        assert data["variant_id"] == variant_id
        assert data["chrom"] == "chr1"

    @pytest.mark.asyncio
    async def test_get_nonexistent_variant(self, client: AsyncClient) -> None:
        """Should return 404 for unknown variant ID."""
        response = await client.get("/variant/nonexistent123")
        assert response.status_code == 404


class TestUploadVCFEndpoint:
    """Tests for POST /upload-vcf."""

    @pytest.mark.asyncio
    async def test_upload_vcf(self, client: AsyncClient) -> None:
        """Should process an uploaded VCF file."""
        vcf_content = create_sample_vcf()
        files = {"file": ("test.vcf", io.BytesIO(vcf_content.encode()), "text/plain")}
        response = await client.post("/upload-vcf", files=files)
        assert response.status_code == 200

        data = response.json()
        assert "job_id" in data
        assert data["status"] == "completed"
        assert data["variants_processed"] > 0
        assert data["variants_passed_filters"] > 0
        assert len(data["results"]) > 0
        assert data["processing_time_seconds"] > 0

    @pytest.mark.asyncio
    async def test_upload_vcf_results_format(self, client: AsyncClient) -> None:
        """VCF upload results should have correct format."""
        vcf_content = create_sample_vcf()
        files = {"file": ("test.vcf", io.BytesIO(vcf_content.encode()), "text/plain")}
        response = await client.post("/upload-vcf", files=files)

        data = response.json()
        for result in data["results"]:
            assert "variant_id" in result
            assert "classification" in result
            assert "confidence" in result
            assert "class_probabilities" in result
            assert "explanation" in result

    @pytest.mark.asyncio
    async def test_upload_invalid_file_type(self, client: AsyncClient) -> None:
        """Non-VCF file should be rejected."""
        files = {"file": ("test.txt", io.BytesIO(b"not a vcf"), "text/plain")}
        response = await client.post("/upload-vcf", files=files)
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_upload_classified_variants_retrievable(self, client: AsyncClient) -> None:
        """Variants from VCF upload should be retrievable via GET."""
        vcf_content = create_sample_vcf()
        files = {"file": ("test.vcf", io.BytesIO(vcf_content.encode()), "text/plain")}
        upload_response = await client.post("/upload-vcf", files=files)
        data = upload_response.json()

        if data["results"]:
            variant_id = data["results"][0]["variant_id"]
            get_response = await client.get(f"/variant/{variant_id}")
            assert get_response.status_code == 200


class TestRequestValidation:
    """Tests for request validation and edge cases."""

    @pytest.mark.asyncio
    async def test_missing_required_fields(self, client: AsyncClient) -> None:
        """Missing required fields should return 422."""
        payload = {"chrom": "chr1"}  # missing pos, ref, alt
        response = await client.post("/classify-variant", json=payload)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_body(self, client: AsyncClient) -> None:
        """Empty request body should return 422."""
        response = await client.post("/classify-variant", json={})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_nonexistent_endpoint(self, client: AsyncClient) -> None:
        """Non-existent endpoint should return 404."""
        response = await client.get("/nonexistent")
        assert response.status_code in (404, 405)
