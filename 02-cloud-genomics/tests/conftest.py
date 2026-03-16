"""Shared test fixtures for CloudGenomics test suite."""

from __future__ import annotations

from typing import Any

import pytest

from cloud_genomics.models.variant_classifier import (
    VariantClass,
    VariantClassifier,
    VariantFeatures,
    generate_synthetic_training_data,
)
from cloud_genomics.monitoring.metrics import MetricsCollector, MetricsConfig
from cloud_genomics.pipeline.vcf_processor import (
    QualityThresholds,
    VCFParser,
    VCFProcessor,
    create_sample_vcf,
)
from cloud_genomics.security.encryption import (
    AuditLogger,
    FieldLevelEncryption,
    GenomicDataEncryption,
)


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set test environment variables."""
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")


# ---------------------------------------------------------------------------
# ML model fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def synthetic_training_data() -> tuple[list[VariantFeatures], list[VariantClass]]:
    """Generate synthetic training data (session-scoped for reuse)."""
    return generate_synthetic_training_data(n_samples=200, random_state=42)


@pytest.fixture(scope="session")
def trained_classifier(
    synthetic_training_data: tuple[list[VariantFeatures], list[VariantClass]],
) -> VariantClassifier:
    """Create and train a classifier (session-scoped)."""
    features, labels = synthetic_training_data
    classifier = VariantClassifier(n_estimators=50, random_state=42)
    classifier.train(features, labels, calibrate=False)
    return classifier


@pytest.fixture
def benign_variant_features() -> VariantFeatures:
    """Features typical of a benign variant."""
    return VariantFeatures(
        phylop_score=-1.0,
        phastcons_score=0.1,
        gerp_score=-0.5,
        gnomad_af=0.15,
        gnomad_af_afr=0.12,
        gnomad_af_eas=0.18,
        gnomad_af_nfe=0.14,
        gnomad_homozygote_count=5000,
        sift_score=0.9,
        polyphen2_score=0.05,
        cadd_phred=5.0,
        revel_score=0.1,
        mutation_taster_score=0.1,
        in_protein_domain=False,
        domain_conservation=0.0,
        distance_to_active_site=-1.0,
        pfam_domain_count=0,
        variant_type="SNV",
        consequence="synonymous",
        exon_number=3,
        total_exons=10,
        amino_acid_change_blosum62=0.0,
        grantham_distance=0.0,
        splice_ai_score=0.01,
        max_splice_distance=100,
    )


@pytest.fixture
def pathogenic_variant_features() -> VariantFeatures:
    """Features typical of a pathogenic variant."""
    return VariantFeatures(
        phylop_score=5.5,
        phastcons_score=0.99,
        gerp_score=5.8,
        gnomad_af=0.0,
        gnomad_af_afr=0.0,
        gnomad_af_eas=0.0,
        gnomad_af_nfe=0.0,
        gnomad_homozygote_count=0,
        sift_score=0.001,
        polyphen2_score=0.998,
        cadd_phred=38.0,
        revel_score=0.95,
        mutation_taster_score=0.99,
        in_protein_domain=True,
        domain_conservation=0.95,
        distance_to_active_site=2.0,
        pfam_domain_count=2,
        variant_type="SNV",
        consequence="nonsense",
        exon_number=5,
        total_exons=12,
        amino_acid_change_blosum62=-4.0,
        grantham_distance=180.0,
        splice_ai_score=0.02,
        max_splice_distance=50,
    )


# ---------------------------------------------------------------------------
# VCF fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_vcf_content() -> str:
    """Sample VCF file content."""
    return create_sample_vcf()


@pytest.fixture
def vcf_parser() -> VCFParser:
    """Fresh VCF parser instance."""
    return VCFParser()


@pytest.fixture
def vcf_processor() -> VCFProcessor:
    """VCF processor with default quality thresholds."""
    return VCFProcessor()


@pytest.fixture
def lenient_quality_thresholds() -> QualityThresholds:
    """Quality thresholds that will pass most variants."""
    return QualityThresholds(
        min_qual=10.0,
        min_depth=5,
        min_genotype_quality=10,
        min_allele_balance=0.1,
        max_allele_balance=0.9,
        min_mapping_quality=20.0,
        require_pass_filter=False,
    )


# ---------------------------------------------------------------------------
# Security fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def audit_logger() -> AuditLogger:
    """Fresh audit logger instance."""
    return AuditLogger()


@pytest.fixture
def field_encryption() -> FieldLevelEncryption:
    """Field-level encryption with a fixed test key."""
    test_key = b"\x00" * 32  # 32-byte key for testing
    return FieldLevelEncryption(encryption_key=test_key)


@pytest.fixture
def genomic_encryption() -> GenomicDataEncryption:
    """Genomic data encryption (local mode, no KMS)."""
    return GenomicDataEncryption(local_key=b"\x01" * 32)


@pytest.fixture
def sample_phi_record() -> dict[str, Any]:
    """Sample record containing PHI fields."""
    return {
        "variant_id": "abc123",
        "chrom": "chr17",
        "pos": 7675088,
        "classification": "pathogenic",
        "patient_id": "PAT-001",
        "patient_name": "Test Patient",
        "date_of_birth": "1990-01-15",
        "medical_record_number": "MRN-12345",
        "sample_id": "SAMP-001",
        "ordering_physician": "Dr. Smith",
        "confidence": 0.95,
    }


# ---------------------------------------------------------------------------
# Metrics fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def metrics_collector() -> MetricsCollector:
    """Metrics collector for testing (no DataDog connection)."""
    config = MetricsConfig(
        api_key="",
        service_name="cloud-genomics-test",
        environment="test",
    )
    return MetricsCollector(config=config)


# ---------------------------------------------------------------------------
# FastAPI test client
# ---------------------------------------------------------------------------
@pytest.fixture
def api_client():
    """FastAPI test client."""
    from httpx import ASGITransport, AsyncClient

    from cloud_genomics.api.main import app

    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")
