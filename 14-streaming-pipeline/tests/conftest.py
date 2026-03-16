"""
Shared pytest fixtures for StreamRx tests.

Provides mock configurations, sample events, and AWS moto fixtures
for integration testing without real infrastructure.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any

import pytest

# Set environment variables before importing application modules
os.environ.setdefault("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")

from stream_rx.config import (
    KafkaConfig,
    KinesisConfig,
    MonitoringConfig,
    RedisConfig,
    S3Config,
    SignalDetectionConfig,
)
from stream_rx.models import (
    AdverseEventReport,
    EventOutcome,
    PrescriptionEvent,
    ReactionTerm,
    SeverityLevel,
    SuspectDrug,
)

# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kafka_config() -> KafkaConfig:
    """Test Kafka configuration pointing to localhost."""
    return KafkaConfig()


@pytest.fixture
def kinesis_config() -> KinesisConfig:
    """Test Kinesis configuration."""
    return KinesisConfig()


@pytest.fixture
def s3_config() -> S3Config:
    """Test S3 configuration."""
    return S3Config()


@pytest.fixture
def redis_config() -> RedisConfig:
    """Test Redis configuration."""
    return RedisConfig()


@pytest.fixture
def monitoring_config() -> MonitoringConfig:
    """Test monitoring configuration."""
    return MonitoringConfig()


@pytest.fixture
def signal_config() -> SignalDetectionConfig:
    """Signal detection config with lower thresholds for testing."""
    return SignalDetectionConfig()


# ---------------------------------------------------------------------------
# Sample event fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_prescription() -> PrescriptionEvent:
    """A valid prescription event for testing."""
    return PrescriptionEvent(
        patient_id="PAT-NE-123456",
        drug_ndc="00002-4462-30",
        drug_name="Atorvastatin 40mg",
        drug_class="statin",
        prescriber_npi="1234567890",
        prescriber_name="Dr. Smith",
        pharmacy_ncpdp="0312456",
        pharmacy_name="CVS Pharmacy #1823",
        pharmacy_state="NY",
        quantity=30.0,
        days_supply=30,
        refill_number=2,
        diagnosis_codes=["E78.5"],
        plan_id="PLAN-1234",
    )


@pytest.fixture
def sample_prescription_opioid() -> PrescriptionEvent:
    """An opioid prescription for interaction testing."""
    return PrescriptionEvent(
        patient_id="PAT-NE-123456",  # Same patient as above
        drug_ndc="61958-1001-01",
        drug_name="Oxycodone 5mg",
        drug_class="opioid",
        prescriber_npi="2345678901",
        pharmacy_ncpdp="0312456",
        quantity=30.0,
        days_supply=10,
        refill_number=0,
        diagnosis_codes=["M54.5"],
    )


@pytest.fixture
def sample_prescription_nsaid() -> PrescriptionEvent:
    """An NSAID prescription for interaction testing."""
    return PrescriptionEvent(
        patient_id="PAT-SE-654321",
        drug_ndc="00904-6214-61",
        drug_name="Ibuprofen 800mg",
        drug_class="nsaid",
        prescriber_npi="3456789012",
        pharmacy_ncpdp="0487921",
        quantity=90.0,
        days_supply=30,
        refill_number=0,
    )


@pytest.fixture
def sample_prescription_anticoagulant() -> PrescriptionEvent:
    """An anticoagulant prescription for interaction testing."""
    return PrescriptionEvent(
        patient_id="PAT-SE-654321",  # Same patient as NSAID
        drug_ndc="00310-0280-30",
        drug_name="Warfarin 5mg",
        drug_class="anticoagulant",
        prescriber_npi="4567890123",
        pharmacy_ncpdp="0487921",
        quantity=30.0,
        days_supply=30,
        refill_number=5,
    )


@pytest.fixture
def sample_adverse_event() -> AdverseEventReport:
    """A valid adverse event report for testing."""
    return AdverseEventReport(
        patient_id="PAT-MW-789012",
        patient_age=67,
        patient_sex="F",
        patient_weight_kg=72.5,
        suspect_drugs=[
            SuspectDrug(
                drug_name="Atorvastatin",
                drug_ndc="00002-4462-30",
                drug_class="statin",
                role="suspect",
                dose="40mg",
                route="oral",
                start_date=datetime.utcnow() - timedelta(days=90),
            )
        ],
        concomitant_drugs=["Lisinopril", "Metformin"],
        reactions=[
            ReactionTerm(
                term="Rhabdomyolysis",
                meddra_pt_code="10039020",
                seriousness="serious",
            )
        ],
        severity=SeverityLevel.SEVERE,
        outcome=EventOutcome.RECOVERING,
        hospitalized=True,
        reporter_type="physician",
        reporter_country="US",
        narrative="67yo female presented with severe myalgia and elevated CK levels.",
    )


@pytest.fixture
def sample_adverse_event_multi_drug() -> AdverseEventReport:
    """An adverse event with multiple suspect drugs."""
    return AdverseEventReport(
        patient_id="PAT-SW-345678",
        patient_age=45,
        patient_sex="M",
        suspect_drugs=[
            SuspectDrug(
                drug_name="Warfarin",
                drug_ndc="00310-0280-30",
                drug_class="anticoagulant",
                role="suspect",
                dose="5mg",
            ),
            SuspectDrug(
                drug_name="Ibuprofen",
                drug_ndc="00904-6214-61",
                drug_class="nsaid",
                role="interacting",
                dose="800mg",
            ),
        ],
        reactions=[
            ReactionTerm(
                term="Gastrointestinal Haemorrhage",
                meddra_pt_code="10017955",
                seriousness="serious",
            ),
            ReactionTerm(
                term="Nausea",
                meddra_pt_code="10028813",
                seriousness="non_serious",
            ),
        ],
        severity=SeverityLevel.SEVERE,
        outcome=EventOutcome.RECOVERED,
        hospitalized=True,
        reporter_type="consumer",
    )


# ---------------------------------------------------------------------------
# Batch event generators
# ---------------------------------------------------------------------------


@pytest.fixture
def prescription_batch() -> list[dict[str, Any]]:
    """A batch of prescription events as dicts for processing tests."""
    base_time = datetime.utcnow()
    events = []
    drugs = [
        ("00002-4462-30", "Atorvastatin 40mg", "statin"),
        ("00093-7180-01", "Metformin 500mg", "biguanide"),
        ("00378-1800-01", "Lisinopril 10mg", "ace_inhibitor"),
        ("00591-0405-01", "Omeprazole 20mg", "ppi"),
        ("55111-0160-30", "Sertraline 50mg", "ssri"),
    ]
    for i in range(20):
        ndc, name, cls = drugs[i % len(drugs)]
        events.append(
            {
                "event_id": f"evt-{i:04d}",
                "event_type": "prescription_fill",
                "timestamp": (base_time - timedelta(seconds=i)).isoformat(),
                "patient_id": f"PAT-NE-{100000 + i}",
                "drug_ndc": ndc,
                "drug_name": name,
                "drug_class": cls,
                "prescriber_npi": "1234567890",
                "prescriber_name": "Dr. Test",
                "pharmacy_ncpdp": "0312456",
                "pharmacy_name": "Test Pharmacy",
                "pharmacy_state": "NY",
                "quantity": 30.0,
                "days_supply": 30,
                "refill_number": 0,
                "diagnosis_codes": ["E78.5"],
                "plan_id": "PLAN-TEST",
            }
        )
    return events


@pytest.fixture
def adverse_event_batch_for_signal() -> list[dict[str, Any]]:
    """
    A batch of adverse events designed to trigger a signal for Atorvastatin-Rhabdomyolysis.
    """
    events = []
    for i in range(15):
        events.append(
            {
                "report_id": f"rpt-{i:04d}",
                "event_type": "adverse_event",
                "timestamp": datetime.utcnow().isoformat(),
                "patient_id": f"PAT-MW-{200000 + i}",
                "patient_age": 60 + i,
                "patient_sex": "M" if i % 2 == 0 else "F",
                "suspect_drugs": [
                    {
                        "drug_name": "Atorvastatin",
                        "drug_ndc": "00002-4462-30",
                        "drug_class": "statin",
                        "role": "suspect",
                        "dose": "40mg",
                        "route": "oral",
                    }
                ],
                "concomitant_drugs": [],
                "reactions": [
                    {
                        "term": "Rhabdomyolysis",
                        "meddra_pt_code": "10039020",
                        "seriousness": "serious",
                    }
                ],
                "severity": "severe",
                "outcome": "recovering",
                "hospitalized": True,
                "reporter_type": "physician",
                "reporter_country": "US",
                "narrative": f"Test adverse event {i}",
            }
        )

    # Add background events with different drugs/reactions
    for i in range(30):
        events.append(
            {
                "report_id": f"rpt-bg-{i:04d}",
                "event_type": "adverse_event",
                "timestamp": datetime.utcnow().isoformat(),
                "patient_id": f"PAT-SE-{300000 + i}",
                "patient_age": 50,
                "patient_sex": "U",
                "suspect_drugs": [
                    {
                        "drug_name": "Metformin",
                        "drug_ndc": "00093-7180-01",
                        "drug_class": "biguanide",
                        "role": "suspect",
                        "dose": "500mg",
                        "route": "oral",
                    }
                ],
                "concomitant_drugs": [],
                "reactions": [
                    {
                        "term": "Nausea",
                        "meddra_pt_code": "10028813",
                        "seriousness": "non_serious",
                    }
                ],
                "severity": "mild",
                "outcome": "recovered",
                "hospitalized": False,
                "reporter_type": "consumer",
                "reporter_country": "US",
                "narrative": f"Background event {i}",
            }
        )
    return events
