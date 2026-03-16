"""Shared test fixtures for PharmaSentinel test suite."""

from __future__ import annotations

import json
import os
from collections.abc import Generator
from typing import Any

import boto3
import pytest
from moto import mock_aws

from pharma_sentinel.config import AppSettings, AWSSettings, DataDogSettings, ModelSettings
from pharma_sentinel.models.adverse_event_classifier import AdverseEventClassifier

# ─────────────────────────────────────────────────────────────────────
# Environment setup
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set environment variables for testing."""
    monkeypatch.setenv("APP_ENVIRONMENT", "local")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("DD_TRACE_ENABLED", "false")


# ─────────────────────────────────────────────────────────────────────
# Settings fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def test_settings() -> AppSettings:
    """Create test application settings."""
    return AppSettings(
        environment="local",
        debug=True,
        log_level="DEBUG",
        aws=AWSSettings(
            region="us-east-1",
            access_key_id="testing",
            secret_access_key="testing",
            s3_input_bucket="test-pharma-input",
            s3_output_bucket="test-pharma-output",
            s3_model_bucket="test-pharma-models",
            sqs_critical_queue_url="",
        ),
        datadog=DataDogSettings(
            trace_enabled=False,
            api_key=None,
        ),
        model=ModelSettings(
            artifact_path="/tmp/test_model.joblib",
            confidence_threshold=0.5,
        ),
    )


# ─────────────────────────────────────────────────────────────────────
# Training data fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_training_texts() -> list[str]:
    """Sample adverse event texts for training."""
    return [
        # Mild
        "Patient reported mild nausea and headache after taking the medication. No treatment required.",
        "Subject experienced dizziness and fatigue lasting two days. Symptoms resolved on their own.",
        "Mild dry mouth and drowsiness reported by patient. Continued medication without changes.",
        "Patient noted mild insomnia and appetite loss during the first week of treatment.",
        "Subject reported constipation and mild muscle ache following dose adjustment.",
        "Headache and mild itching reported. No intervention needed.",
        "Fatigue and mild weight gain observed over three months of treatment.",
        "Patient experienced drowsiness and dry mouth. Dose was not changed.",
        # Moderate
        "Patient developed a significant rash with edema and fever requiring medical attention.",
        "Vomiting and diarrhea for three days. Patient required IV fluids for dehydration.",
        "Moderate hypertension observed with tachycardia. Antihypertensive treatment initiated.",
        "Patient presented with persistent vomiting and moderate bleeding from the injection site.",
        "Syncope episode reported along with hypotension. Patient was evaluated in the ER.",
        "Moderate infection at the surgical site requiring antibiotic therapy.",
        "Fever and pneumonia developed during treatment. Course of antibiotics prescribed.",
        "Patient experienced moderate edema and bradycardia requiring dose modification.",
        # Severe
        "Patient was hospitalized due to severe seizure activity after drug administration.",
        "Liver failure suspected, patient transferred to ICU. Drug was discontinued immediately.",
        "Severe rhabdomyolysis with renal failure requiring dialysis and hospitalization.",
        "Stevens-Johnson syndrome diagnosed. Patient required extensive hospitalization.",
        "Pancreatitis with disability. Patient underwent emergency surgery.",
        "Severe agranulocytosis detected. Hospitalization and treatment initiated.",
        "Thrombocytopenia leading to significant incapacity. Transplant evaluation started.",
        "Patient hospitalized with severe convulsions and liver failure.",
        # Critical
        "Patient died following cardiac arrest believed to be related to the medication.",
        "Fatal anaphylactic reaction occurred within minutes of drug administration.",
        "Life-threatening respiratory failure. Patient on ventilator in critical condition.",
        "Death reported due to hemorrhagic stroke during the treatment period.",
        "Fatal organ failure following prolonged use. Pulmonary embolism confirmed.",
        "Patient experienced myocardial infarction and coma. Outcome was death.",
        "Anaphylaxis leading to cardiac arrest. Despite resuscitation, patient was fatal.",
        "Sepsis with multi-organ failure. Patient in coma, life-threatening condition.",
    ]


@pytest.fixture
def sample_training_labels() -> list[str]:
    """Corresponding severity labels for training texts."""
    return ["mild"] * 8 + ["moderate"] * 8 + ["severe"] * 8 + ["critical"] * 8


@pytest.fixture
def trained_classifier(
    sample_training_texts: list[str],
    sample_training_labels: list[str],
) -> AdverseEventClassifier:
    """A trained classifier instance for testing predictions."""
    classifier = AdverseEventClassifier(
        max_features=5000,
        ngram_range=(1, 2),
        c_param=1.0,
    )
    classifier.train(
        texts=sample_training_texts,
        labels=sample_training_labels,
        validate=False,  # Skip CV for speed in tests
    )
    return classifier


# ─────────────────────────────────────────────────────────────────────
# AWS Mock fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def aws_credentials() -> None:
    """Mock AWS credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@pytest.fixture
def mock_s3(aws_credentials: None) -> Generator[Any, None, None]:
    """Mock S3 service with test buckets."""
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")

        # Create test buckets
        for bucket in ["test-pharma-input", "test-pharma-output", "test-pharma-models"]:
            s3.create_bucket(Bucket=bucket)

        yield s3


@pytest.fixture
def mock_sqs(aws_credentials: None) -> Generator[Any, None, None]:
    """Mock SQS service with test queues."""
    with mock_aws():
        sqs = boto3.client("sqs", region_name="us-east-1")

        # Create test queues
        dlq = sqs.create_queue(QueueName="test-pharma-critical-dlq")
        queue = sqs.create_queue(
            QueueName="test-pharma-critical",
            Attributes={
                "RedrivePolicy": json.dumps(
                    {
                        "deadLetterTargetArn": "arn:aws:sqs:us-east-1:000000000000:test-pharma-critical-dlq",
                        "maxReceiveCount": "3",
                    }
                ),
            },
        )

        yield sqs


# ─────────────────────────────────────────────────────────────────────
# Sample data fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_csv_data() -> bytes:
    """Sample FAERS CSV data."""
    header = "primaryid$caseid$event_dt$age$sex$drugname$drug_ind$pt$outc_cod$rept_cod"
    rows = [
        "100001$C100001$20240101$55$F$Ibuprofen$Pain$Nausea and headache$DE$HP",
        "100002$C100002$20240102$72$M$Metformin$Diabetes$Severe liver failure and hospitalization$HO$HP",
        "100003$C100003$20240103$34$F$Aspirin$Heart Disease$Mild dizziness$OT$CS",
    ]
    return ("\n".join([header] + rows)).encode("utf-8")


@pytest.fixture
def sample_xml_data() -> bytes:
    """Sample FAERS XML data."""
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <ichicsr>
        <safetyreport>
            <safetyreportid>XML001</safetyreportid>
            <companynumb>COMP001</companynumb>
            <receiptdate>20240101</receiptdate>
            <serious>1</serious>
            <patient>
                <patientonsetage>65</patientonsetage>
                <patientsex>1</patientsex>
                <drug>
                    <medicinalproduct>Warfarin</medicinalproduct>
                    <drugindication>Atrial Fibrillation</drugindication>
                </drug>
                <reaction>
                    <reactionmeddrapt>Hemorrhagic stroke</reactionmeddrapt>
                </reaction>
            </patient>
        </safetyreport>
        <safetyreport>
            <safetyreportid>XML002</safetyreportid>
            <companynumb>COMP002</companynumb>
            <receiptdate>20240115</receiptdate>
            <patient>
                <patientonsetage>42</patientonsetage>
                <patientsex>2</patientsex>
                <drug>
                    <medicinalproduct>Lisinopril</medicinalproduct>
                    <drugindication>Hypertension</drugindication>
                </drug>
                <reaction>
                    <reactionmeddrapt>Dry cough and dizziness</reactionmeddrapt>
                </reaction>
            </patient>
        </safetyreport>
    </ichicsr>"""
    return xml.encode("utf-8")
