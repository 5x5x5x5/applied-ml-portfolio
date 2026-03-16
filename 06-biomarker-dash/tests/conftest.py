"""Shared test fixtures for BiomarkerDash tests."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest

from biomarker_dash.alerts.alert_engine import AlertEngine
from biomarker_dash.models.anomaly_detector import AnomalyDetector
from biomarker_dash.models.trend_analyzer import TrendAnalyzer
from biomarker_dash.schemas import (
    BiomarkerReading,
    BiomarkerType,
    PatientContext,
    Sex,
)


@pytest.fixture
def anomaly_detector() -> AnomalyDetector:
    """Fresh anomaly detector instance."""
    return AnomalyDetector()


@pytest.fixture
def trend_analyzer() -> TrendAnalyzer:
    """Fresh trend analyzer instance."""
    return TrendAnalyzer()


@pytest.fixture
def alert_engine() -> AlertEngine:
    """Fresh alert engine instance."""
    return AlertEngine()


@pytest.fixture
def patient_context() -> PatientContext:
    """Sample patient context for testing."""
    return PatientContext(
        patient_id="TEST001",
        age=55,
        sex=Sex.MALE,
        conditions=["hypertension", "diabetes_type2"],
        medications=["metformin", "lisinopril"],
    )


@pytest.fixture
def patient_context_female() -> PatientContext:
    """Sample female patient context for testing."""
    return PatientContext(
        patient_id="TEST002",
        age=40,
        sex=Sex.FEMALE,
        conditions=[],
        medications=[],
    )


def make_reading(
    patient_id: str = "TEST001",
    biomarker_type: BiomarkerType = BiomarkerType.GLUCOSE,
    value: float = 85.0,
    timestamp: datetime | None = None,
    unit: str = "mg/dL",
) -> BiomarkerReading:
    """Helper to create a BiomarkerReading with defaults."""
    return BiomarkerReading(
        reading_id=str(uuid.uuid4()),
        patient_id=patient_id,
        biomarker_type=biomarker_type,
        value=value,
        unit=unit,
        timestamp=timestamp or datetime.utcnow(),
        source="test",
    )


def make_reading_series(
    patient_id: str = "TEST001",
    biomarker_type: BiomarkerType = BiomarkerType.GLUCOSE,
    values: list[float] | None = None,
    start_time: datetime | None = None,
    interval_minutes: int = 30,
    unit: str = "mg/dL",
) -> list[BiomarkerReading]:
    """Create a time-ordered series of readings."""
    if values is None:
        values = [85.0, 87.0, 84.0, 88.0, 86.0, 90.0, 85.0]

    start = start_time or (datetime.utcnow() - timedelta(hours=len(values)))
    readings = []
    for i, val in enumerate(values):
        ts = start + timedelta(minutes=i * interval_minutes)
        readings.append(
            make_reading(
                patient_id=patient_id,
                biomarker_type=biomarker_type,
                value=val,
                timestamp=ts,
                unit=unit,
            )
        )
    return readings


@pytest.fixture
def normal_glucose_readings() -> list[BiomarkerReading]:
    """Series of normal glucose readings."""
    return make_reading_series(
        values=[82.0, 85.0, 88.0, 84.0, 86.0, 83.0, 87.0, 85.0, 89.0, 84.0],
        unit="mg/dL",
    )


@pytest.fixture
def rising_glucose_readings() -> list[BiomarkerReading]:
    """Series of glucose readings with a clear upward trend."""
    return make_reading_series(
        values=[85.0, 88.0, 92.0, 96.0, 100.0, 105.0, 110.0, 116.0, 122.0, 130.0],
        unit="mg/dL",
    )


@pytest.fixture
def anomalous_heart_rate_reading() -> BiomarkerReading:
    """A clearly anomalous heart rate reading."""
    return make_reading(
        biomarker_type=BiomarkerType.HEART_RATE,
        value=155.0,
        unit="bpm",
    )


@pytest.fixture
def critical_potassium_reading() -> BiomarkerReading:
    """Critically high potassium reading."""
    return make_reading(
        biomarker_type=BiomarkerType.POTASSIUM,
        value=7.0,
        unit="mEq/L",
    )
