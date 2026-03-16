"""Pydantic schemas for biomarker data models."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class BiomarkerType(str, enum.Enum):
    """Supported biomarker types."""

    GLUCOSE = "glucose"
    HEMOGLOBIN = "hemoglobin"
    WBC = "wbc"
    PLATELET = "platelet"
    CREATININE = "creatinine"
    ALT = "alt"
    AST = "ast"
    HEART_RATE = "heart_rate"
    BLOOD_PRESSURE_SYS = "blood_pressure_sys"
    BLOOD_PRESSURE_DIA = "blood_pressure_dia"
    TEMPERATURE = "temperature"
    OXYGEN_SAT = "oxygen_sat"
    POTASSIUM = "potassium"
    SODIUM = "sodium"
    TSH = "tsh"
    CHOLESTEROL_TOTAL = "cholesterol_total"
    TROPONIN = "troponin"
    CRP = "crp"


class Sex(str, enum.Enum):
    MALE = "male"
    FEMALE = "female"


class AlertSeverity(str, enum.Enum):
    """Alert severity levels following clinical triage."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, enum.Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class TrendDirection(str, enum.Enum):
    IMPROVING = "improving"
    WORSENING = "worsening"
    STABLE = "stable"
    UNKNOWN = "unknown"


# Normal ranges: (low, high) for general adult population
NORMAL_RANGES: dict[BiomarkerType, tuple[float, float]] = {
    BiomarkerType.GLUCOSE: (70.0, 100.0),
    BiomarkerType.HEMOGLOBIN: (12.0, 17.5),
    BiomarkerType.WBC: (4.5, 11.0),
    BiomarkerType.PLATELET: (150.0, 400.0),
    BiomarkerType.CREATININE: (0.6, 1.2),
    BiomarkerType.ALT: (7.0, 56.0),
    BiomarkerType.AST: (10.0, 40.0),
    BiomarkerType.HEART_RATE: (60.0, 100.0),
    BiomarkerType.BLOOD_PRESSURE_SYS: (90.0, 120.0),
    BiomarkerType.BLOOD_PRESSURE_DIA: (60.0, 80.0),
    BiomarkerType.TEMPERATURE: (97.0, 99.5),
    BiomarkerType.OXYGEN_SAT: (95.0, 100.0),
    BiomarkerType.POTASSIUM: (3.5, 5.0),
    BiomarkerType.SODIUM: (136.0, 145.0),
    BiomarkerType.TSH: (0.4, 4.0),
    BiomarkerType.CHOLESTEROL_TOTAL: (0.0, 200.0),
    BiomarkerType.TROPONIN: (0.0, 0.04),
    BiomarkerType.CRP: (0.0, 3.0),
}

# Sex-adjusted normal ranges where applicable
SEX_ADJUSTED_RANGES: dict[BiomarkerType, dict[Sex, tuple[float, float]]] = {
    BiomarkerType.HEMOGLOBIN: {
        Sex.MALE: (13.5, 17.5),
        Sex.FEMALE: (12.0, 15.5),
    },
    BiomarkerType.CREATININE: {
        Sex.MALE: (0.7, 1.3),
        Sex.FEMALE: (0.6, 1.1),
    },
}


class PatientContext(BaseModel):
    """Patient demographic and clinical context for adjusted ranges."""

    patient_id: str
    age: int = Field(ge=0, le=150)
    sex: Sex
    conditions: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)


class BiomarkerReading(BaseModel):
    """A single biomarker measurement."""

    reading_id: str = Field(default="")
    patient_id: str
    biomarker_type: BiomarkerType
    value: float
    unit: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = "lab"
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnomalyResult(BaseModel):
    """Result of anomaly detection on a reading."""

    reading_id: str
    patient_id: str
    biomarker_type: BiomarkerType
    value: float
    is_anomaly: bool
    anomaly_score: float = Field(ge=0.0, le=1.0)
    severity: AlertSeverity
    detection_method: str
    explanation: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    normal_range: tuple[float, float] = (0.0, 0.0)


class TrendResult(BaseModel):
    """Result of trend analysis for a biomarker time series."""

    patient_id: str
    biomarker_type: BiomarkerType
    direction: TrendDirection
    rate_of_change: float
    predicted_value_24h: float | None = None
    predicted_exit_normal: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    window_hours: int = 24
    data_points_used: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ClinicalAlert(BaseModel):
    """A clinical alert triggered by the alert engine."""

    alert_id: str
    patient_id: str
    biomarker_type: BiomarkerType
    severity: AlertSeverity
    status: AlertStatus = AlertStatus.ACTIVE
    title: str
    message: str
    value: float
    threshold: float | None = None
    detection_source: str  # "rule", "anomaly_ml", "trend"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None
    escalated_at: datetime | None = None
    related_reading_id: str = ""


class HealthStatus(BaseModel):
    """Application health check response."""

    status: str
    version: str
    redis_connected: bool
    uptime_seconds: float
    active_ws_connections: int
