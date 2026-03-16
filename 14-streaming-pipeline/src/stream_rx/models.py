"""
Pydantic models for all streaming events in the StreamRx pipeline.

These schemas are the single source of truth for event structure across
producers, consumers, and storage layers.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

import orjson
from pydantic import BaseModel, Field, field_validator


def _orjson_dumps(v: Any, *, default: Any = None) -> str:
    return orjson.dumps(v, default=default).decode()


class SeverityLevel(str, enum.Enum):
    """Severity classification for adverse events (MedDRA-aligned)."""

    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    LIFE_THREATENING = "life_threatening"
    FATAL = "fatal"


class EventOutcome(str, enum.Enum):
    """Patient outcome following an adverse event."""

    RECOVERED = "recovered"
    RECOVERING = "recovering"
    NOT_RECOVERED = "not_recovered"
    FATAL = "fatal"
    UNKNOWN = "unknown"


class AlertPriority(str, enum.Enum):
    """Priority levels for generated safety alerts."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PrescriptionEvent(BaseModel):
    """
    A single prescription fill or dispense event.

    Represents a real-time event generated when a prescription is filled at a
    pharmacy, capturing the drug, prescriber, patient, and pharmacy details.
    """

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = Field(default="prescription_fill", frozen=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    patient_id: str = Field(..., min_length=6, max_length=20)
    drug_ndc: str = Field(..., description="National Drug Code (11-digit)")
    drug_name: str = Field(..., min_length=1, max_length=200)
    drug_class: str = Field(default="unknown")
    prescriber_npi: str = Field(..., description="Prescriber NPI (10-digit)")
    prescriber_name: str = Field(default="")
    pharmacy_ncpdp: str = Field(..., description="Pharmacy NCPDP ID")
    pharmacy_name: str = Field(default="")
    pharmacy_state: str = Field(default="", max_length=2)
    quantity: float = Field(gt=0)
    days_supply: int = Field(gt=0, le=365)
    refill_number: int = Field(ge=0, le=99)
    diagnosis_codes: list[str] = Field(default_factory=list)
    plan_id: str = Field(default="")

    @field_validator("drug_ndc")
    @classmethod
    def validate_ndc(cls, v: str) -> str:
        cleaned = v.replace("-", "")
        if not cleaned.isdigit() or len(cleaned) not in (10, 11):
            raise ValueError(f"Invalid NDC format: {v}")
        return v

    @field_validator("prescriber_npi")
    @classmethod
    def validate_npi(cls, v: str) -> str:
        if not v.isdigit() or len(v) != 10:
            raise ValueError(f"Invalid NPI: must be 10 digits, got {v}")
        return v

    def partition_key(self) -> str:
        """Return the key used for Kafka/Kinesis partitioning."""
        return self.patient_id

    def to_bytes(self) -> bytes:
        return orjson.dumps(self.model_dump(), default=str)


class AdverseEventReport(BaseModel):
    """
    An adverse event report modelled after FDA FAERS submissions.

    Captures a patient's adverse reaction to one or more drugs, including
    severity, outcome, and reporter information.
    """

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}

    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = Field(default="adverse_event", frozen=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    patient_id: str = Field(..., min_length=6, max_length=20)
    patient_age: int | None = Field(default=None, ge=0, le=150)
    patient_sex: str | None = Field(default=None, pattern=r"^[MFU]$")
    patient_weight_kg: float | None = Field(default=None, ge=0)

    # Drug information - supports multi-drug reporting
    suspect_drugs: list[SuspectDrug] = Field(..., min_length=1)
    concomitant_drugs: list[str] = Field(default_factory=list)

    # Reaction details
    reactions: list[ReactionTerm] = Field(..., min_length=1)
    severity: SeverityLevel = Field(default=SeverityLevel.MODERATE)
    outcome: EventOutcome = Field(default=EventOutcome.UNKNOWN)
    hospitalized: bool = Field(default=False)

    # Reporter
    reporter_type: str = Field(default="consumer")  # consumer, physician, pharmacist, other
    reporter_country: str = Field(default="US", max_length=2)

    # Narrative
    narrative: str = Field(default="")

    def partition_key(self) -> str:
        return self.patient_id

    def to_bytes(self) -> bytes:
        return orjson.dumps(self.model_dump(), default=str)


class SuspectDrug(BaseModel):
    """A drug suspected of causing an adverse reaction."""

    drug_name: str = Field(..., min_length=1)
    drug_ndc: str = Field(default="")
    drug_class: str = Field(default="unknown")
    role: str = Field(default="suspect")  # suspect, concomitant, interacting
    dose: str = Field(default="")
    route: str = Field(default="oral")
    start_date: datetime | None = None
    end_date: datetime | None = None


class ReactionTerm(BaseModel):
    """An adverse reaction term (MedDRA Preferred Term)."""

    term: str = Field(..., min_length=1)
    meddra_pt_code: str = Field(default="")
    seriousness: str = Field(default="non_serious")


# Rebuild AdverseEventReport now that SuspectDrug and ReactionTerm are defined
AdverseEventReport.model_rebuild()


class DrugInteractionAlert(BaseModel):
    """Alert generated when a dangerous drug combination is detected."""

    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    alert_type: str = Field(default="drug_interaction")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    priority: AlertPriority = Field(default=AlertPriority.HIGH)
    patient_id: str
    drug_a: str
    drug_b: str
    interaction_description: str
    risk_level: str
    prescriber_npi: str = Field(default="")

    def to_bytes(self) -> bytes:
        return orjson.dumps(self.model_dump(), default=str)


class SafetySignalAlert(BaseModel):
    """Alert generated when a pharmacovigilance safety signal is detected."""

    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    alert_type: str = Field(default="safety_signal")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    priority: AlertPriority = Field(default=AlertPriority.HIGH)
    drug_name: str
    reaction_term: str
    metric_name: str  # PRR, ROR, etc.
    metric_value: float
    threshold: float
    case_count: int
    window_start: datetime
    window_end: datetime
    description: str = Field(default="")

    def to_bytes(self) -> bytes:
        return orjson.dumps(self.model_dump(), default=str)


class StreamMetrics(BaseModel):
    """Snapshot of stream processing metrics."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    consumer_group: str
    topic: str
    partition: int
    current_offset: int
    log_end_offset: int
    lag: int
    messages_per_sec: float
    avg_latency_ms: float
    error_rate: float
