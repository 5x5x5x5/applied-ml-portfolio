"""Pydantic schemas for API request/response validation."""

from __future__ import annotations

import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

# --- Enums ---


class SubmissionType(str, Enum):
    NDA = "NDA"
    ANDA = "ANDA"
    BLA = "BLA"
    IND = "IND"
    SNDA = "sNDA"
    SBLA = "sBLA"
    MAA = "MAA"
    TYPE_II_VARIATION = "TYPE_II_VARIATION"
    TYPE_IB_VARIATION = "TYPE_IB_VARIATION"
    RENEWAL = "RENEWAL"
    ANNUAL_REPORT = "ANNUAL_REPORT"
    PSUR = "PSUR"
    DSUR = "DSUR"


class SubmissionStatus(str, Enum):
    DRAFT = "DRAFT"
    SUBMITTED = "SUBMITTED"
    UNDER_REVIEW = "UNDER_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    WITHDRAWN = "WITHDRAWN"
    ON_HOLD = "ON_HOLD"
    COMPLETE_RESPONSE = "COMPLETE_RESPONSE"


class Priority(str, Enum):
    STANDARD = "STANDARD"
    PRIORITY = "PRIORITY"
    ACCELERATED = "ACCELERATED"
    BREAKTHROUGH = "BREAKTHROUGH"
    FAST_TRACK = "FAST_TRACK"


class DocType(str, Enum):
    COVER_LETTER = "COVER_LETTER"
    MODULE_1 = "MODULE_1"
    MODULE_2 = "MODULE_2"
    MODULE_3 = "MODULE_3"
    MODULE_4 = "MODULE_4"
    MODULE_5 = "MODULE_5"
    LABEL = "LABEL"
    PI = "PI"
    PPI = "PPI"
    CLINICAL_STUDY_REPORT = "CLINICAL_STUDY_REPORT"
    SAFETY_UPDATE = "SAFETY_UPDATE"
    CMC_DATA = "CMC_DATA"
    CORRESPONDENCE = "CORRESPONDENCE"
    MEETING_MINUTES = "MEETING_MINUTES"
    OTHER = "OTHER"


class PseudoType(str, Enum):
    SUBMISSION_ID = "SUBMISSION_ID"
    DOCUMENT_ID = "DOCUMENT_ID"
    PATIENT_ID = "PATIENT_ID"
    INVESTIGATOR_ID = "INVESTIGATOR_ID"
    SITE_ID = "SITE_ID"
    BATCH_NUMBER = "BATCH_NUMBER"
    COMPOUND_CODE = "COMPOUND_CODE"
    PROTOCOL_NUMBER = "PROTOCOL_NUMBER"


class ComplianceStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    OVERDUE = "OVERDUE"
    ESCALATED = "ESCALATED"
    WAIVED = "WAIVED"
    CANCELLED = "CANCELLED"


# --- Submission Schemas ---


class SubmissionCreate(BaseModel):
    drug_id: int
    submission_type: SubmissionType
    agency: str = Field(max_length=20)
    priority: Priority = Priority.STANDARD
    assigned_to: str | None = None
    description: str | None = None
    created_by: str


class SubmissionUpdate(BaseModel):
    status: SubmissionStatus | None = None
    priority: Priority | None = None
    assigned_to: str | None = None
    description: str | None = None
    target_date: datetime.datetime | None = None
    modified_by: str


class SubmissionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    drug_id: int
    submission_type: str
    status: str
    submitted_date: datetime.datetime | None = None
    target_date: datetime.datetime | None = None
    agency: str
    tracking_number: str | None = None
    priority: str
    assigned_to: str | None = None
    description: str | None = None
    created_by: str
    created_date: datetime.datetime
    modified_by: str | None = None
    modified_date: datetime.datetime


class SubmissionListResponse(BaseModel):
    items: list[SubmissionResponse]
    total: int
    page: int
    page_size: int


# --- Document Schemas ---


class DocumentCreate(BaseModel):
    submission_id: int
    doc_type: DocType
    doc_title: str = Field(max_length=500)
    file_path: str
    checksum: str
    uploaded_by: str
    confidentiality: str = "CONFIDENTIAL"


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    submission_id: int
    doc_type: str
    doc_title: str
    version: float
    file_path: str
    file_size_bytes: int | None = None
    checksum: str
    checksum_algorithm: str
    uploaded_by: str
    upload_date: datetime.datetime
    status: str
    confidentiality: str


# --- Approval Schemas ---


class ApprovalCreate(BaseModel):
    submission_id: int
    approval_date: datetime.date
    approval_type: str
    conditions: str | None = None
    expiry_date: datetime.date | None = None
    approval_number: str | None = None
    approved_indication: str | None = None
    market_exclusivity: str | None = None
    created_by: str


class ApprovalResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    submission_id: int
    approval_date: datetime.date
    approval_type: str
    conditions: str | None = None
    expiry_date: datetime.date | None = None
    approval_number: str | None = None
    approved_indication: str | None = None
    market_exclusivity: str | None = None
    pediatric_exclusivity: str
    created_by: str
    created_date: datetime.datetime


# --- Pseudo Record Schemas ---


class PseudoRecordCreate(BaseModel):
    original_record_id: int
    source_table: str
    pseudo_type: PseudoType
    mapping_key: str = Field(min_length=16, max_length=128)
    created_by: str
    purpose: str | None = None
    expiry_days: int = Field(default=365, ge=1, le=3650)


class PseudoRecordResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    pseudo_value: str
    pseudo_type: str
    source_table: str
    is_active: str
    created_date: datetime.datetime
    expiry_date: datetime.datetime | None = None
    access_count: int


class ReidentificationRequest(BaseModel):
    pseudo_value: str
    requested_by: str
    authorized_by: str
    reason: str = Field(min_length=10)


class ReidentificationResponse(BaseModel):
    original_record_id: int
    source_table: str
    pseudo_type: str
    authorized_by: str
    request_logged: bool = True


class BatchPseudoRequest(BaseModel):
    source_table: str
    pseudo_type: PseudoType
    mapping_key: str = Field(min_length=16, max_length=128)
    created_by: str
    purpose: str | None = None
    expiry_days: int = 365


class BatchPseudoResponse(BaseModel):
    generated_count: int
    source_table: str
    pseudo_type: str


# --- Compliance Schemas ---


class ComplianceCreate(BaseModel):
    drug_id: int
    requirement_type: str
    requirement_desc: str
    due_date: datetime.date
    responsible_party: str
    linked_submission: int | None = None
    regulatory_reference: str | None = None
    created_by: str


class ComplianceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    drug_id: int
    requirement_type: str
    requirement_desc: str
    due_date: datetime.date
    completion_date: datetime.date | None = None
    status: str
    responsible_party: str
    escalation_level: int
    risk_score: float | None = None
    linked_submission: int | None = None
    regulatory_reference: str | None = None
    created_date: datetime.datetime


class ComplianceSummary(BaseModel):
    total: int
    pending: int
    in_progress: int
    overdue: int
    escalated: int
    completed: int
    avg_risk_score: float | None = None


# --- Audit Schemas ---


class AuditTrailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    table_name: str
    record_id: int
    action: str
    old_values: str | None = None
    new_values: str | None = None
    changed_by: str
    changed_at: datetime.datetime
    ip_address: str | None = None


class AuditTrailQuery(BaseModel):
    table_name: str | None = None
    record_id: int | None = None
    changed_by: str | None = None
    date_from: datetime.datetime | None = None
    date_to: datetime.datetime | None = None
    action: str | None = None
    page: int = 1
    page_size: int = 50
