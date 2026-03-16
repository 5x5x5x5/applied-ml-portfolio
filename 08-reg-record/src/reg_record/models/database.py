"""SQLAlchemy ORM models for the RegRecord database."""

from __future__ import annotations

import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class Drug(Base):
    __tablename__ = "drug"

    drug_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_name: Mapped[str] = mapped_column(String(200), nullable=False)
    generic_name: Mapped[str | None] = mapped_column(String(200))
    ndc_code: Mapped[str | None] = mapped_column(String(50), unique=True)
    therapeutic_area: Mapped[str | None] = mapped_column(String(100))
    manufacturer: Mapped[str] = mapped_column(String(200), nullable=False)
    active_flag: Mapped[str] = mapped_column(String(1), default="Y")
    created_date: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    modified_date: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    submissions: Mapped[list[RegulatorySubmission]] = relationship(back_populates="drug")
    labeling_changes: Mapped[list[LabelingChange]] = relationship(back_populates="drug")
    compliance_records: Mapped[list[ComplianceRecord]] = relationship(back_populates="drug")


class RegulatorySubmission(Base):
    __tablename__ = "regulatory_submission"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_id: Mapped[int] = mapped_column(Integer, ForeignKey("drug.drug_id"), nullable=False)
    submission_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="DRAFT", nullable=False)
    submitted_date: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    target_date: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    agency: Mapped[str] = mapped_column(String(20), nullable=False)
    tracking_number: Mapped[str | None] = mapped_column(String(100), unique=True)
    priority: Mapped[str] = mapped_column(String(20), default="STANDARD")
    assigned_to: Mapped[str | None] = mapped_column(String(100))
    description: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(String(100), nullable=False)
    created_date: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    modified_by: Mapped[str | None] = mapped_column(String(100))
    modified_date: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    drug: Mapped[Drug] = relationship(back_populates="submissions")
    documents: Mapped[list[SubmissionDocument]] = relationship(
        back_populates="submission", cascade="all, delete-orphan"
    )
    approvals: Mapped[list[ApprovalRecord]] = relationship(back_populates="submission")
    compliance_records: Mapped[list[ComplianceRecord]] = relationship(
        back_populates="linked_submission_rel"
    )

    __table_args__ = (
        CheckConstraint(
            "submission_type IN ('NDA','ANDA','BLA','IND','sNDA','sBLA',"
            "'MAA','TYPE_II_VARIATION','TYPE_IB_VARIATION',"
            "'RENEWAL','ANNUAL_REPORT','PSUR','DSUR')",
            name="chk_sub_type",
        ),
        CheckConstraint(
            "status IN ('DRAFT','SUBMITTED','UNDER_REVIEW','APPROVED',"
            "'REJECTED','WITHDRAWN','ON_HOLD','COMPLETE_RESPONSE')",
            name="chk_sub_status",
        ),
        Index("idx_sub_drug_id", "drug_id"),
        Index("idx_sub_status", "status"),
        Index("idx_sub_agency", "agency"),
    )


class SubmissionDocument(Base):
    __tablename__ = "submission_document"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    submission_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("regulatory_submission.id", ondelete="CASCADE"), nullable=False
    )
    doc_type: Mapped[str] = mapped_column(String(50), nullable=False)
    doc_title: Mapped[str] = mapped_column(String(500), nullable=False)
    version: Mapped[float] = mapped_column(Numeric(5, 1), default=1.0)
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer)
    checksum: Mapped[str] = mapped_column(String(128), nullable=False)
    checksum_algorithm: Mapped[str] = mapped_column(String(20), default="SHA-256")
    uploaded_by: Mapped[str] = mapped_column(String(100), nullable=False)
    upload_date: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    effective_date: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    expiry_date: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(20), default="ACTIVE")
    confidentiality: Mapped[str] = mapped_column(String(20), default="CONFIDENTIAL")
    created_date: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    modified_date: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    submission: Mapped[RegulatorySubmission] = relationship(back_populates="documents")

    __table_args__ = (
        UniqueConstraint("submission_id", "doc_type", "version", name="uk_doc_version"),
        Index("idx_doc_submission", "submission_id"),
    )


class ApprovalRecord(Base):
    __tablename__ = "approval_record"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    submission_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("regulatory_submission.id"), nullable=False
    )
    approval_date: Mapped[datetime.date] = mapped_column(DateTime, nullable=False)
    approval_type: Mapped[str] = mapped_column(String(50), nullable=False)
    conditions: Mapped[str | None] = mapped_column(Text)
    expiry_date: Mapped[datetime.date | None] = mapped_column(DateTime)
    approval_number: Mapped[str | None] = mapped_column(String(100))
    approved_indication: Mapped[str | None] = mapped_column(Text)
    market_exclusivity: Mapped[str | None] = mapped_column(String(50))
    exclusivity_expiry: Mapped[datetime.date | None] = mapped_column(DateTime)
    pediatric_exclusivity: Mapped[str] = mapped_column(String(1), default="N")
    created_by: Mapped[str] = mapped_column(String(100), nullable=False)
    created_date: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    modified_date: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    submission: Mapped[RegulatorySubmission] = relationship(back_populates="approvals")

    __table_args__ = (
        Index("idx_appr_submission", "submission_id"),
        Index("idx_appr_date", "approval_date"),
    )


class LabelingChange(Base):
    __tablename__ = "labeling_change"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_id: Mapped[int] = mapped_column(Integer, ForeignKey("drug.drug_id"), nullable=False)
    change_type: Mapped[str] = mapped_column(String(50), nullable=False)
    change_category: Mapped[str] = mapped_column(String(50), default="MINOR")
    effective_date: Mapped[datetime.date] = mapped_column(DateTime, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    rationale: Mapped[str | None] = mapped_column(Text)
    affected_sections: Mapped[str | None] = mapped_column(String(500))
    approved_by: Mapped[str] = mapped_column(String(100), nullable=False)
    reviewed_by: Mapped[str | None] = mapped_column(String(100))
    review_date: Mapped[datetime.date | None] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(20), default="PENDING")
    created_by: Mapped[str] = mapped_column(String(100), nullable=False)
    created_date: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    modified_date: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    drug: Mapped[Drug] = relationship(back_populates="labeling_changes")


class ComplianceRecord(Base):
    __tablename__ = "compliance_record"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_id: Mapped[int] = mapped_column(Integer, ForeignKey("drug.drug_id"), nullable=False)
    requirement_type: Mapped[str] = mapped_column(String(50), nullable=False)
    requirement_desc: Mapped[str] = mapped_column(Text, nullable=False)
    due_date: Mapped[datetime.date] = mapped_column(DateTime, nullable=False)
    completion_date: Mapped[datetime.date | None] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(20), default="PENDING", nullable=False)
    responsible_party: Mapped[str] = mapped_column(String(100), nullable=False)
    escalation_level: Mapped[int] = mapped_column(Integer, default=0)
    risk_score: Mapped[float | None] = mapped_column(Numeric(5, 2))
    linked_submission: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("regulatory_submission.id")
    )
    regulatory_reference: Mapped[str | None] = mapped_column(String(200))
    notes: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(String(100), nullable=False)
    created_date: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    modified_by: Mapped[str | None] = mapped_column(String(100))
    modified_date: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    drug: Mapped[Drug] = relationship(back_populates="compliance_records")
    linked_submission_rel: Mapped[RegulatorySubmission | None] = relationship(
        back_populates="compliance_records"
    )


class AuditTrail(Base):
    __tablename__ = "audit_trail"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    table_name: Mapped[str] = mapped_column(String(50), nullable=False)
    record_id: Mapped[int] = mapped_column(Integer, nullable=False)
    action: Mapped[str] = mapped_column(String(10), nullable=False)
    old_values: Mapped[str | None] = mapped_column(Text)
    new_values: Mapped[str | None] = mapped_column(Text)
    changed_by: Mapped[str] = mapped_column(String(100), nullable=False)
    changed_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    ip_address: Mapped[str | None] = mapped_column(String(45))
    session_id: Mapped[str | None] = mapped_column(String(100))
    application_name: Mapped[str | None] = mapped_column(String(100))
    transaction_id: Mapped[str | None] = mapped_column(String(100))

    __table_args__ = (
        Index("idx_audit_table_record", "table_name", "record_id"),
        Index("idx_audit_changed_by", "changed_by"),
    )


class PseudoRecord(Base):
    __tablename__ = "pseudo_record"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    original_record_id: Mapped[int] = mapped_column(Integer, nullable=False)
    source_table: Mapped[str] = mapped_column(String(50), nullable=False)
    pseudo_type: Mapped[str] = mapped_column(String(30), nullable=False)
    pseudo_value: Mapped[str] = mapped_column(String(256), nullable=False)
    mapping_key: Mapped[str] = mapped_column(String(128), nullable=False)
    hash_algorithm: Mapped[str] = mapped_column(String(20), default="SHA-256")
    salt: Mapped[str | None] = mapped_column(String(64))
    is_active: Mapped[str] = mapped_column(String(1), default="Y")
    created_date: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    expiry_date: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    last_accessed: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    created_by: Mapped[str] = mapped_column(String(100), nullable=False)
    purpose: Mapped[str | None] = mapped_column(String(200))

    reid_logs: Mapped[list[ReidentificationLog]] = relationship(back_populates="pseudo_record")

    __table_args__ = (
        UniqueConstraint(
            "source_table",
            "original_record_id",
            "pseudo_type",
            "mapping_key",
            name="uk_pseudo_mapping",
        ),
        Index("idx_pseudo_value", "pseudo_value"),
        Index("idx_pseudo_original", "original_record_id", "source_table"),
    )


class ReidentificationLog(Base):
    __tablename__ = "reidentification_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pseudo_record_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("pseudo_record.id"), nullable=False
    )
    requested_by: Mapped[str] = mapped_column(String(100), nullable=False)
    authorized_by: Mapped[str] = mapped_column(String(100), nullable=False)
    request_reason: Mapped[str] = mapped_column(Text, nullable=False)
    request_date: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    approval_date: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(20), default="PENDING")
    ip_address: Mapped[str | None] = mapped_column(String(45))

    pseudo_record: Mapped[PseudoRecord] = relationship(back_populates="reid_logs")
