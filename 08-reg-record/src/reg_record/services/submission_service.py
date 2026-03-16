"""
Submission Management Service

Manages the lifecycle of regulatory submissions including state machine
enforcement, document management, timeline tracking, and notification triggers.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from reg_record.models.database import (
    AuditTrail,
    Drug,
    RegulatorySubmission,
    SubmissionDocument,
)
from reg_record.models.schemas import SubmissionCreate, SubmissionUpdate

logger = logging.getLogger(__name__)


# Valid state transitions for the submission workflow
STATE_TRANSITIONS: dict[str, set[str]] = {
    "DRAFT": {"SUBMITTED", "WITHDRAWN"},
    "SUBMITTED": {"UNDER_REVIEW", "WITHDRAWN"},
    "UNDER_REVIEW": {"APPROVED", "REJECTED", "ON_HOLD", "COMPLETE_RESPONSE", "WITHDRAWN"},
    "ON_HOLD": {"UNDER_REVIEW", "WITHDRAWN"},
    "COMPLETE_RESPONSE": {"SUBMITTED", "WITHDRAWN"},
    "REJECTED": {"DRAFT", "WITHDRAWN"},
    "APPROVED": {"WITHDRAWN"},  # Terminal state, but can be withdrawn in rare cases
    "WITHDRAWN": set(),  # Terminal state
}


class SubmissionError(Exception):
    """Base exception for submission operations."""


class InvalidTransitionError(SubmissionError):
    """Raised when an invalid status transition is attempted."""


class SubmissionNotFoundError(SubmissionError):
    """Raised when a submission is not found."""


class DocumentChecksumError(SubmissionError):
    """Raised when document checksum verification fails."""


class SubmissionService:
    """Service for managing regulatory submission lifecycle."""

    def __init__(self, db: Session) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    @staticmethod
    def is_valid_transition(current_status: str, new_status: str) -> bool:
        """Check if a status transition is valid per the state machine."""
        valid_next = STATE_TRANSITIONS.get(current_status, set())
        return new_status in valid_next

    @staticmethod
    def get_valid_transitions(current_status: str) -> set[str]:
        """Get all valid next states from the current status."""
        return STATE_TRANSITIONS.get(current_status, set())

    # ------------------------------------------------------------------
    # Tracking number generation
    # ------------------------------------------------------------------

    def _generate_tracking_number(self, agency: str, submission_type: str) -> str:
        """Generate a unique tracking number."""
        now = datetime.now(UTC)
        year = now.year

        # Count existing submissions for this agency/type this year
        count = (
            self._db.execute(
                select(func.count())
                .select_from(RegulatorySubmission)
                .where(
                    and_(
                        RegulatorySubmission.agency == agency,
                        func.extract("year", RegulatorySubmission.created_date) == year,
                    )
                )
            ).scalar()
            or 0
        )

        seq = count + 1
        return f"{agency}-{submission_type[:3]}-{year}-{seq:06d}"

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def create_submission(self, data: SubmissionCreate) -> RegulatorySubmission:
        """
        Create a new regulatory submission in DRAFT status.

        Validates that the drug exists and is active, generates a tracking
        number, and records the creation in the audit trail.
        """
        # Validate drug
        drug = self._db.execute(
            select(Drug).where(and_(Drug.drug_id == data.drug_id, Drug.active_flag == "Y"))
        ).scalar_one_or_none()

        if drug is None:
            raise SubmissionError(f"Drug ID {data.drug_id} does not exist or is inactive")

        tracking_number = self._generate_tracking_number(data.agency, data.submission_type.value)

        submission = RegulatorySubmission(
            drug_id=data.drug_id,
            submission_type=data.submission_type.value,
            status="DRAFT",
            agency=data.agency,
            tracking_number=tracking_number,
            priority=data.priority.value,
            assigned_to=data.assigned_to,
            description=data.description,
            created_by=data.created_by,
            modified_by=data.created_by,
        )
        self._db.add(submission)
        self._db.flush()

        # Record audit trail
        self._record_audit(
            "REGULATORY_SUBMISSION",
            submission.id,
            "INSERT",
            None,
            {
                "drug_id": data.drug_id,
                "submission_type": data.submission_type.value,
                "status": "DRAFT",
                "agency": data.agency,
                "tracking_number": tracking_number,
            },
            data.created_by,
        )

        self._db.commit()
        self._db.refresh(submission)

        logger.info(
            "Created submission %d (tracking: %s) for drug %d",
            submission.id,
            tracking_number,
            data.drug_id,
        )
        return submission

    def update_submission(self, submission_id: int, data: SubmissionUpdate) -> RegulatorySubmission:
        """
        Update a submission with state machine enforcement.

        If status is being changed, validates the transition. Automatically
        sets submitted_date when transitioning to SUBMITTED.
        """
        submission = self._get_submission_or_raise(submission_id)
        old_values: dict[str, object] = {}
        new_values: dict[str, object] = {}

        if data.status is not None and data.status.value != submission.status:
            old_status = submission.status
            new_status = data.status.value

            if not self.is_valid_transition(old_status, new_status):
                valid = self.get_valid_transitions(old_status)
                raise InvalidTransitionError(
                    f"Cannot transition from {old_status} to {new_status}. "
                    f"Valid transitions: {', '.join(sorted(valid)) or 'none'}"
                )

            old_values["status"] = old_status
            submission.status = new_status
            new_values["status"] = new_status

            # Auto-set submitted_date
            if new_status == "SUBMITTED" and submission.submitted_date is None:
                submission.submitted_date = datetime.now(UTC)
                new_values["submitted_date"] = submission.submitted_date.isoformat()

            logger.info(
                "Submission %d status: %s -> %s",
                submission_id,
                old_status,
                new_status,
            )

        if data.priority is not None:
            old_values["priority"] = submission.priority
            submission.priority = data.priority.value
            new_values["priority"] = data.priority.value

        if data.assigned_to is not None:
            old_values["assigned_to"] = submission.assigned_to
            submission.assigned_to = data.assigned_to
            new_values["assigned_to"] = data.assigned_to

        if data.description is not None:
            old_values["description"] = (
                str(submission.description)[:200] if submission.description else None
            )
            submission.description = data.description
            new_values["description"] = data.description[:200]

        if data.target_date is not None:
            old_values["target_date"] = (
                submission.target_date.isoformat() if submission.target_date else None
            )
            submission.target_date = data.target_date
            new_values["target_date"] = data.target_date.isoformat()

        submission.modified_by = data.modified_by
        submission.modified_date = datetime.now(UTC)

        self._record_audit(
            "REGULATORY_SUBMISSION",
            submission_id,
            "UPDATE",
            old_values,
            new_values,
            data.modified_by,
        )

        self._db.commit()
        self._db.refresh(submission)
        return submission

    def get_submission(self, submission_id: int) -> RegulatorySubmission:
        """Get a submission by ID."""
        return self._get_submission_or_raise(submission_id)

    def list_submissions(
        self,
        agency: str | None = None,
        status: str | None = None,
        drug_id: int | None = None,
        submission_type: str | None = None,
        assigned_to: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[RegulatorySubmission], int]:
        """
        List submissions with optional filters and pagination.

        Returns tuple of (submissions, total_count).
        """
        query = select(RegulatorySubmission)
        count_query = select(func.count()).select_from(RegulatorySubmission)

        filters = []
        if agency:
            filters.append(RegulatorySubmission.agency == agency)
        if status:
            filters.append(RegulatorySubmission.status == status)
        if drug_id:
            filters.append(RegulatorySubmission.drug_id == drug_id)
        if submission_type:
            filters.append(RegulatorySubmission.submission_type == submission_type)
        if assigned_to:
            filters.append(RegulatorySubmission.assigned_to == assigned_to)

        if filters:
            query = query.where(and_(*filters))
            count_query = count_query.where(and_(*filters))

        total = self._db.execute(count_query).scalar() or 0

        query = (
            query.order_by(RegulatorySubmission.created_date.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )

        submissions = list(self._db.execute(query).scalars().all())
        return submissions, total

    def delete_submission(self, submission_id: int, deleted_by: str) -> None:
        """Delete a submission (only allowed for DRAFT status)."""
        submission = self._get_submission_or_raise(submission_id)

        if submission.status != "DRAFT":
            raise SubmissionError(
                f"Cannot delete submission in {submission.status} status. "
                "Only DRAFT submissions can be deleted."
            )

        self._record_audit(
            "REGULATORY_SUBMISSION",
            submission_id,
            "DELETE",
            {
                "drug_id": submission.drug_id,
                "submission_type": submission.submission_type,
                "status": submission.status,
                "tracking_number": submission.tracking_number,
            },
            None,
            deleted_by,
        )

        self._db.delete(submission)
        self._db.commit()
        logger.info("Deleted submission %d by %s", submission_id, deleted_by)

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

    def add_document(
        self,
        submission_id: int,
        doc_type: str,
        doc_title: str,
        file_path: str,
        checksum: str,
        uploaded_by: str,
        file_size_bytes: int | None = None,
        confidentiality: str = "CONFIDENTIAL",
    ) -> SubmissionDocument:
        """
        Add a document to a submission.

        Auto-increments the version if a document of the same type already
        exists. Marks previous versions as SUPERSEDED.
        """
        self._get_submission_or_raise(submission_id)

        # Find current max version for this doc_type
        max_version = self._db.execute(
            select(func.max(SubmissionDocument.version)).where(
                and_(
                    SubmissionDocument.submission_id == submission_id,
                    SubmissionDocument.doc_type == doc_type,
                    SubmissionDocument.status == "ACTIVE",
                )
            )
        ).scalar()

        new_version = (max_version or 0) + 1

        # Mark previous versions as superseded
        if max_version is not None:
            self._db.query(SubmissionDocument).filter(
                and_(
                    SubmissionDocument.submission_id == submission_id,
                    SubmissionDocument.doc_type == doc_type,
                    SubmissionDocument.status == "ACTIVE",
                )
            ).update({"status": "SUPERSEDED"})

        doc = SubmissionDocument(
            submission_id=submission_id,
            doc_type=doc_type,
            doc_title=doc_title,
            version=new_version,
            file_path=file_path,
            file_size_bytes=file_size_bytes,
            checksum=checksum,
            uploaded_by=uploaded_by,
            confidentiality=confidentiality,
        )
        self._db.add(doc)

        self._record_audit(
            "SUBMISSION_DOCUMENT",
            submission_id,
            "INSERT",
            None,
            {
                "doc_type": doc_type,
                "version": new_version,
                "checksum": checksum,
            },
            uploaded_by,
        )

        self._db.commit()
        self._db.refresh(doc)

        logger.info(
            "Added document %d (type=%s, v%s) to submission %d",
            doc.id,
            doc_type,
            new_version,
            submission_id,
        )
        return doc

    def verify_document_checksum(self, document_id: int, expected_checksum: str) -> bool:
        """Verify a document's checksum matches the expected value."""
        doc = self._db.execute(
            select(SubmissionDocument).where(SubmissionDocument.id == document_id)
        ).scalar_one_or_none()

        if doc is None:
            raise SubmissionError(f"Document {document_id} not found")

        return doc.checksum == expected_checksum

    @staticmethod
    def compute_file_checksum(file_path: str) -> str:
        """Compute SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def get_documents(
        self, submission_id: int, active_only: bool = True
    ) -> list[SubmissionDocument]:
        """Get all documents for a submission."""
        query = select(SubmissionDocument).where(SubmissionDocument.submission_id == submission_id)
        if active_only:
            query = query.where(SubmissionDocument.status == "ACTIVE")
        query = query.order_by(SubmissionDocument.doc_type, SubmissionDocument.version.desc())
        return list(self._db.execute(query).scalars().all())

    # ------------------------------------------------------------------
    # Timeline and history
    # ------------------------------------------------------------------

    def get_submission_timeline(self, submission_id: int) -> list[dict[str, object]]:
        """Get the complete timeline of a submission from audit trail."""
        self._get_submission_or_raise(submission_id)

        entries = (
            self._db.execute(
                select(AuditTrail)
                .where(
                    and_(
                        AuditTrail.table_name == "REGULATORY_SUBMISSION",
                        AuditTrail.record_id == submission_id,
                    )
                )
                .order_by(AuditTrail.changed_at.asc())
            )
            .scalars()
            .all()
        )

        return [
            {
                "audit_id": entry.id,
                "action": entry.action,
                "old_values": entry.old_values,
                "new_values": entry.new_values,
                "changed_by": entry.changed_by,
                "changed_at": entry.changed_at.isoformat() if entry.changed_at else None,
                "ip_address": entry.ip_address,
            }
            for entry in entries
        ]

    def get_pipeline_summary(self, agency: str | None = None) -> dict[str, int]:
        """Get submission counts by status for the pipeline view."""
        query = select(
            RegulatorySubmission.status,
            func.count().label("count"),
        ).group_by(RegulatorySubmission.status)

        if agency:
            query = query.where(RegulatorySubmission.agency == agency)

        results = self._db.execute(query).all()
        return {row.status: row.count for row in results}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_submission_or_raise(self, submission_id: int) -> RegulatorySubmission:
        """Get submission by ID or raise SubmissionNotFoundError."""
        submission = self._db.execute(
            select(RegulatorySubmission).where(RegulatorySubmission.id == submission_id)
        ).scalar_one_or_none()

        if submission is None:
            raise SubmissionNotFoundError(f"Submission {submission_id} not found")
        return submission

    def _record_audit(
        self,
        table_name: str,
        record_id: int,
        action: str,
        old_values: dict[str, object] | None,
        new_values: dict[str, object] | None,
        changed_by: str,
        ip_address: str | None = None,
    ) -> None:
        """Record an entry in the audit trail."""
        audit = AuditTrail(
            table_name=table_name,
            record_id=record_id,
            action=action,
            old_values=json.dumps(old_values, default=str) if old_values else None,
            new_values=json.dumps(new_values, default=str) if new_values else None,
            changed_by=changed_by,
            ip_address=ip_address,
            application_name="RegRecord-API",
        )
        self._db.add(audit)
