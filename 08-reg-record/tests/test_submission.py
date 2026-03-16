"""Tests for the Submission Management Service."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from reg_record.models.database import AuditTrail, Drug, RegulatorySubmission
from reg_record.models.schemas import (
    Priority,
    SubmissionCreate,
    SubmissionStatus,
    SubmissionType,
    SubmissionUpdate,
)
from reg_record.services.submission_service import (
    InvalidTransitionError,
    SubmissionError,
    SubmissionNotFoundError,
    SubmissionService,
)


class TestCreateSubmission:
    """Tests for submission creation."""

    def test_create_submission_returns_draft(self, db: Session, sample_drug: Drug) -> None:
        """A new submission should start in DRAFT status."""
        svc = SubmissionService(db)
        data = SubmissionCreate(
            drug_id=sample_drug.drug_id,
            submission_type=SubmissionType.NDA,
            agency="FDA",
            priority=Priority.STANDARD,
            created_by="test_user",
        )
        submission = svc.create_submission(data)

        assert submission.id is not None
        assert submission.status == "DRAFT"
        assert submission.submission_type == "NDA"
        assert submission.agency == "FDA"
        assert submission.tracking_number is not None
        assert submission.tracking_number.startswith("FDA-NDA-")

    def test_create_submission_generates_unique_tracking(
        self, db: Session, sample_drug: Drug
    ) -> None:
        """Each submission should get a unique tracking number."""
        svc = SubmissionService(db)

        sub1 = svc.create_submission(
            SubmissionCreate(
                drug_id=sample_drug.drug_id,
                submission_type=SubmissionType.NDA,
                agency="FDA",
                created_by="user1",
            )
        )
        sub2 = svc.create_submission(
            SubmissionCreate(
                drug_id=sample_drug.drug_id,
                submission_type=SubmissionType.NDA,
                agency="FDA",
                created_by="user2",
            )
        )

        assert sub1.tracking_number != sub2.tracking_number

    def test_create_submission_records_audit_trail(self, db: Session, sample_drug: Drug) -> None:
        """Creating a submission should produce an audit trail entry."""
        svc = SubmissionService(db)
        submission = svc.create_submission(
            SubmissionCreate(
                drug_id=sample_drug.drug_id,
                submission_type=SubmissionType.BLA,
                agency="FDA",
                created_by="test_user",
            )
        )

        audit = (
            db.query(AuditTrail)
            .filter(
                AuditTrail.table_name == "REGULATORY_SUBMISSION",
                AuditTrail.record_id == submission.id,
                AuditTrail.action == "INSERT",
            )
            .first()
        )

        assert audit is not None
        assert audit.changed_by == "test_user"
        assert '"status": "DRAFT"' in (audit.new_values or "")

    def test_create_submission_invalid_drug_raises_error(self, db: Session) -> None:
        """Creating with a nonexistent drug ID should fail."""
        svc = SubmissionService(db)
        with pytest.raises(SubmissionError, match="does not exist"):
            svc.create_submission(
                SubmissionCreate(
                    drug_id=99999,
                    submission_type=SubmissionType.NDA,
                    agency="FDA",
                    created_by="test_user",
                )
            )


class TestStatusTransitions:
    """Tests for submission status state machine."""

    def _create_draft(self, db: Session, drug: Drug) -> RegulatorySubmission:
        svc = SubmissionService(db)
        return svc.create_submission(
            SubmissionCreate(
                drug_id=drug.drug_id,
                submission_type=SubmissionType.NDA,
                agency="FDA",
                created_by="test_user",
            )
        )

    def test_valid_transition_draft_to_submitted(self, db: Session, sample_drug: Drug) -> None:
        """DRAFT -> SUBMITTED should succeed."""
        submission = self._create_draft(db, sample_drug)
        svc = SubmissionService(db)

        updated = svc.update_submission(
            submission.id,
            SubmissionUpdate(
                status=SubmissionStatus.SUBMITTED,
                modified_by="test_user",
            ),
        )

        assert updated.status == "SUBMITTED"
        assert updated.submitted_date is not None

    def test_valid_transition_submitted_to_under_review(
        self, db: Session, sample_drug: Drug
    ) -> None:
        """SUBMITTED -> UNDER_REVIEW should succeed."""
        submission = self._create_draft(db, sample_drug)
        svc = SubmissionService(db)

        svc.update_submission(
            submission.id, SubmissionUpdate(status=SubmissionStatus.SUBMITTED, modified_by="user1")
        )
        updated = svc.update_submission(
            submission.id,
            SubmissionUpdate(status=SubmissionStatus.UNDER_REVIEW, modified_by="user2"),
        )

        assert updated.status == "UNDER_REVIEW"

    def test_valid_transition_under_review_to_approved(
        self, db: Session, sample_drug: Drug
    ) -> None:
        """UNDER_REVIEW -> APPROVED should succeed."""
        submission = self._create_draft(db, sample_drug)
        svc = SubmissionService(db)

        svc.update_submission(
            submission.id, SubmissionUpdate(status=SubmissionStatus.SUBMITTED, modified_by="user1")
        )
        svc.update_submission(
            submission.id,
            SubmissionUpdate(status=SubmissionStatus.UNDER_REVIEW, modified_by="user2"),
        )
        updated = svc.update_submission(
            submission.id, SubmissionUpdate(status=SubmissionStatus.APPROVED, modified_by="user3")
        )

        assert updated.status == "APPROVED"

    def test_invalid_transition_draft_to_approved_fails(
        self, db: Session, sample_drug: Drug
    ) -> None:
        """DRAFT -> APPROVED should fail (skipping intermediate states)."""
        submission = self._create_draft(db, sample_drug)
        svc = SubmissionService(db)

        with pytest.raises(InvalidTransitionError, match="Cannot transition"):
            svc.update_submission(
                submission.id,
                SubmissionUpdate(status=SubmissionStatus.APPROVED, modified_by="test_user"),
            )

    def test_invalid_transition_draft_to_under_review_fails(
        self, db: Session, sample_drug: Drug
    ) -> None:
        """DRAFT -> UNDER_REVIEW should fail."""
        submission = self._create_draft(db, sample_drug)
        svc = SubmissionService(db)

        with pytest.raises(InvalidTransitionError):
            svc.update_submission(
                submission.id,
                SubmissionUpdate(status=SubmissionStatus.UNDER_REVIEW, modified_by="test_user"),
            )

    def test_any_state_can_withdraw(self, db: Session, sample_drug: Drug) -> None:
        """Any state should be able to transition to WITHDRAWN."""
        submission = self._create_draft(db, sample_drug)
        svc = SubmissionService(db)

        # DRAFT -> WITHDRAWN
        updated = svc.update_submission(
            submission.id,
            SubmissionUpdate(status=SubmissionStatus.WITHDRAWN, modified_by="test_user"),
        )
        assert updated.status == "WITHDRAWN"

    def test_on_hold_returns_to_under_review(self, db: Session, sample_drug: Drug) -> None:
        """ON_HOLD -> UNDER_REVIEW should succeed."""
        submission = self._create_draft(db, sample_drug)
        svc = SubmissionService(db)

        svc.update_submission(
            submission.id, SubmissionUpdate(status=SubmissionStatus.SUBMITTED, modified_by="u1")
        )
        svc.update_submission(
            submission.id, SubmissionUpdate(status=SubmissionStatus.UNDER_REVIEW, modified_by="u2")
        )
        svc.update_submission(
            submission.id, SubmissionUpdate(status=SubmissionStatus.ON_HOLD, modified_by="u3")
        )
        updated = svc.update_submission(
            submission.id, SubmissionUpdate(status=SubmissionStatus.UNDER_REVIEW, modified_by="u4")
        )

        assert updated.status == "UNDER_REVIEW"

    def test_is_valid_transition_static_method(self) -> None:
        """Static method should correctly identify valid/invalid transitions."""
        assert SubmissionService.is_valid_transition("DRAFT", "SUBMITTED") is True
        assert SubmissionService.is_valid_transition("DRAFT", "APPROVED") is False
        assert SubmissionService.is_valid_transition("UNDER_REVIEW", "APPROVED") is True
        assert SubmissionService.is_valid_transition("APPROVED", "SUBMITTED") is False


class TestDocumentManagement:
    """Tests for document management."""

    def _create_submission(self, db: Session, drug: Drug) -> RegulatorySubmission:
        svc = SubmissionService(db)
        return svc.create_submission(
            SubmissionCreate(
                drug_id=drug.drug_id,
                submission_type=SubmissionType.NDA,
                agency="FDA",
                created_by="test_user",
            )
        )

    def test_add_document(self, db: Session, sample_drug: Drug) -> None:
        """Adding a document should succeed."""
        submission = self._create_submission(db, sample_drug)
        svc = SubmissionService(db)

        doc = svc.add_document(
            submission_id=submission.id,
            doc_type="COVER_LETTER",
            doc_title="Cover Letter for NDA Submission",
            file_path="/docs/cover_letter.pdf",
            checksum="abc123def456",
            uploaded_by="test_user",
        )

        assert doc.id is not None
        assert doc.doc_type == "COVER_LETTER"
        assert doc.version == 1
        assert doc.status == "ACTIVE"

    def test_add_same_doc_type_increments_version(self, db: Session, sample_drug: Drug) -> None:
        """Adding a second document of the same type should increment version."""
        submission = self._create_submission(db, sample_drug)
        svc = SubmissionService(db)

        doc1 = svc.add_document(
            submission_id=submission.id,
            doc_type="MODULE_1",
            doc_title="Module 1 v1",
            file_path="/docs/mod1_v1.pdf",
            checksum="hash1",
            uploaded_by="user1",
        )
        doc2 = svc.add_document(
            submission_id=submission.id,
            doc_type="MODULE_1",
            doc_title="Module 1 v2",
            file_path="/docs/mod1_v2.pdf",
            checksum="hash2",
            uploaded_by="user2",
        )

        assert doc1.version == 1
        assert doc2.version == 2

        # First version should be superseded
        db.refresh(doc1)
        assert doc1.status == "SUPERSEDED"

    def test_verify_document_checksum(self, db: Session, sample_drug: Drug) -> None:
        """Checksum verification should match stored value."""
        submission = self._create_submission(db, sample_drug)
        svc = SubmissionService(db)

        doc = svc.add_document(
            submission_id=submission.id,
            doc_type="LABEL",
            doc_title="Drug Label",
            file_path="/docs/label.pdf",
            checksum="correct_checksum_value",
            uploaded_by="user",
        )

        assert svc.verify_document_checksum(doc.id, "correct_checksum_value") is True
        assert svc.verify_document_checksum(doc.id, "wrong_checksum") is False


class TestListAndDelete:
    """Tests for listing and deleting submissions."""

    def test_list_submissions_with_filters(self, db: Session, sample_drug: Drug) -> None:
        """Listing submissions should support filtering."""
        svc = SubmissionService(db)

        svc.create_submission(
            SubmissionCreate(
                drug_id=sample_drug.drug_id,
                submission_type=SubmissionType.NDA,
                agency="FDA",
                priority=Priority.PRIORITY,
                created_by="user1",
            )
        )
        svc.create_submission(
            SubmissionCreate(
                drug_id=sample_drug.drug_id,
                submission_type=SubmissionType.MAA,
                agency="EMA",
                created_by="user2",
            )
        )

        # Filter by agency
        fda_subs, fda_total = svc.list_submissions(agency="FDA")
        assert fda_total >= 1
        assert all(s.agency == "FDA" for s in fda_subs)

    def test_delete_draft_submission(self, db: Session, sample_drug: Drug) -> None:
        """Deleting a DRAFT submission should succeed."""
        svc = SubmissionService(db)
        submission = svc.create_submission(
            SubmissionCreate(
                drug_id=sample_drug.drug_id,
                submission_type=SubmissionType.IND,
                agency="FDA",
                created_by="test_user",
            )
        )

        svc.delete_submission(submission.id, "test_user")

        with pytest.raises(SubmissionNotFoundError):
            svc.get_submission(submission.id)

    def test_delete_non_draft_raises_error(self, db: Session, sample_drug: Drug) -> None:
        """Deleting a non-DRAFT submission should fail."""
        svc = SubmissionService(db)
        submission = svc.create_submission(
            SubmissionCreate(
                drug_id=sample_drug.drug_id,
                submission_type=SubmissionType.NDA,
                agency="FDA",
                created_by="test_user",
            )
        )

        svc.update_submission(
            submission.id, SubmissionUpdate(status=SubmissionStatus.SUBMITTED, modified_by="user")
        )

        with pytest.raises(SubmissionError, match="Cannot delete"):
            svc.delete_submission(submission.id, "test_user")

    def test_get_nonexistent_submission_raises_error(self, db: Session) -> None:
        """Getting a nonexistent submission should raise SubmissionNotFoundError."""
        svc = SubmissionService(db)
        with pytest.raises(SubmissionNotFoundError):
            svc.get_submission(99999)


class TestTimeline:
    """Tests for submission timeline."""

    def test_timeline_captures_status_changes(self, db: Session, sample_drug: Drug) -> None:
        """Timeline should show all status transitions."""
        svc = SubmissionService(db)
        submission = svc.create_submission(
            SubmissionCreate(
                drug_id=sample_drug.drug_id,
                submission_type=SubmissionType.NDA,
                agency="FDA",
                created_by="creator",
            )
        )

        svc.update_submission(
            submission.id,
            SubmissionUpdate(status=SubmissionStatus.SUBMITTED, modified_by="submitter"),
        )

        timeline = svc.get_submission_timeline(submission.id)

        assert len(timeline) >= 2  # INSERT + UPDATE
        assert timeline[0]["action"] == "INSERT"
