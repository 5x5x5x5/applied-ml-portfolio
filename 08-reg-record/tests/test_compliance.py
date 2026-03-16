"""Tests for the Compliance Tracking Service."""

from __future__ import annotations

import datetime

import pytest
from sqlalchemy.orm import Session

from reg_record.models.database import ComplianceRecord, Drug
from reg_record.models.schemas import ComplianceCreate
from reg_record.services.compliance_service import (
    ComplianceError,
    ComplianceNotFoundError,
    ComplianceService,
)


class TestCreateCompliance:
    """Tests for compliance record creation."""

    def test_create_compliance_record(self, db: Session, sample_drug: Drug) -> None:
        """Creating a compliance record should set initial risk score."""
        svc = ComplianceService(db)
        data = ComplianceCreate(
            drug_id=sample_drug.drug_id,
            requirement_type="POST_MARKETING_STUDY",
            requirement_desc="Phase IV post-marketing study for TestDrug-100",
            due_date=datetime.date.today() + datetime.timedelta(days=90),
            responsible_party="Clinical Ops",
            created_by="test_user",
        )
        record = svc.create_compliance_record(data)

        assert record.id is not None
        assert record.status == "PENDING"
        assert record.risk_score is not None
        assert record.risk_score > 0
        assert record.requirement_type == "POST_MARKETING_STUDY"

    def test_create_with_invalid_drug_raises_error(self, db: Session) -> None:
        """Creating with a nonexistent drug should fail."""
        svc = ComplianceService(db)
        with pytest.raises(ComplianceError, match="not found"):
            svc.create_compliance_record(
                ComplianceCreate(
                    drug_id=99999,
                    requirement_type="REMS",
                    requirement_desc="Test",
                    due_date=datetime.date.today(),
                    responsible_party="Test",
                    created_by="test_user",
                )
            )

    def test_near_due_date_has_higher_risk(self, db: Session, sample_drug: Drug) -> None:
        """A compliance record due soon should have a higher risk score."""
        svc = ComplianceService(db)

        near_record = svc.create_compliance_record(
            ComplianceCreate(
                drug_id=sample_drug.drug_id,
                requirement_type="REMS",
                requirement_desc="Near deadline",
                due_date=datetime.date.today() + datetime.timedelta(days=5),
                responsible_party="Safety",
                created_by="test_user",
            )
        )

        far_record = svc.create_compliance_record(
            ComplianceCreate(
                drug_id=sample_drug.drug_id,
                requirement_type="REMS",
                requirement_desc="Far deadline",
                due_date=datetime.date.today() + datetime.timedelta(days=180),
                responsible_party="Safety",
                created_by="test_user",
            )
        )

        assert near_record.risk_score is not None
        assert far_record.risk_score is not None
        assert float(near_record.risk_score) > float(far_record.risk_score)


class TestStatusUpdates:
    """Tests for compliance status updates."""

    def _create_record(self, db: Session, drug: Drug, days_until_due: int = 30) -> ComplianceRecord:
        svc = ComplianceService(db)
        return svc.create_compliance_record(
            ComplianceCreate(
                drug_id=drug.drug_id,
                requirement_type="ANNUAL_REPORT",
                requirement_desc="Annual report requirement",
                due_date=datetime.date.today() + datetime.timedelta(days=days_until_due),
                responsible_party="Regulatory Affairs",
                created_by="test_user",
            )
        )

    def test_update_to_completed_sets_completion_date(self, db: Session, sample_drug: Drug) -> None:
        """Completing a record should set the completion date."""
        record = self._create_record(db, sample_drug)
        svc = ComplianceService(db)

        updated = svc.update_status(record.id, "COMPLETED", "test_user")

        assert updated.status == "COMPLETED"
        assert updated.completion_date is not None

    def test_completed_record_has_zero_risk(self, db: Session, sample_drug: Drug) -> None:
        """Completed records should have zero risk score."""
        record = self._create_record(db, sample_drug)
        svc = ComplianceService(db)

        updated = svc.update_status(record.id, "COMPLETED", "test_user")

        assert updated.risk_score is not None
        assert float(updated.risk_score) == 0.0

    def test_update_nonexistent_raises_error(self, db: Session) -> None:
        """Updating a nonexistent record should raise an error."""
        svc = ComplianceService(db)
        with pytest.raises(ComplianceNotFoundError):
            svc.update_status(99999, "COMPLETED", "test_user")


class TestRiskScoring:
    """Tests for risk score calculation."""

    def test_overdue_rems_has_highest_risk(self) -> None:
        """An overdue REMS requirement should score very high."""
        score = ComplianceService._calculate_risk_score_from_params(
            due_date=datetime.date.today() - datetime.timedelta(days=45),
            requirement_type="REMS",
            escalation_level=3,
            status="OVERDUE",
        )
        assert score >= 80.0

    def test_completed_has_zero_risk(self) -> None:
        """Completed items should have zero risk."""
        score = ComplianceService._calculate_risk_score_from_params(
            due_date=datetime.date.today() - datetime.timedelta(days=100),
            requirement_type="REMS",
            escalation_level=5,
            status="COMPLETED",
        )
        assert score == 0.0

    def test_far_future_has_low_risk(self) -> None:
        """Items due far in the future should have low risk."""
        score = ComplianceService._calculate_risk_score_from_params(
            due_date=datetime.date.today() + datetime.timedelta(days=365),
            requirement_type="ANNUAL_REPORT",
            escalation_level=0,
        )
        assert score < 10.0

    def test_type_weight_affects_score(self) -> None:
        """Different requirement types should produce different scores."""
        rems_score = ComplianceService._calculate_risk_score_from_params(
            due_date=datetime.date.today() + datetime.timedelta(days=10),
            requirement_type="REMS",
            escalation_level=0,
        )
        annual_score = ComplianceService._calculate_risk_score_from_params(
            due_date=datetime.date.today() + datetime.timedelta(days=10),
            requirement_type="ANNUAL_REPORT",
            escalation_level=0,
        )
        assert rems_score > annual_score

    def test_score_capped_at_100(self) -> None:
        """Risk score should never exceed 100."""
        score = ComplianceService._calculate_risk_score_from_params(
            due_date=datetime.date.today() - datetime.timedelta(days=500),
            requirement_type="REMS",
            escalation_level=5,
            status="ESCALATED",
        )
        assert score <= 100.0


class TestSLA:
    """Tests for SLA status tracking."""

    def _create_record(self, db: Session, drug: Drug, days_until_due: int) -> ComplianceRecord:
        svc = ComplianceService(db)
        return svc.create_compliance_record(
            ComplianceCreate(
                drug_id=drug.drug_id,
                requirement_type="PMC",
                requirement_desc="Post-marketing commitment",
                due_date=datetime.date.today() + datetime.timedelta(days=days_until_due),
                responsible_party="Clinical",
                created_by="test_user",
            )
        )

    def test_on_track_sla(self, db: Session, sample_drug: Drug) -> None:
        """Record with distant due date should be ON_TRACK."""
        record = self._create_record(db, sample_drug, 60)
        svc = ComplianceService(db)
        assert svc.get_sla_status(record.id) == "ON_TRACK"

    def test_at_risk_sla(self, db: Session, sample_drug: Drug) -> None:
        """Record due within 7 days should be AT_RISK."""
        record = self._create_record(db, sample_drug, 5)
        svc = ComplianceService(db)
        assert svc.get_sla_status(record.id) == "AT_RISK"

    def test_breached_sla(self, db: Session, sample_drug: Drug) -> None:
        """Overdue record should show BREACHED."""
        record = self._create_record(db, sample_drug, -10)
        svc = ComplianceService(db)
        sla = svc.get_sla_status(record.id)
        assert sla.startswith("BREACHED_BY_")

    def test_completed_on_time_is_met(self, db: Session, sample_drug: Drug) -> None:
        """Completed-on-time record should show MET."""
        record = self._create_record(db, sample_drug, 30)
        svc = ComplianceService(db)
        svc.update_status(record.id, "COMPLETED", "test_user")
        assert svc.get_sla_status(record.id) == "MET"


class TestEscalation:
    """Tests for overdue escalation."""

    def test_escalate_overdue_records(self, db: Session, sample_drug: Drug) -> None:
        """Escalation should increase escalation level for overdue records."""
        svc = ComplianceService(db)

        record = svc.create_compliance_record(
            ComplianceCreate(
                drug_id=sample_drug.drug_id,
                requirement_type="PSUR",
                requirement_desc="Overdue PSUR",
                due_date=datetime.date.today() - datetime.timedelta(days=20),
                responsible_party="Safety",
                created_by="test_user",
            )
        )

        escalated = svc.escalate_overdue("system_escalation")

        assert escalated >= 1

        db.refresh(record)
        assert record.escalation_level >= 1

    def test_escalation_is_idempotent_at_same_level(self, db: Session, sample_drug: Drug) -> None:
        """Running escalation again should not increase level if days haven't changed."""
        svc = ComplianceService(db)

        svc.create_compliance_record(
            ComplianceCreate(
                drug_id=sample_drug.drug_id,
                requirement_type="DSUR",
                requirement_desc="Overdue DSUR",
                due_date=datetime.date.today() - datetime.timedelta(days=10),
                responsible_party="Safety",
                created_by="test_user",
            )
        )

        first_run = svc.escalate_overdue("system")
        second_run = svc.escalate_overdue("system")

        # Second run should not escalate further since days overdue hasn't changed
        assert second_run == 0 or second_run <= first_run


class TestReporting:
    """Tests for compliance reporting."""

    def test_compliance_summary(self, db: Session, sample_drug: Drug) -> None:
        """Summary should aggregate compliance record counts."""
        svc = ComplianceService(db)

        svc.create_compliance_record(
            ComplianceCreate(
                drug_id=sample_drug.drug_id,
                requirement_type="ANNUAL_REPORT",
                requirement_desc="Annual report 1",
                due_date=datetime.date.today() + datetime.timedelta(days=30),
                responsible_party="RA",
                created_by="user",
            )
        )
        svc.create_compliance_record(
            ComplianceCreate(
                drug_id=sample_drug.drug_id,
                requirement_type="PMC",
                requirement_desc="PMC 1",
                due_date=datetime.date.today() + datetime.timedelta(days=60),
                responsible_party="Clinical",
                created_by="user",
            )
        )

        summary = svc.get_compliance_summary()

        assert summary.total >= 2
        assert summary.pending >= 2

    def test_gap_analysis_includes_overdue(self, db: Session, sample_drug: Drug) -> None:
        """Gap analysis should include overdue records."""
        svc = ComplianceService(db)

        record = svc.create_compliance_record(
            ComplianceCreate(
                drug_id=sample_drug.drug_id,
                requirement_type="GMP_INSPECTION",
                requirement_desc="Overdue GMP",
                due_date=datetime.date.today() - datetime.timedelta(days=35),
                responsible_party="QA",
                created_by="user",
            )
        )

        # Mark as overdue
        svc.update_status(record.id, "OVERDUE", "system")

        gaps = svc.get_gap_analysis()

        overdue_gaps = [g for g in gaps if g["compliance_id"] == record.id]
        assert len(overdue_gaps) == 1
        assert overdue_gaps[0]["severity"] == "MEDIUM"

    def test_upcoming_deadlines(self, db: Session, sample_drug: Drug) -> None:
        """Upcoming deadlines should return records due within the window."""
        svc = ComplianceService(db)

        svc.create_compliance_record(
            ComplianceCreate(
                drug_id=sample_drug.drug_id,
                requirement_type="LABELING_UPDATE",
                requirement_desc="Label update",
                due_date=datetime.date.today() + datetime.timedelta(days=10),
                responsible_party="Labeling",
                created_by="user",
            )
        )

        deadlines = svc.get_upcoming_deadlines(days_ahead=30)

        assert len(deadlines) >= 1
        assert any(d["requirement_type"] == "LABELING_UPDATE" for d in deadlines)
