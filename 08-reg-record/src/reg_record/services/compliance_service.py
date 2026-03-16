"""
Compliance Tracking Service

Monitors regulatory compliance requirements, calculates risk scores,
manages SLA tracking, handles escalation, and generates compliance reports.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime

from sqlalchemy import and_, case, func, select
from sqlalchemy.orm import Session

from reg_record.models.database import (
    AuditTrail,
    ComplianceRecord,
    Drug,
    RegulatorySubmission,
)
from reg_record.models.schemas import ComplianceCreate, ComplianceSummary

logger = logging.getLogger(__name__)

# Risk weights by requirement type
REQUIREMENT_RISK_WEIGHTS: dict[str, float] = {
    "REMS": 2.0,
    "POST_MARKETING_STUDY": 1.8,
    "PHARMACOVIGILANCE": 1.7,
    "GMP_INSPECTION": 1.6,
    "PMR": 1.5,
    "LABELING_UPDATE": 1.4,
    "PMC": 1.3,
    "PEDIATRIC_STUDY": 1.2,
    "PSUR": 1.1,
    "DSUR": 1.1,
    "ANNUAL_REPORT": 1.0,
    "PERIODIC_REPORT": 1.0,
    "RISK_EVALUATION": 1.5,
    "AD_HOC": 0.8,
}

VALID_REQUIREMENT_TYPES = set(REQUIREMENT_RISK_WEIGHTS.keys())


class ComplianceError(Exception):
    """Base exception for compliance operations."""


class ComplianceNotFoundError(ComplianceError):
    """Raised when a compliance record is not found."""


class ComplianceService:
    """Service for managing regulatory compliance tracking."""

    def __init__(self, db: Session) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_compliance_record(self, data: ComplianceCreate) -> ComplianceRecord:
        """
        Create a new compliance record.

        Validates the drug, calculates initial risk score based on due date
        proximity and requirement type.
        """
        # Validate drug
        drug = self._db.execute(
            select(Drug).where(Drug.drug_id == data.drug_id)
        ).scalar_one_or_none()

        if drug is None:
            raise ComplianceError(f"Drug ID {data.drug_id} not found")

        # Validate linked submission if provided
        if data.linked_submission is not None:
            sub = self._db.execute(
                select(RegulatorySubmission).where(
                    RegulatorySubmission.id == data.linked_submission
                )
            ).scalar_one_or_none()
            if sub is None:
                raise ComplianceError(f"Linked submission {data.linked_submission} not found")

        # Calculate initial risk score
        risk_score = self._calculate_risk_score_from_params(
            due_date=data.due_date,
            requirement_type=data.requirement_type,
            escalation_level=0,
        )

        record = ComplianceRecord(
            drug_id=data.drug_id,
            requirement_type=data.requirement_type,
            requirement_desc=data.requirement_desc,
            due_date=datetime.combine(data.due_date, datetime.min.time()),
            status="PENDING",
            responsible_party=data.responsible_party,
            risk_score=risk_score,
            linked_submission=data.linked_submission,
            regulatory_reference=data.regulatory_reference,
            created_by=data.created_by,
            modified_by=data.created_by,
        )
        self._db.add(record)
        self._db.flush()

        self._record_audit(
            record.id,
            "INSERT",
            None,
            {
                "drug_id": data.drug_id,
                "requirement_type": data.requirement_type,
                "due_date": data.due_date.isoformat(),
                "status": "PENDING",
                "risk_score": risk_score,
            },
            data.created_by,
        )

        self._db.commit()
        self._db.refresh(record)

        logger.info(
            "Created compliance record %d for drug %d (type=%s, due=%s, risk=%.1f)",
            record.id,
            data.drug_id,
            data.requirement_type,
            data.due_date,
            risk_score,
        )
        return record

    def update_status(
        self,
        compliance_id: int,
        new_status: str,
        modified_by: str,
        notes: str | None = None,
    ) -> ComplianceRecord:
        """Update the status of a compliance record."""
        record = self._get_record_or_raise(compliance_id)
        old_status = record.status
        old_values = {"status": old_status}
        new_values: dict[str, object] = {"status": new_status}

        record.status = new_status
        record.modified_by = modified_by
        record.modified_date = datetime.now(UTC)

        if new_status == "COMPLETED" and record.completion_date is None:
            record.completion_date = datetime.now(UTC)
            new_values["completion_date"] = record.completion_date.isoformat()

        if notes:
            record.notes = notes

        # Recalculate risk score
        record.risk_score = self._calculate_risk_score_from_params(
            due_date=record.due_date.date()
            if isinstance(record.due_date, datetime)
            else record.due_date,
            requirement_type=record.requirement_type,
            escalation_level=record.escalation_level,
            status=new_status,
        )
        new_values["risk_score"] = record.risk_score

        self._record_audit(compliance_id, "UPDATE", old_values, new_values, modified_by)
        self._db.commit()
        self._db.refresh(record)

        logger.info(
            "Compliance %d status: %s -> %s (by %s)",
            compliance_id,
            old_status,
            new_status,
            modified_by,
        )
        return record

    def get_compliance_record(self, compliance_id: int) -> ComplianceRecord:
        """Get a compliance record by ID."""
        return self._get_record_or_raise(compliance_id)

    def list_compliance_records(
        self,
        drug_id: int | None = None,
        status: str | None = None,
        responsible_party: str | None = None,
        due_before: date | None = None,
        include_completed: bool = False,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[ComplianceRecord], int]:
        """List compliance records with filters and pagination."""
        query = select(ComplianceRecord)
        count_query = select(func.count()).select_from(ComplianceRecord)

        filters = []
        if drug_id:
            filters.append(ComplianceRecord.drug_id == drug_id)
        if status:
            filters.append(ComplianceRecord.status == status)
        if responsible_party:
            filters.append(ComplianceRecord.responsible_party == responsible_party)
        if due_before:
            filters.append(
                ComplianceRecord.due_date <= datetime.combine(due_before, datetime.min.time())
            )
        if not include_completed:
            filters.append(ComplianceRecord.status.notin_(["COMPLETED", "WAIVED", "CANCELLED"]))

        if filters:
            query = query.where(and_(*filters))
            count_query = count_query.where(and_(*filters))

        total = self._db.execute(count_query).scalar() or 0

        query = (
            query.order_by(ComplianceRecord.due_date.asc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )

        records = list(self._db.execute(query).scalars().all())
        return records, total

    # ------------------------------------------------------------------
    # Risk scoring
    # ------------------------------------------------------------------

    def calculate_risk_score(self, compliance_id: int) -> float:
        """Calculate and update the risk score for a compliance record."""
        record = self._get_record_or_raise(compliance_id)

        due = record.due_date
        if isinstance(due, datetime):
            due = due.date()

        score = self._calculate_risk_score_from_params(
            due_date=due,
            requirement_type=record.requirement_type,
            escalation_level=record.escalation_level,
            status=record.status,
        )
        record.risk_score = score
        self._db.commit()
        return score

    @staticmethod
    def _calculate_risk_score_from_params(
        due_date: date,
        requirement_type: str,
        escalation_level: int = 0,
        status: str | None = None,
    ) -> float:
        """
        Calculate risk score based on multiple factors.

        Factors:
          - Time until/past due date (proximity factor)
          - Requirement type weight (regulatory significance)
          - Current escalation level
          - Current status

        Returns a score between 0.0 and 100.0.
        """
        if status in ("COMPLETED", "WAIVED", "CANCELLED"):
            return 0.0

        today = date.today()
        days_until_due = (due_date - today).days

        # Time-based factor (0-50 points)
        if days_until_due < -30:
            time_factor = 50.0
        elif days_until_due < 0:
            time_factor = 30.0 + (abs(days_until_due) * 20.0 / 30.0)
        elif days_until_due <= 7:
            time_factor = 25.0
        elif days_until_due <= 14:
            time_factor = 18.0
        elif days_until_due <= 30:
            time_factor = 12.0
        elif days_until_due <= 60:
            time_factor = 6.0
        else:
            time_factor = 2.0

        # Type weight factor (multiplier)
        type_weight = REQUIREMENT_RISK_WEIGHTS.get(requirement_type, 1.0)

        # Escalation factor (0-20 points)
        escalation_factor = min(escalation_level * 5, 20)

        # Status factor
        status_factor = 0.0
        if status == "OVERDUE":
            status_factor = 10.0
        elif status == "ESCALATED":
            status_factor = 15.0

        raw_score = (time_factor + escalation_factor + status_factor) * type_weight
        return min(100.0, round(raw_score, 2))

    # ------------------------------------------------------------------
    # SLA management
    # ------------------------------------------------------------------

    def get_sla_status(self, compliance_id: int) -> str:
        """
        Get the SLA status for a compliance requirement.

        Returns one of: MET, ON_TRACK, AT_RISK, BREACHED, BREACHED_BY_N_DAYS
        """
        record = self._get_record_or_raise(compliance_id)

        due = record.due_date
        if isinstance(due, datetime):
            due = due.date()

        if record.status in ("COMPLETED", "WAIVED"):
            comp = record.completion_date
            if comp is not None:
                if isinstance(comp, datetime):
                    comp = comp.date()
                if comp <= due:
                    return "MET"
                delta = (comp - due).days
                return f"BREACHED_BY_{delta}_DAYS"
            return "MET"

        if record.status == "CANCELLED":
            return "N/A"

        today = date.today()
        days_remaining = (due - today).days

        if days_remaining < 0:
            return f"BREACHED_BY_{abs(days_remaining)}_DAYS"
        if days_remaining <= 7:
            return "AT_RISK"
        return "ON_TRACK"

    # ------------------------------------------------------------------
    # Due date monitoring
    # ------------------------------------------------------------------

    def get_upcoming_deadlines(self, days_ahead: int = 30) -> list[dict[str, object]]:
        """Get compliance records with deadlines in the next N days."""
        today = date.today()
        cutoff = datetime.combine(today, datetime.min.time())
        from datetime import timedelta

        end_date = cutoff + timedelta(days=days_ahead)

        records = self._db.execute(
            select(ComplianceRecord, Drug.drug_name)
            .join(Drug, Drug.drug_id == ComplianceRecord.drug_id)
            .where(
                and_(
                    ComplianceRecord.status.in_(["PENDING", "IN_PROGRESS"]),
                    ComplianceRecord.due_date <= end_date,
                )
            )
            .order_by(ComplianceRecord.due_date.asc())
        ).all()

        results = []
        for record, drug_name in records:
            due = record.due_date
            if isinstance(due, datetime):
                due = due.date()
            days_remaining = (due - today).days

            urgency = "NORMAL"
            if days_remaining <= 0:
                urgency = "OVERDUE"
            elif days_remaining <= 7:
                urgency = "CRITICAL"
            elif days_remaining <= 14:
                urgency = "URGENT"
            elif days_remaining <= 30:
                urgency = "HIGH"

            results.append(
                {
                    "compliance_id": record.id,
                    "drug_name": drug_name,
                    "requirement_type": record.requirement_type,
                    "due_date": due.isoformat(),
                    "days_remaining": days_remaining,
                    "status": record.status,
                    "responsible_party": record.responsible_party,
                    "risk_score": float(record.risk_score) if record.risk_score else 0.0,
                    "urgency": urgency,
                }
            )

        return results

    # ------------------------------------------------------------------
    # Escalation
    # ------------------------------------------------------------------

    def escalate_overdue(self, escalation_user: str) -> int:
        """
        Review all overdue compliance records and escalate as needed.

        Escalation levels are determined by how many days overdue:
          1-14 days:  Level 1
          15-30 days: Level 2
          31-60 days: Level 3
          61-90 days: Level 4
          90+ days:   Level 5

        Returns the number of records escalated.
        """
        today = date.today()
        overdue_records = (
            self._db.execute(
                select(ComplianceRecord).where(
                    and_(
                        ComplianceRecord.status.in_(["PENDING", "IN_PROGRESS", "OVERDUE"]),
                        ComplianceRecord.due_date < datetime.combine(today, datetime.min.time()),
                        ComplianceRecord.completion_date.is_(None),
                    )
                )
            )
            .scalars()
            .all()
        )

        escalated_count = 0
        for record in overdue_records:
            due = record.due_date
            if isinstance(due, datetime):
                due = due.date()
            days_overdue = (today - due).days

            if days_overdue > 90:
                new_level = 5
            elif days_overdue > 60:
                new_level = 4
            elif days_overdue > 30:
                new_level = 3
            elif days_overdue > 14:
                new_level = 2
            else:
                new_level = 1

            if new_level > record.escalation_level:
                old_level = record.escalation_level
                record.escalation_level = new_level
                record.status = "ESCALATED" if new_level >= 3 else "OVERDUE"
                record.risk_score = self._calculate_risk_score_from_params(
                    due_date=due,
                    requirement_type=record.requirement_type,
                    escalation_level=new_level,
                    status=record.status,
                )
                record.modified_by = escalation_user
                record.modified_date = datetime.now(UTC)

                self._record_audit(
                    record.id,
                    "UPDATE",
                    {"escalation_level": old_level, "status": "OVERDUE"},
                    {"escalation_level": new_level, "status": record.status},
                    escalation_user,
                )

                escalated_count += 1

        self._db.commit()
        logger.info(
            "Escalation check: %d records escalated by %s",
            escalated_count,
            escalation_user,
        )
        return escalated_count

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_compliance_summary(self, drug_id: int | None = None) -> ComplianceSummary:
        """Get an aggregated compliance summary."""
        query = select(
            func.count().label("total"),
            func.sum(case((ComplianceRecord.status == "PENDING", 1), else_=0)).label("pending"),
            func.sum(case((ComplianceRecord.status == "IN_PROGRESS", 1), else_=0)).label(
                "in_progress"
            ),
            func.sum(case((ComplianceRecord.status == "OVERDUE", 1), else_=0)).label("overdue"),
            func.sum(case((ComplianceRecord.status == "ESCALATED", 1), else_=0)).label("escalated"),
            func.sum(case((ComplianceRecord.status == "COMPLETED", 1), else_=0)).label("completed"),
            func.avg(ComplianceRecord.risk_score).label("avg_risk_score"),
        ).where(ComplianceRecord.status.notin_(["WAIVED", "CANCELLED"]))

        if drug_id:
            query = query.where(ComplianceRecord.drug_id == drug_id)

        row = self._db.execute(query).one()

        return ComplianceSummary(
            total=row.total or 0,
            pending=row.pending or 0,
            in_progress=row.in_progress or 0,
            overdue=row.overdue or 0,
            escalated=row.escalated or 0,
            completed=row.completed or 0,
            avg_risk_score=round(float(row.avg_risk_score), 2) if row.avg_risk_score else None,
        )

    def get_gap_analysis(self) -> list[dict[str, object]]:
        """
        Identify compliance gaps: overdue/escalated items grouped by drug
        and requirement type with severity assessment.
        """
        records = self._db.execute(
            select(
                ComplianceRecord,
                Drug.drug_name,
            )
            .join(Drug, Drug.drug_id == ComplianceRecord.drug_id)
            .where(ComplianceRecord.status.in_(["OVERDUE", "ESCALATED"]))
            .order_by(ComplianceRecord.escalation_level.desc(), ComplianceRecord.due_date.asc())
        ).all()

        gaps = []
        for record, drug_name in records:
            due = record.due_date
            if isinstance(due, datetime):
                due = due.date()
            days_overdue = (date.today() - due).days

            severity = "LOW"
            if days_overdue > 90:
                severity = "CRITICAL"
            elif days_overdue > 60:
                severity = "HIGH"
            elif days_overdue > 30:
                severity = "MEDIUM"

            gaps.append(
                {
                    "compliance_id": record.id,
                    "drug_id": record.drug_id,
                    "drug_name": drug_name,
                    "requirement_type": record.requirement_type,
                    "due_date": due.isoformat(),
                    "days_overdue": days_overdue,
                    "severity": severity,
                    "escalation_level": record.escalation_level,
                    "responsible_party": record.responsible_party,
                    "risk_score": float(record.risk_score) if record.risk_score else 0.0,
                }
            )

        return gaps

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_record_or_raise(self, compliance_id: int) -> ComplianceRecord:
        """Get compliance record or raise ComplianceNotFoundError."""
        record = self._db.execute(
            select(ComplianceRecord).where(ComplianceRecord.id == compliance_id)
        ).scalar_one_or_none()

        if record is None:
            raise ComplianceNotFoundError(f"Compliance record {compliance_id} not found")
        return record

    def _record_audit(
        self,
        record_id: int,
        action: str,
        old_values: dict[str, object] | None,
        new_values: dict[str, object] | None,
        changed_by: str,
    ) -> None:
        """Record an audit trail entry for compliance changes."""
        audit = AuditTrail(
            table_name="COMPLIANCE_RECORD",
            record_id=record_id,
            action=action,
            old_values=json.dumps(old_values, default=str) if old_values else None,
            new_values=json.dumps(new_values, default=str) if new_values else None,
            changed_by=changed_by,
            application_name="RegRecord-API",
        )
        self._db.add(audit)
