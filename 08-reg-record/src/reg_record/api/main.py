"""
RegRecord FastAPI Application

Provides REST API endpoints for:
  - Regulatory submission management (CRUD + workflow)
  - Document upload with checksum verification
  - Approval record management
  - Pseudo record management (generate, validate, re-identify)
  - Compliance tracking and reporting
  - Audit trail queries
  - Role-based access control (RBAC)
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import date

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile, status
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from reg_record.models.database import Base
from reg_record.models.schemas import (
    ApprovalCreate,
    ApprovalResponse,
    AuditTrailQuery,
    AuditTrailResponse,
    BatchPseudoRequest,
    BatchPseudoResponse,
    ComplianceCreate,
    ComplianceResponse,
    ComplianceSummary,
    DocumentCreate,
    DocumentResponse,
    PseudoRecordCreate,
    PseudoRecordResponse,
    ReidentificationRequest,
    ReidentificationResponse,
    SubmissionCreate,
    SubmissionListResponse,
    SubmissionResponse,
    SubmissionUpdate,
)
from reg_record.services.compliance_service import (
    ComplianceError,
    ComplianceNotFoundError,
    ComplianceService,
)
from reg_record.services.pseudo_record_service import (
    InvalidMappingKeyError,
    MappingExpiredError,
    PseudoRecordError,
    PseudoRecordService,
    UnauthorizedReidentificationError,
)
from reg_record.services.submission_service import (
    InvalidTransitionError,
    SubmissionError,
    SubmissionNotFoundError,
    SubmissionService,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./regrecord.db",
)

engine = create_engine(
    DATABASE_URL,
    echo=os.getenv("SQL_ECHO", "false").lower() == "true",
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Session:  # type: ignore[misc]
    """Dependency that provides a database session."""
    db = SessionLocal()
    try:
        yield db  # type: ignore[misc]
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Role-based access control
# ---------------------------------------------------------------------------

VALID_ROLES = {"admin", "regulatory_writer", "regulatory_reader", "pseudo_admin", "auditor"}

# Simple role -> permission mapping
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "admin": {"read", "write", "delete", "pseudo_admin", "audit", "reidentify"},
    "regulatory_writer": {"read", "write"},
    "regulatory_reader": {"read"},
    "pseudo_admin": {"read", "pseudo_admin", "reidentify"},
    "auditor": {"read", "audit"},
}


def require_permission(permission: str):  # type: ignore[no-untyped-def]
    """Dependency factory that checks for a specific permission."""

    def _check(
        x_user_role: str = Header(default="regulatory_reader"),
        x_user_name: str = Header(default="anonymous"),
    ) -> str:
        role_perms = ROLE_PERMISSIONS.get(x_user_role, set())
        if permission not in role_perms:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{x_user_role}' lacks permission '{permission}'",
            )
        return x_user_name

    return _check


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Create tables on startup."""
    Base.metadata.create_all(bind=engine)
    logger.info("RegRecord API started; database tables created.")
    yield
    logger.info("RegRecord API shutting down.")


app = FastAPI(
    title="RegRecord - Regulatory Record Keeping System",
    description=(
        "Comprehensive system for pharmaceutical regulatory submissions, "
        "approvals, compliance tracking, and pseudo record keeping."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ===================================================================
# SUBMISSION ENDPOINTS
# ===================================================================


@app.post(
    "/api/v1/submissions",
    response_model=SubmissionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Submissions"],
)
def create_submission(
    data: SubmissionCreate,
    db: Session = Depends(get_db),
    user: str = Depends(require_permission("write")),
) -> SubmissionResponse:
    """Create a new regulatory submission in DRAFT status."""
    try:
        svc = SubmissionService(db)
        submission = svc.create_submission(data)
        return SubmissionResponse.model_validate(submission)
    except SubmissionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get(
    "/api/v1/submissions",
    response_model=SubmissionListResponse,
    tags=["Submissions"],
)
def list_submissions(
    agency: str | None = Query(None),
    sub_status: str | None = Query(None, alias="status"),
    drug_id: int | None = Query(None),
    submission_type: str | None = Query(None),
    assigned_to: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    _user: str = Depends(require_permission("read")),
) -> SubmissionListResponse:
    """List submissions with optional filters and pagination."""
    svc = SubmissionService(db)
    items, total = svc.list_submissions(
        agency=agency,
        status=sub_status,
        drug_id=drug_id,
        submission_type=submission_type,
        assigned_to=assigned_to,
        page=page,
        page_size=page_size,
    )
    return SubmissionListResponse(
        items=[SubmissionResponse.model_validate(s) for s in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@app.get(
    "/api/v1/submissions/{submission_id}",
    response_model=SubmissionResponse,
    tags=["Submissions"],
)
def get_submission(
    submission_id: int,
    db: Session = Depends(get_db),
    _user: str = Depends(require_permission("read")),
) -> SubmissionResponse:
    """Get a submission by ID."""
    try:
        svc = SubmissionService(db)
        submission = svc.get_submission(submission_id)
        return SubmissionResponse.model_validate(submission)
    except SubmissionNotFoundError:
        raise HTTPException(status_code=404, detail="Submission not found")


@app.patch(
    "/api/v1/submissions/{submission_id}",
    response_model=SubmissionResponse,
    tags=["Submissions"],
)
def update_submission(
    submission_id: int,
    data: SubmissionUpdate,
    db: Session = Depends(get_db),
    user: str = Depends(require_permission("write")),
) -> SubmissionResponse:
    """Update a submission (status changes enforce state machine)."""
    try:
        svc = SubmissionService(db)
        submission = svc.update_submission(submission_id, data)
        return SubmissionResponse.model_validate(submission)
    except SubmissionNotFoundError:
        raise HTTPException(status_code=404, detail="Submission not found")
    except InvalidTransitionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except SubmissionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete(
    "/api/v1/submissions/{submission_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Submissions"],
)
def delete_submission(
    submission_id: int,
    db: Session = Depends(get_db),
    user: str = Depends(require_permission("delete")),
) -> None:
    """Delete a DRAFT submission."""
    try:
        svc = SubmissionService(db)
        svc.delete_submission(submission_id, user)
    except SubmissionNotFoundError:
        raise HTTPException(status_code=404, detail="Submission not found")
    except SubmissionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get(
    "/api/v1/submissions/{submission_id}/timeline",
    tags=["Submissions"],
)
def get_submission_timeline(
    submission_id: int,
    db: Session = Depends(get_db),
    _user: str = Depends(require_permission("read")),
) -> list[dict[str, object]]:
    """Get the full audit timeline for a submission."""
    try:
        svc = SubmissionService(db)
        return svc.get_submission_timeline(submission_id)
    except SubmissionNotFoundError:
        raise HTTPException(status_code=404, detail="Submission not found")


@app.get("/api/v1/submissions/pipeline/summary", tags=["Submissions"])
def get_pipeline_summary(
    agency: str | None = Query(None),
    db: Session = Depends(get_db),
    _user: str = Depends(require_permission("read")),
) -> dict[str, int]:
    """Get submission pipeline summary by status."""
    svc = SubmissionService(db)
    return svc.get_pipeline_summary(agency)


# ===================================================================
# DOCUMENT ENDPOINTS
# ===================================================================


@app.post(
    "/api/v1/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
)
def add_document(
    data: DocumentCreate,
    db: Session = Depends(get_db),
    user: str = Depends(require_permission("write")),
) -> DocumentResponse:
    """Add a document to a submission with checksum verification."""
    try:
        svc = SubmissionService(db)
        doc = svc.add_document(
            submission_id=data.submission_id,
            doc_type=data.doc_type.value,
            doc_title=data.doc_title,
            file_path=data.file_path,
            checksum=data.checksum,
            uploaded_by=data.uploaded_by,
            confidentiality=data.confidentiality,
        )
        return DocumentResponse.model_validate(doc)
    except SubmissionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post(
    "/api/v1/documents/upload/{submission_id}",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
)
async def upload_document(
    submission_id: int,
    doc_type: str = Query(...),
    doc_title: str = Query(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: str = Depends(require_permission("write")),
) -> DocumentResponse:
    """Upload a document file with automatic checksum computation."""
    try:
        # Read file content and compute checksum
        content = await file.read()
        checksum = hashlib.sha256(content).hexdigest()

        # Save file (in production, use object storage)
        upload_dir = os.getenv("UPLOAD_DIR", "/tmp/regrecord_uploads")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, f"{submission_id}_{doc_type}_{file.filename}")

        with open(file_path, "wb") as f:
            f.write(content)

        svc = SubmissionService(db)
        doc = svc.add_document(
            submission_id=submission_id,
            doc_type=doc_type,
            doc_title=doc_title,
            file_path=file_path,
            checksum=checksum,
            uploaded_by=user,
            file_size_bytes=len(content),
        )
        return DocumentResponse.model_validate(doc)
    except SubmissionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get(
    "/api/v1/documents/{submission_id}",
    response_model=list[DocumentResponse],
    tags=["Documents"],
)
def list_documents(
    submission_id: int,
    active_only: bool = Query(True),
    db: Session = Depends(get_db),
    _user: str = Depends(require_permission("read")),
) -> list[DocumentResponse]:
    """List all documents for a submission."""
    svc = SubmissionService(db)
    docs = svc.get_documents(submission_id, active_only=active_only)
    return [DocumentResponse.model_validate(d) for d in docs]


@app.get(
    "/api/v1/documents/{document_id}/verify",
    tags=["Documents"],
)
def verify_document_checksum(
    document_id: int,
    checksum: str = Query(...),
    db: Session = Depends(get_db),
    _user: str = Depends(require_permission("read")),
) -> dict[str, object]:
    """Verify a document's checksum."""
    try:
        svc = SubmissionService(db)
        matches = svc.verify_document_checksum(document_id, checksum)
        return {"document_id": document_id, "checksum_valid": matches}
    except SubmissionError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ===================================================================
# APPROVAL ENDPOINTS
# ===================================================================


@app.post(
    "/api/v1/approvals",
    response_model=ApprovalResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Approvals"],
)
def create_approval(
    data: ApprovalCreate,
    db: Session = Depends(get_db),
    user: str = Depends(require_permission("write")),
) -> ApprovalResponse:
    """Record a regulatory approval."""
    from reg_record.models.database import ApprovalRecord, RegulatorySubmission

    sub = (
        db.query(RegulatorySubmission).filter(RegulatorySubmission.id == data.submission_id).first()
    )
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")

    approval = ApprovalRecord(
        submission_id=data.submission_id,
        approval_date=data.approval_date,
        approval_type=data.approval_type,
        conditions=data.conditions,
        expiry_date=data.expiry_date,
        approval_number=data.approval_number,
        approved_indication=data.approved_indication,
        market_exclusivity=data.market_exclusivity,
        created_by=data.created_by,
    )
    db.add(approval)
    db.commit()
    db.refresh(approval)
    return ApprovalResponse.model_validate(approval)


@app.get(
    "/api/v1/approvals/{submission_id}",
    response_model=list[ApprovalResponse],
    tags=["Approvals"],
)
def list_approvals(
    submission_id: int,
    db: Session = Depends(get_db),
    _user: str = Depends(require_permission("read")),
) -> list[ApprovalResponse]:
    """List all approvals for a submission."""
    from reg_record.models.database import ApprovalRecord

    approvals = (
        db.query(ApprovalRecord)
        .filter(ApprovalRecord.submission_id == submission_id)
        .order_by(ApprovalRecord.approval_date.desc())
        .all()
    )
    return [ApprovalResponse.model_validate(a) for a in approvals]


# ===================================================================
# PSEUDO RECORD ENDPOINTS
# ===================================================================


@app.post(
    "/api/v1/pseudo/generate",
    response_model=PseudoRecordResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Pseudo Records"],
)
def generate_pseudo_id(
    data: PseudoRecordCreate,
    db: Session = Depends(get_db),
    user: str = Depends(require_permission("pseudo_admin")),
) -> PseudoRecordResponse:
    """Generate a pseudo ID for a record."""
    try:
        svc = PseudoRecordService(db)
        pseudo_value = svc.generate_pseudo_id(
            original_record_id=data.original_record_id,
            source_table=data.source_table,
            pseudo_type=data.pseudo_type.value,
            mapping_key=data.mapping_key,
            created_by=data.created_by,
            purpose=data.purpose,
            expiry_days=data.expiry_days,
        )
        record = svc.get_pseudo_record(pseudo_value)
        if record is None:
            raise HTTPException(
                status_code=500, detail="Failed to retrieve generated pseudo record"
            )
        return PseudoRecordResponse.model_validate(record)
    except PseudoRecordError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post(
    "/api/v1/pseudo/batch",
    response_model=BatchPseudoResponse,
    tags=["Pseudo Records"],
)
def batch_generate_pseudo_ids(
    data: BatchPseudoRequest,
    record_ids: list[int] = Query(...),
    db: Session = Depends(get_db),
    user: str = Depends(require_permission("pseudo_admin")),
) -> BatchPseudoResponse:
    """Generate pseudo IDs for multiple records."""
    try:
        svc = PseudoRecordService(db)
        results = svc.batch_generate_pseudo_ids(
            record_ids=record_ids,
            source_table=data.source_table,
            pseudo_type=data.pseudo_type.value,
            mapping_key=data.mapping_key,
            created_by=data.created_by,
            purpose=data.purpose,
            expiry_days=data.expiry_days,
        )
        return BatchPseudoResponse(
            generated_count=len(results),
            source_table=data.source_table,
            pseudo_type=data.pseudo_type.value,
        )
    except PseudoRecordError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get(
    "/api/v1/pseudo/validate/{pseudo_value}",
    tags=["Pseudo Records"],
)
def validate_pseudo_mapping(
    pseudo_value: str,
    db: Session = Depends(get_db),
    _user: str = Depends(require_permission("read")),
) -> dict[str, object]:
    """Validate whether a pseudo mapping is active and not expired."""
    svc = PseudoRecordService(db)
    is_valid = svc.validate_pseudo_mapping(pseudo_value)
    return {"pseudo_value": pseudo_value, "is_valid": is_valid}


@app.post(
    "/api/v1/pseudo/reidentify",
    response_model=ReidentificationResponse,
    tags=["Pseudo Records"],
)
def reidentify(
    data: ReidentificationRequest,
    db: Session = Depends(get_db),
    user: str = Depends(require_permission("reidentify")),
) -> ReidentificationResponse:
    """
    Re-identify a pseudo record (map back to real ID).

    Requires separate requester and authorizer for separation of duties.
    All requests are logged in the re-identification audit trail.
    """
    try:
        svc = PseudoRecordService(db)
        original_id, source_table = svc.map_pseudo_to_real(
            pseudo_value=data.pseudo_value,
            requested_by=data.requested_by,
            authorized_by=data.authorized_by,
            reason=data.reason,
        )
        record = svc.get_pseudo_record(data.pseudo_value)
        return ReidentificationResponse(
            original_record_id=original_id,
            source_table=source_table,
            pseudo_type=record.pseudo_type if record else "UNKNOWN",
            authorized_by=data.authorized_by,
            request_logged=True,
        )
    except UnauthorizedReidentificationError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except InvalidMappingKeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except MappingExpiredError as e:
        raise HTTPException(status_code=410, detail=str(e))


@app.get(
    "/api/v1/pseudo/coverage",
    tags=["Pseudo Records"],
)
def get_pseudo_coverage(
    db: Session = Depends(get_db),
    _user: str = Depends(require_permission("pseudo_admin")),
) -> list[dict[str, object]]:
    """Get pseudo record coverage statistics."""
    svc = PseudoRecordService(db)
    return svc.get_coverage_stats()


@app.post(
    "/api/v1/pseudo/purge",
    tags=["Pseudo Records"],
)
def purge_expired_pseudo_records(
    retention_days: int = Query(90, ge=1),
    db: Session = Depends(get_db),
    user: str = Depends(require_permission("pseudo_admin")),
) -> dict[str, int]:
    """Purge expired pseudo record mappings."""
    svc = PseudoRecordService(db)
    count = svc.purge_expired_mappings(retention_days=retention_days)
    return {"purged_count": count}


# ===================================================================
# COMPLIANCE ENDPOINTS
# ===================================================================


@app.post(
    "/api/v1/compliance",
    response_model=ComplianceResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Compliance"],
)
def create_compliance_record(
    data: ComplianceCreate,
    db: Session = Depends(get_db),
    user: str = Depends(require_permission("write")),
) -> ComplianceResponse:
    """Create a new compliance tracking record."""
    try:
        svc = ComplianceService(db)
        record = svc.create_compliance_record(data)
        return ComplianceResponse.model_validate(record)
    except ComplianceError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get(
    "/api/v1/compliance",
    tags=["Compliance"],
)
def list_compliance_records(
    drug_id: int | None = Query(None),
    comp_status: str | None = Query(None, alias="status"),
    responsible_party: str | None = Query(None),
    due_before: date | None = Query(None),
    include_completed: bool = Query(False),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    _user: str = Depends(require_permission("read")),
) -> dict[str, object]:
    """List compliance records with filters."""
    svc = ComplianceService(db)
    records, total = svc.list_compliance_records(
        drug_id=drug_id,
        status=comp_status,
        responsible_party=responsible_party,
        due_before=due_before,
        include_completed=include_completed,
        page=page,
        page_size=page_size,
    )
    return {
        "items": [ComplianceResponse.model_validate(r) for r in records],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.get(
    "/api/v1/compliance/{compliance_id}",
    response_model=ComplianceResponse,
    tags=["Compliance"],
)
def get_compliance_record(
    compliance_id: int,
    db: Session = Depends(get_db),
    _user: str = Depends(require_permission("read")),
) -> ComplianceResponse:
    """Get a compliance record by ID."""
    try:
        svc = ComplianceService(db)
        record = svc.get_compliance_record(compliance_id)
        return ComplianceResponse.model_validate(record)
    except ComplianceNotFoundError:
        raise HTTPException(status_code=404, detail="Compliance record not found")


@app.patch(
    "/api/v1/compliance/{compliance_id}/status",
    response_model=ComplianceResponse,
    tags=["Compliance"],
)
def update_compliance_status(
    compliance_id: int,
    new_status: str = Query(...),
    notes: str | None = Query(None),
    db: Session = Depends(get_db),
    user: str = Depends(require_permission("write")),
) -> ComplianceResponse:
    """Update compliance record status."""
    try:
        svc = ComplianceService(db)
        record = svc.update_status(compliance_id, new_status, user, notes)
        return ComplianceResponse.model_validate(record)
    except ComplianceError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get(
    "/api/v1/compliance/summary",
    response_model=ComplianceSummary,
    tags=["Compliance"],
)
def get_compliance_summary(
    drug_id: int | None = Query(None),
    db: Session = Depends(get_db),
    _user: str = Depends(require_permission("read")),
) -> ComplianceSummary:
    """Get aggregated compliance summary."""
    svc = ComplianceService(db)
    return svc.get_compliance_summary(drug_id)


@app.get("/api/v1/compliance/deadlines", tags=["Compliance"])
def get_upcoming_deadlines(
    days_ahead: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
    _user: str = Depends(require_permission("read")),
) -> list[dict[str, object]]:
    """Get upcoming compliance deadlines."""
    svc = ComplianceService(db)
    return svc.get_upcoming_deadlines(days_ahead)


@app.post("/api/v1/compliance/escalate", tags=["Compliance"])
def escalate_overdue(
    db: Session = Depends(get_db),
    user: str = Depends(require_permission("write")),
) -> dict[str, int]:
    """Escalate all overdue compliance records."""
    svc = ComplianceService(db)
    count = svc.escalate_overdue(user)
    return {"escalated_count": count}


@app.get("/api/v1/compliance/gaps", tags=["Compliance"])
def get_gap_analysis(
    db: Session = Depends(get_db),
    _user: str = Depends(require_permission("read")),
) -> list[dict[str, object]]:
    """Get compliance gap analysis."""
    svc = ComplianceService(db)
    return svc.get_gap_analysis()


# ===================================================================
# AUDIT TRAIL ENDPOINTS
# ===================================================================


@app.post(
    "/api/v1/audit/query",
    response_model=list[AuditTrailResponse],
    tags=["Audit"],
)
def query_audit_trail(
    query: AuditTrailQuery,
    db: Session = Depends(get_db),
    _user: str = Depends(require_permission("audit")),
) -> list[AuditTrailResponse]:
    """Query the audit trail with flexible filters."""
    from sqlalchemy import and_, select

    from reg_record.models.database import AuditTrail

    stmt = select(AuditTrail)
    filters = []

    if query.table_name:
        filters.append(AuditTrail.table_name == query.table_name)
    if query.record_id:
        filters.append(AuditTrail.record_id == query.record_id)
    if query.changed_by:
        filters.append(AuditTrail.changed_by == query.changed_by)
    if query.date_from:
        filters.append(AuditTrail.changed_at >= query.date_from)
    if query.date_to:
        filters.append(AuditTrail.changed_at <= query.date_to)
    if query.action:
        filters.append(AuditTrail.action == query.action)

    if filters:
        stmt = stmt.where(and_(*filters))

    stmt = (
        stmt.order_by(AuditTrail.changed_at.desc())
        .offset((query.page - 1) * query.page_size)
        .limit(query.page_size)
    )

    results = db.execute(stmt).scalars().all()
    return [AuditTrailResponse.model_validate(r) for r in results]


# ===================================================================
# HEALTH CHECK
# ===================================================================


@app.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "RegRecord", "version": "1.0.0"}


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "reg_record.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
