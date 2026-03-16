# RegRecord - Regulatory Record Keeping System

A comprehensive system for pharmaceutical companies to track drug regulatory
submissions, approvals, labeling changes, and compliance documents with full
audit trails and pseudo record keeping capabilities.

## Overview

RegRecord demonstrates enterprise database application design including:

- **Pseudo Record Keeping System**: Cryptographic pseudonymization of regulatory
  identifiers with hash-based deterministic IDs, bidirectional mapping, key
  rotation, expiry management, and controlled re-identification with separation
  of duties.
- **Database Design (PL/SQL)**: Oracle-compatible schema with sequences,
  constraints, indexes, range partitioning, triggers for audit trails and state
  machine enforcement, and PL/SQL packages for business logic.
- **ETL Pipeline**: PL/SQL-based ETL with staging tables, agency code
  standardization, MERGE/UPSERT operations, business rule validation, and load
  summary reporting.
- **Control-M Scheduling**: Job definitions for daily compliance checks, monthly
  audit archival, and daily regulatory data ETL with dependency chains, alerting,
  and error handling.
- **UNIX Environment**: Shell scripts for compliance monitoring, audit trail
  archival, and encrypted database backups with remote copy.
- **REST API**: FastAPI application with CRUD endpoints, role-based access
  control, document upload with checksum verification, and audit trail queries.

## Project Structure

```
08-reg-record/
├── pyproject.toml                          # Project dependencies and config
├── src/reg_record/
│   ├── __init__.py
│   ├── api/main.py                         # FastAPI application
│   ├── models/
│   │   ├── database.py                     # SQLAlchemy ORM models
│   │   └── schemas.py                      # Pydantic request/response schemas
│   └── services/
│       ├── pseudo_record_service.py        # Pseudo record keeping system
│       ├── submission_service.py           # Submission lifecycle management
│       └── compliance_service.py           # Compliance tracking and risk scoring
├── sql/
│   ├── ddl/
│   │   ├── schema.sql                      # Database schema (Oracle PL/SQL)
│   │   ├── triggers.sql                    # Audit, validation, versioning triggers
│   │   └── packages.sql                    # PL/SQL packages (4 packages)
│   ├── etl/load_regulatory_data.sql        # ETL staging, transform, merge
│   └── queries/regulatory_reports.sql      # Complex reporting queries
├── controlm/jobs/
│   ├── daily_compliance_check.xml          # Daily compliance job chain
│   ├── monthly_audit_archive.xml           # Monthly archival job chain
│   └── regulatory_data_load.xml            # Daily ETL job chain
├── scripts/unix/
│   ├── compliance_check.sh                 # Compliance monitoring script
│   ├── audit_archive.sh                    # Audit trail archival script
│   └── backup_records.sh                   # Encrypted backup script
└── tests/
    ├── conftest.py                         # Shared fixtures
    ├── test_pseudo_record.py               # Pseudo record service tests
    ├── test_submission.py                  # Submission service tests
    └── test_compliance.py                  # Compliance service tests
```

## Key Features

### Pseudo Record Keeping System

The pseudo record system provides privacy-preserving identifiers for regulatory
records:

- **Deterministic hashing** using SHA-256, SHA-512, HMAC-SHA256, or BLAKE2B
- **Per-record salt** for uniqueness even with the same mapping key
- **Type-prefixed values** (e.g., `PSB-A1B2C3D4...` for submissions)
- **Bidirectional mapping** with controlled re-identification
- **Separation of duties**: requester and authorizer must be different users
- **Expiry and rotation**: mappings expire after configurable periods; keys can
  be rotated without data loss
- **Full audit trail**: every re-identification attempt is logged

### Submission Workflow State Machine

```
DRAFT -> SUBMITTED -> UNDER_REVIEW -> APPROVED
                                   -> REJECTED -> DRAFT (resubmit)
                                   -> ON_HOLD -> UNDER_REVIEW
                                   -> COMPLETE_RESPONSE -> SUBMITTED
Any state -> WITHDRAWN
```

### Compliance Risk Scoring

Risk scores (0-100) are calculated from:
- Time until/past due date (proximity factor)
- Requirement type weight (REMS=2.0x, PMR=1.5x, etc.)
- Escalation level (5 tiers based on days overdue)
- Current status

## Setup

```bash
cd 08-reg-record
uv sync
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Running the API

```bash
uv run uvicorn reg_record.api.main:app --reload --host 0.0.0.0 --port 8000
```

API documentation is available at `http://localhost:8000/docs`.

## Database Setup (Oracle)

```sql
-- Run in order:
@sql/ddl/schema.sql
@sql/ddl/triggers.sql
@sql/ddl/packages.sql
@sql/etl/load_regulatory_data.sql
```

## Technologies

| Component | Technology |
|-----------|-----------|
| Database  | Oracle PL/SQL (SQLite for development) |
| API       | FastAPI + Pydantic + SQLAlchemy |
| ETL       | PL/SQL packages with MERGE/UPSERT |
| Scheduling | Control-M |
| Scripts   | Bash (UNIX) |
| Encryption | AES-256-CBC (OpenSSL) |
| Hashing   | SHA-256, HMAC-SHA256, BLAKE2B |
| Testing   | pytest |
