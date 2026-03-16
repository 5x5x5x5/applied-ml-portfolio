# PharmaDataVault

Pharmaceutical Data Warehouse using **Data Vault 2.0** methodology for managing clinical trial data, drug manufacturing records, and regulatory submissions.

## Architecture Overview

```
Source Systems          Staging Layer         Raw Vault              Business Vault       Data Marts
+--------------+       +-----------+        +---------------+      +-------------+      +---------------+
| Drug Feed    |------>| STG_DRUG  |------->| HUB_DRUG      |      | PIT_DRUG    |      | DIM_DRUG      |
| Patient Feed |------>| STG_PAT   |------->| HUB_PATIENT   |----->| PIT_PATIENT |----->| DIM_PATIENT   |
| Trial Feed   |------>| STG_TRIAL |------->| HUB_TRIAL     |      | PIT_TRIAL   |      | DIM_TIME      |
| AE Feed      |------>| STG_AE    |------->| HUB_AE        |      | BRIDGE_*    |      | DIM_FACILITY  |
| Mfg Feed     |------>| STG_MFG   |------->| HUB_FACILITY  |      +-------------+      | FACT_ENROLL   |
+--------------+       +-----------+        | LNK_* (5)     |                           | FACT_AE       |
                                            | SAT_* (5)     |                           | MV_* (3)      |
                                            +---------------+                           +---------------+
```

## Data Vault 2.0 Model

### Hubs (Business Keys)
- **HUB_DRUG** - NDC (National Drug Code)
- **HUB_PATIENT** - MRN (Medical Record Number)
- **HUB_CLINICAL_TRIAL** - NCT ID (ClinicalTrials.gov)
- **HUB_FACILITY** - Internal facility ID
- **HUB_ADVERSE_EVENT** - AE report ID (FDA MedWatch)

### Links (Relationships)
- **LNK_PATIENT_TRIAL** - Patient enrolled in trial
- **LNK_TRIAL_DRUG** - Drug investigated in trial
- **LNK_PATIENT_DRUG** - Drug prescribed to patient
- **LNK_DRUG_ADVERSE_EVENT** - AE linked to drug
- **LNK_TRIAL_FACILITY** - Trial conducted at facility

### Satellites (Descriptive Attributes)
- **SAT_DRUG_DETAILS** - Drug name, manufacturer, form, strength
- **SAT_PATIENT_DEMOGRAPHICS** - Age, sex, ethnicity, weight, BMI
- **SAT_CLINICAL_TRIAL_DETAILS** - Phase, status, sponsor, endpoints
- **SAT_ADVERSE_EVENT_DETAILS** - Severity, outcome, MedDRA coding
- **SAT_DRUG_MANUFACTURING** - Lot, QC status, yield, facility

## Project Structure

```
07-pharma-data-vault/
├── pyproject.toml                          # Project dependencies and config
├── src/pharma_vault/
│   ├── __init__.py                         # Package init
│   ├── _config.py                          # Central configuration
│   ├── etl/
│   │   ├── pipeline.py                     # Python ETL orchestrator
│   │   └── file_processor.py               # File ingestion and validation
│   └── quality/
│       └── data_quality.py                 # Data quality framework
├── sql/
│   ├── ddl/
│   │   ├── raw_vault/
│   │   │   ├── hubs.sql                    # Hub table DDL
│   │   │   ├── links.sql                   # Link table DDL
│   │   │   └── satellites.sql              # Satellite table DDL
│   │   ├── business_vault/
│   │   │   ├── pit_tables.sql              # Point-in-Time tables
│   │   │   └── bridge_tables.sql           # Bridge tables
│   │   └── data_marts/
│   │       └── clinical_mart.sql           # Star schema (facts, dims, MVs)
│   ├── etl/
│   │   ├── staging/
│   │   │   └── stage_drug_data.sql         # PL/SQL staging procedures
│   │   ├── loading/
│   │   │   ├── load_hub_drug.sql           # PL/SQL hub loading package
│   │   │   └── load_satellites.sql         # PL/SQL satellite loading package
│   │   └── mart/
│   │       └── populate_clinical_mart.sql  # PL/SQL mart population package
│   └── queries/
│       └── clinical_analytics.sql          # Analytical queries (PRR, trends)
├── controlm/
│   ├── jobs/
│   │   ├── daily_etl.xml                   # Daily ETL workflow
│   │   └── weekly_mart_refresh.xml         # Weekly mart refresh workflow
│   └── calendars/
│       └── pharma_calendar.xml             # Business calendar definitions
├── scripts/unix/
│   ├── daily_etl.sh                        # UNIX daily ETL driver script
│   └── monitor_etl.sh                      # UNIX ETL monitoring script
└── tests/
    ├── conftest.py                         # Shared test fixtures
    ├── test_etl_pipeline.py                # ETL pipeline tests
    └── test_data_quality.py                # Data quality framework tests
```

## ETL Pipeline Flow

### Daily ETL (Control-M: `PHARMA_DAILY_ETL`)

1. **File Arrival Check** - Verify source files arrived in staging directory
2. **Stage Data** - Load CSV/DAT files, cleanse, validate, deduplicate (parallel)
3. **Load Raw Vault** - Insert new hubs, links, satellites with hashdiff change detection
4. **Data Quality** - Run completeness, RI, business rule, and uniqueness checks
5. **Business Vault** - Refresh PIT tables and bridge tables
6. **Archive** - Move processed files, purge old staging data

### Weekly Mart Refresh (Control-M: `PHARMA_WEEKLY_MART`)

1. **Pre-Validation** - Verify raw vault completeness
2. **Refresh Dimensions** - SCD Type 2 processing for DIM_DRUG, DIM_PATIENT
3. **Load Facts** - FACT_TRIAL_ENROLLMENT, FACT_ADVERSE_EVENTS (parallel)
4. **Refresh MVs** - MV_AE_DRUG_SEVERITY, MV_ENROLLMENT_SUMMARY, MV_MFG_QUALITY
5. **Post-Validation** - Verify mart integrity

## Key Analytical Capabilities

- **Drug Safety Signal Detection** - PRR (Proportional Reporting Ratio) disproportionality analysis
- **Trial Enrollment Trends** - Rolling averages, month-over-month growth, SLA tracking
- **AE Severity Analysis** - Pivot queries by drug class, severity distribution
- **Patient Cohort Comparison** - Multi-dimensional demographic and outcome analysis
- **Manufacturing Quality** - Statistical process control with UCL/LCL monitoring

## Development

```bash
# Install dependencies
uv sync --dev

# Run tests
uv run pytest

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type checking
uv run mypy src/
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Database | Oracle (PL/SQL) / PostgreSQL |
| ETL Orchestration | Python + SQLAlchemy |
| Job Scheduling | Control-M (XML definitions) |
| UNIX Scripts | Bash (file ops, SQL*Plus calls) |
| Data Quality | Custom framework + Great Expectations |
| Modeling | Data Vault 2.0 + Kimball Star Schema |
