-- ============================================================================
-- PharmaFlow Operational Data Store (ODS) Schema
-- Oracle PL/SQL DDL
--
-- Schema components:
--   1. Staging tables with audit columns
--   2. ODS tables with effective dating
--   3. Error log tables
--   4. Metadata tables (mapping run history, row counts)
-- ============================================================================

-- ============================================================================
-- TABLESPACE (optional, adjust per environment)
-- ============================================================================
-- CREATE TABLESPACE pharma_data
--   DATAFILE '/u01/app/oracle/oradata/PHARMA/pharma_data01.dbf'
--   SIZE 10G AUTOEXTEND ON NEXT 1G MAXSIZE 50G;

-- ============================================================================
-- 1. STAGING TABLES
--    Loaded by ETL sessions; truncated before each load.
--    All have standard audit columns for traceability.
-- ============================================================================

CREATE TABLE stg_drug_master (
    stg_drug_id         NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    drug_name           VARCHAR2(500),
    generic_name        VARCHAR2(500),
    ndc_code            VARCHAR2(20),
    strength            VARCHAR2(100),
    dosage_form         VARCHAR2(100),
    route               VARCHAR2(100),
    manufacturer        VARCHAR2(300),
    dea_schedule        VARCHAR2(10),
    therapeutic_class   VARCHAR2(200),
    supplier_code       VARCHAR2(50),
    -- Audit columns
    source_file_name    VARCHAR2(500),
    source_row_number   NUMBER,
    load_batch_id       NUMBER          NOT NULL,
    load_timestamp      TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL,
    record_status       VARCHAR2(20)    DEFAULT 'NEW' NOT NULL,
    dq_error_code       VARCHAR2(100)
);

CREATE INDEX idx_stg_drug_ndc ON stg_drug_master(ndc_code);
CREATE INDEX idx_stg_drug_batch ON stg_drug_master(load_batch_id);


CREATE TABLE stg_clinical_trial (
    stg_trial_id        NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    nct_id              VARCHAR2(20),
    brief_title         VARCHAR2(1000),
    official_title      VARCHAR2(2000),
    overall_status      VARCHAR2(100),
    phase               VARCHAR2(50),
    study_type          VARCHAR2(100),
    enrollment          NUMBER,
    start_date          DATE,
    completion_date     DATE,
    lead_sponsor        VARCHAR2(500),
    lead_investigator   VARCHAR2(300),
    indication_code     VARCHAR2(50),
    -- Audit
    source_file_name    VARCHAR2(500),
    load_batch_id       NUMBER          NOT NULL,
    load_timestamp      TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL,
    record_status       VARCHAR2(20)    DEFAULT 'NEW' NOT NULL,
    dq_error_code       VARCHAR2(100)
);

CREATE INDEX idx_stg_trial_nct ON stg_clinical_trial(nct_id);
CREATE INDEX idx_stg_trial_batch ON stg_clinical_trial(load_batch_id);


CREATE TABLE stg_faers_demo (
    stg_demo_id         NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    primaryid           VARCHAR2(20),
    caseid              VARCHAR2(20),
    caseversion         VARCHAR2(5),
    age                 NUMBER(10,2),
    age_cod             VARCHAR2(10),
    sex                 VARCHAR2(5),
    wt                  NUMBER(10,2),
    wt_cod              VARCHAR2(10),
    reporter_country    VARCHAR2(5),
    event_dt            VARCHAR2(8),
    init_fda_dt         VARCHAR2(8),
    fda_dt              VARCHAR2(8),
    -- Audit
    source_file_name    VARCHAR2(500),
    load_batch_id       NUMBER          NOT NULL,
    load_timestamp      TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL,
    record_status       VARCHAR2(20)    DEFAULT 'NEW' NOT NULL
);

CREATE INDEX idx_stg_demo_pid ON stg_faers_demo(primaryid);


CREATE TABLE stg_faers_drug (
    stg_drug_id         NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    primaryid           VARCHAR2(20),
    caseid              VARCHAR2(20),
    drug_seq            NUMBER,
    role_cod            VARCHAR2(5),
    drugname            VARCHAR2(500),
    prod_ai             VARCHAR2(500),
    route               VARCHAR2(50),
    dose_vbm            VARCHAR2(200),
    dose_amt            VARCHAR2(50),
    dose_unit           VARCHAR2(20),
    dose_form           VARCHAR2(100),
    -- Audit
    source_file_name    VARCHAR2(500),
    load_batch_id       NUMBER          NOT NULL,
    load_timestamp      TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL,
    record_status       VARCHAR2(20)    DEFAULT 'NEW' NOT NULL
);

CREATE INDEX idx_stg_fdrug_pid ON stg_faers_drug(primaryid);


CREATE TABLE stg_faers_reac (
    stg_reac_id         NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    primaryid           VARCHAR2(20),
    caseid              VARCHAR2(20),
    pt                  VARCHAR2(200),     -- MedDRA Preferred Term
    drug_rec_act        VARCHAR2(500),
    -- Audit
    source_file_name    VARCHAR2(500),
    load_batch_id       NUMBER          NOT NULL,
    load_timestamp      TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL,
    record_status       VARCHAR2(20)    DEFAULT 'NEW' NOT NULL
);

CREATE INDEX idx_stg_reac_pid ON stg_faers_reac(primaryid);


CREATE TABLE stg_faers_outc (
    stg_outc_id         NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    primaryid           VARCHAR2(20),
    caseid              VARCHAR2(20),
    outc_cod            VARCHAR2(10),   -- DE=Death, LT=Life-Threatening, etc.
    -- Audit
    source_file_name    VARCHAR2(500),
    load_batch_id       NUMBER          NOT NULL,
    load_timestamp      TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL,
    record_status       VARCHAR2(20)    DEFAULT 'NEW' NOT NULL
);

CREATE INDEX idx_stg_outc_pid ON stg_faers_outc(primaryid);


-- ============================================================================
-- 2. ODS TABLES (with effective dating for temporal queries)
-- ============================================================================

CREATE TABLE ods_drug_master (
    drug_id             NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ndc_code            VARCHAR2(20)    NOT NULL,
    drug_name           VARCHAR2(500)   NOT NULL,
    generic_name        VARCHAR2(500),
    strength            VARCHAR2(100),
    dosage_form         VARCHAR2(100),
    route               VARCHAR2(100),
    manufacturer        VARCHAR2(300),
    dea_schedule        VARCHAR2(10),
    therapeutic_class   VARCHAR2(200),
    -- Effective dating
    effective_start_dt  DATE            NOT NULL,
    effective_end_dt    DATE            DEFAULT TO_DATE('9999-12-31','YYYY-MM-DD') NOT NULL,
    current_flag        CHAR(1)         DEFAULT 'Y' CHECK (current_flag IN ('Y','N')),
    record_hash         VARCHAR2(64)    NOT NULL,
    -- Audit
    created_by          VARCHAR2(100)   DEFAULT 'PHARMAFLOW',
    created_dt          TIMESTAMP       DEFAULT SYSTIMESTAMP,
    updated_by          VARCHAR2(100),
    updated_dt          TIMESTAMP,
    load_batch_id       NUMBER          NOT NULL,
    source_system       VARCHAR2(50)    DEFAULT 'DRUG_SUPPLIER'
);

CREATE UNIQUE INDEX idx_ods_drug_ndc_eff ON ods_drug_master(ndc_code, effective_start_dt);
CREATE INDEX idx_ods_drug_current ON ods_drug_master(current_flag) WHERE current_flag = 'Y';


CREATE TABLE ods_clinical_trial (
    trial_id            NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    nct_id              VARCHAR2(20)    NOT NULL,
    brief_title         VARCHAR2(1000),
    official_title      VARCHAR2(2000),
    overall_status      VARCHAR2(100),
    phase_std           VARCHAR2(20),
    study_type          VARCHAR2(100),
    enrollment          NUMBER,
    start_date          DATE,
    completion_date     DATE,
    trial_duration_days NUMBER,
    lead_sponsor        VARCHAR2(500),
    indication_code     VARCHAR2(50),
    meddra_pt           VARCHAR2(200),
    -- Effective dating
    effective_start_dt  DATE            NOT NULL,
    effective_end_dt    DATE            DEFAULT TO_DATE('9999-12-31','YYYY-MM-DD'),
    current_flag        CHAR(1)         DEFAULT 'Y',
    -- Audit
    created_dt          TIMESTAMP       DEFAULT SYSTIMESTAMP,
    updated_dt          TIMESTAMP,
    load_batch_id       NUMBER          NOT NULL,
    source_system       VARCHAR2(50)    DEFAULT 'CLINICALTRIALS_GOV'
);

CREATE UNIQUE INDEX idx_ods_trial_nct_eff ON ods_clinical_trial(nct_id, effective_start_dt);


CREATE TABLE ods_adverse_event (
    ae_id               NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    primaryid           VARCHAR2(20)    NOT NULL,
    caseid              VARCHAR2(20),
    age_group           VARCHAR2(20),
    sex                 VARCHAR2(10),
    reporter_country    VARCHAR2(5),
    drugname_std        VARCHAR2(500),
    drug_role           VARCHAR2(10),
    pt_std              VARCHAR2(200),   -- Standardized MedDRA Preferred Term
    meddra_soc          VARCHAR2(200),
    outcome_code        VARCHAR2(10),
    event_date          DATE,
    -- Audit
    created_dt          TIMESTAMP       DEFAULT SYSTIMESTAMP,
    load_batch_id       NUMBER          NOT NULL,
    load_quarter        VARCHAR2(6),     -- e.g., '2025Q4'
    source_system       VARCHAR2(50)    DEFAULT 'FAERS'
);

CREATE INDEX idx_ods_ae_pid ON ods_adverse_event(primaryid);
CREATE INDEX idx_ods_ae_drug ON ods_adverse_event(drugname_std);
CREATE INDEX idx_ods_ae_pt ON ods_adverse_event(pt_std);
CREATE INDEX idx_ods_ae_quarter ON ods_adverse_event(load_quarter);


CREATE TABLE ods_safety_signal (
    signal_id           NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    drugname_std        VARCHAR2(500)   NOT NULL,
    pt_std              VARCHAR2(200)   NOT NULL,
    case_count          NUMBER          NOT NULL,
    prr                 NUMBER(10,4),
    ror                 NUMBER(10,4),
    chi_square          NUMBER(10,4),
    signal_category     VARCHAR2(20),   -- ALERT, REVIEW, ARCHIVE
    detection_date      DATE            DEFAULT SYSDATE,
    -- Audit
    created_dt          TIMESTAMP       DEFAULT SYSTIMESTAMP,
    load_batch_id       NUMBER          NOT NULL,
    load_quarter        VARCHAR2(6)
);

CREATE INDEX idx_ods_signal_drug ON ods_safety_signal(drugname_std);
CREATE INDEX idx_ods_signal_cat ON ods_safety_signal(signal_category);


-- ============================================================================
-- 3. ERROR LOG TABLES
-- ============================================================================

CREATE TABLE etl_error_log (
    error_id            NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    batch_id            NUMBER          NOT NULL,
    mapping_name        VARCHAR2(200)   NOT NULL,
    session_name        VARCHAR2(200),
    transformation_name VARCHAR2(200),
    source_table        VARCHAR2(200),
    target_table        VARCHAR2(200),
    error_code          VARCHAR2(50),
    error_message       VARCHAR2(4000),
    error_severity      VARCHAR2(20)    DEFAULT 'ERROR',  -- ERROR, WARNING, INFO
    source_row_id       NUMBER,
    source_row_data     CLOB,
    error_timestamp     TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL,
    resolved_flag       CHAR(1)         DEFAULT 'N',
    resolved_by         VARCHAR2(100),
    resolved_dt         TIMESTAMP
);

CREATE INDEX idx_err_batch ON etl_error_log(batch_id);
CREATE INDEX idx_err_mapping ON etl_error_log(mapping_name);
CREATE INDEX idx_err_ts ON etl_error_log(error_timestamp);
CREATE INDEX idx_err_unresolved ON etl_error_log(resolved_flag) WHERE resolved_flag = 'N';


CREATE TABLE dq_validation_result (
    validation_id       NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    batch_id            NUMBER          NOT NULL,
    rule_name           VARCHAR2(200)   NOT NULL,
    rule_type           VARCHAR2(50),   -- REFERENTIAL, NULL_CHECK, RANGE, CONSISTENCY
    table_name          VARCHAR2(200),
    column_name         VARCHAR2(200),
    total_rows          NUMBER,
    passed_rows         NUMBER,
    failed_rows         NUMBER,
    pass_rate           NUMBER(5,2),
    quality_score       NUMBER(5,2),
    validation_sql      VARCHAR2(4000),
    execution_timestamp TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL,
    status              VARCHAR2(20)    DEFAULT 'PASS'  -- PASS, FAIL, WARNING
);

CREATE INDEX idx_dq_batch ON dq_validation_result(batch_id);
CREATE INDEX idx_dq_status ON dq_validation_result(status);


-- ============================================================================
-- 4. METADATA TABLES
-- ============================================================================

CREATE TABLE etl_batch_control (
    batch_id            NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    batch_name          VARCHAR2(200)   NOT NULL,
    workflow_name       VARCHAR2(200),
    batch_status        VARCHAR2(20)    DEFAULT 'RUNNING'
                        CHECK (batch_status IN ('RUNNING','SUCCEEDED','FAILED','ABORTED')),
    start_timestamp     TIMESTAMP       NOT NULL,
    end_timestamp       TIMESTAMP,
    elapsed_seconds     NUMBER,
    initiated_by        VARCHAR2(100)   DEFAULT 'SCHEDULER',
    parameters          CLOB,           -- JSON of runtime parameters
    comments            VARCHAR2(2000)
);

CREATE INDEX idx_batch_status ON etl_batch_control(batch_status);
CREATE INDEX idx_batch_start ON etl_batch_control(start_timestamp);


CREATE TABLE etl_mapping_run_log (
    run_id              NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    batch_id            NUMBER          NOT NULL REFERENCES etl_batch_control(batch_id),
    mapping_name        VARCHAR2(200)   NOT NULL,
    session_name        VARCHAR2(200),
    run_status          VARCHAR2(20)    NOT NULL
                        CHECK (run_status IN ('RUNNING','SUCCEEDED','FAILED','ABORTED')),
    start_timestamp     TIMESTAMP       NOT NULL,
    end_timestamp       TIMESTAMP,
    elapsed_seconds     NUMBER(10,2),
    -- Row counts (Informatica-style statistics)
    source_rows_read    NUMBER          DEFAULT 0,
    target_rows_written NUMBER          DEFAULT 0,
    target_rows_updated NUMBER          DEFAULT 0,
    target_rows_deleted NUMBER          DEFAULT 0,
    target_rows_rejected NUMBER         DEFAULT 0,
    error_count         NUMBER          DEFAULT 0,
    throughput_rows_sec NUMBER(10,2),
    -- Source/target metadata
    source_name         VARCHAR2(500),
    target_name         VARCHAR2(500),
    -- Runtime info
    server_name         VARCHAR2(200),
    process_id          NUMBER,
    parameters          CLOB
);

CREATE INDEX idx_run_batch ON etl_mapping_run_log(batch_id);
CREATE INDEX idx_run_mapping ON etl_mapping_run_log(mapping_name);
CREATE INDEX idx_run_status ON etl_mapping_run_log(run_status);
CREATE INDEX idx_run_start ON etl_mapping_run_log(start_timestamp);


CREATE TABLE etl_row_count_audit (
    audit_id            NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    batch_id            NUMBER          NOT NULL REFERENCES etl_batch_control(batch_id),
    table_name          VARCHAR2(200)   NOT NULL,
    count_type          VARCHAR2(20)    NOT NULL
                        CHECK (count_type IN ('PRE_LOAD','POST_LOAD','SOURCE','STAGED','REJECTED')),
    row_count           NUMBER          NOT NULL,
    count_timestamp     TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL
);

CREATE INDEX idx_rowcnt_batch ON etl_row_count_audit(batch_id);
CREATE INDEX idx_rowcnt_table ON etl_row_count_audit(table_name);


-- ============================================================================
-- Dimensional model tables (SCD Type 2 targets)
-- ============================================================================

CREATE TABLE dim_drug (
    drug_key            NUMBER          NOT NULL PRIMARY KEY,
    ndc_code            VARCHAR2(20)    NOT NULL,
    drug_name           VARCHAR2(500)   NOT NULL,
    generic_name        VARCHAR2(500),
    strength            VARCHAR2(100),
    dosage_form         VARCHAR2(100),
    route               VARCHAR2(100),
    manufacturer        VARCHAR2(300),
    dea_schedule        VARCHAR2(10),
    therapeutic_class   VARCHAR2(200),
    -- SCD Type 2 columns
    effective_start_date DATE           NOT NULL,
    effective_end_date  DATE            DEFAULT TO_DATE('9999-12-31','YYYY-MM-DD'),
    current_flag        CHAR(1)         DEFAULT 'Y' CHECK (current_flag IN ('Y','N')),
    record_hash         VARCHAR2(64)    NOT NULL,
    -- Audit
    created_dt          TIMESTAMP       DEFAULT SYSTIMESTAMP,
    updated_dt          TIMESTAMP,
    load_batch_id       NUMBER
);

CREATE INDEX idx_dim_drug_ndc ON dim_drug(ndc_code);
CREATE INDEX idx_dim_drug_current ON dim_drug(current_flag);


CREATE TABLE dim_trial (
    trial_key           NUMBER          NOT NULL PRIMARY KEY,
    nct_id              VARCHAR2(20)    NOT NULL,
    brief_title         VARCHAR2(1000),
    phase_std           VARCHAR2(20),
    study_type          VARCHAR2(100),
    lead_sponsor        VARCHAR2(500),
    indication_code     VARCHAR2(50),
    meddra_pt           VARCHAR2(200),
    -- SCD Type 2
    effective_start_date DATE           NOT NULL,
    effective_end_date  DATE            DEFAULT TO_DATE('9999-12-31','YYYY-MM-DD'),
    current_flag        CHAR(1)         DEFAULT 'Y',
    record_hash         VARCHAR2(64),
    -- Audit
    created_dt          TIMESTAMP       DEFAULT SYSTIMESTAMP,
    updated_dt          TIMESTAMP,
    load_batch_id       NUMBER
);


CREATE TABLE fact_clinical_trial (
    trial_fact_key      NUMBER          NOT NULL PRIMARY KEY,
    trial_key           NUMBER          REFERENCES dim_trial(trial_key),
    drug_key            NUMBER          REFERENCES dim_drug(drug_key),
    enrollment_count    NUMBER,
    trial_duration_days NUMBER,
    start_date          DATE,
    completion_date     DATE,
    overall_status      VARCHAR2(100),
    -- Audit
    load_date           DATE            DEFAULT SYSDATE,
    load_batch_id       NUMBER
);


CREATE TABLE fact_adverse_event (
    ae_fact_key         NUMBER          NOT NULL PRIMARY KEY,
    drug_key            NUMBER          REFERENCES dim_drug(drug_key),
    primaryid           VARCHAR2(20),
    age_group           VARCHAR2(20),
    sex                 VARCHAR2(10),
    reporter_country    VARCHAR2(5),
    drugname_std        VARCHAR2(500),
    pt_std              VARCHAR2(200),
    meddra_soc          VARCHAR2(200),
    outcome_code        VARCHAR2(10),
    event_date          DATE,
    prr                 NUMBER(10,4),
    -- Audit
    load_date           DATE            DEFAULT SYSDATE,
    load_batch_id       NUMBER,
    load_quarter        VARCHAR2(6)
);

CREATE INDEX idx_fact_ae_drug ON fact_adverse_event(drug_key);
CREATE INDEX idx_fact_ae_pt ON fact_adverse_event(pt_std);


-- ============================================================================
-- Grants (adjust role names per environment)
-- ============================================================================

-- GRANT SELECT, INSERT, UPDATE, DELETE ON stg_drug_master TO etl_role;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ods_drug_master TO etl_role;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON dim_drug TO etl_role;
-- GRANT SELECT ON dim_drug TO reporting_role;
-- GRANT SELECT ON fact_clinical_trial TO reporting_role;
-- GRANT SELECT ON fact_adverse_event TO reporting_role;
