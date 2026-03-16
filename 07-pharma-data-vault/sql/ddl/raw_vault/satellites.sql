/*******************************************************************************
 * PharmaDataVault - Raw Vault Satellite Tables DDL
 *
 * Data Vault 2.0 Satellite tables storing descriptive/contextual attributes.
 * Satellites capture the full history of changes via insert-only pattern.
 *
 * Conventions:
 *   - Primary key = (parent_hash_key, LOAD_DATE) for full history
 *   - HASHDIFF = MD5 of all descriptive columns for change detection
 *   - LOAD_DATE = timestamp when this version was loaded
 *   - LOAD_END_DATE = end of effectivity (9999-12-31 for current)
 *   - Only new/changed records are inserted (hashdiff comparison)
 *
 * Oracle PL/SQL compatible DDL
 ******************************************************************************/

-- ============================================================================
-- SAT_DRUG_DETAILS
-- Descriptive attributes for drugs. Tracks changes to drug information
-- such as name changes, manufacturer updates, reformulations.
-- ============================================================================

CREATE TABLE SAT_DRUG_DETAILS (
    DRUG_KEY            RAW(16)         NOT NULL,   -- FK to HUB_DRUG
    LOAD_DATE           TIMESTAMP       NOT NULL,   -- When this version was loaded
    LOAD_END_DATE       TIMESTAMP       DEFAULT TO_TIMESTAMP('9999-12-31 23:59:59', 'YYYY-MM-DD HH24:MI:SS') NOT NULL,
    HASHDIFF            RAW(16)         NOT NULL,   -- MD5 of descriptive columns
    RECORD_SOURCE       VARCHAR2(100)   NOT NULL,
    -- Descriptive attributes
    DRUG_NAME           VARCHAR2(200)   NOT NULL,
    GENERIC_NAME        VARCHAR2(200)   NULL,
    MANUFACTURER        VARCHAR2(200)   NOT NULL,
    DRUG_FORM           VARCHAR2(50)    NOT NULL,   -- tablet, capsule, injection, etc.
    STRENGTH            VARCHAR2(50)    NOT NULL,   -- e.g., '500mg', '10mg/mL'
    STRENGTH_UNIT       VARCHAR2(20)    NULL,
    ROUTE_OF_ADMIN      VARCHAR2(50)    NULL,       -- oral, IV, subcutaneous, etc.
    DEA_SCHEDULE        VARCHAR2(5)     NULL,       -- I, II, III, IV, V, or NULL
    APPROVAL_DATE       DATE            NULL,       -- FDA approval date
    THERAPEUTIC_CLASS   VARCHAR2(100)   NULL,       -- ATC classification
    NDA_NUMBER          VARCHAR2(20)    NULL,       -- New Drug Application number
    CONSTRAINT PK_SAT_DRUG_DETAILS PRIMARY KEY (DRUG_KEY, LOAD_DATE),
    CONSTRAINT FK_SAT_DRUG_HUB FOREIGN KEY (DRUG_KEY)
        REFERENCES HUB_DRUG (DRUG_KEY)
)
TABLESPACE TBS_RAW_VAULT
PCTFREE 10
COMPRESS;

COMMENT ON TABLE SAT_DRUG_DETAILS IS 'Satellite - Drug descriptive attributes with full change history';
COMMENT ON COLUMN SAT_DRUG_DETAILS.HASHDIFF IS 'MD5 hash of all descriptive columns for change detection';
COMMENT ON COLUMN SAT_DRUG_DETAILS.LOAD_END_DATE IS 'End of effectivity; 9999-12-31 indicates current record';

CREATE INDEX IDX_SAT_DRUG_END ON SAT_DRUG_DETAILS (DRUG_KEY, LOAD_END_DATE)
    TABLESPACE TBS_RAW_VAULT_IDX;

CREATE INDEX IDX_SAT_DRUG_HASHDIFF ON SAT_DRUG_DETAILS (DRUG_KEY, HASHDIFF)
    TABLESPACE TBS_RAW_VAULT_IDX;

-- ============================================================================
-- SAT_PATIENT_DEMOGRAPHICS
-- Descriptive attributes for patients. Captures demographic changes
-- over time (e.g., weight changes during trial participation).
-- ============================================================================

CREATE TABLE SAT_PATIENT_DEMOGRAPHICS (
    PATIENT_KEY         RAW(16)         NOT NULL,   -- FK to HUB_PATIENT
    LOAD_DATE           TIMESTAMP       NOT NULL,
    LOAD_END_DATE       TIMESTAMP       DEFAULT TO_TIMESTAMP('9999-12-31 23:59:59', 'YYYY-MM-DD HH24:MI:SS') NOT NULL,
    HASHDIFF            RAW(16)         NOT NULL,
    RECORD_SOURCE       VARCHAR2(100)   NOT NULL,
    -- Descriptive attributes
    DATE_OF_BIRTH       DATE            NULL,
    AGE_AT_LOAD         NUMBER(3)       NULL,       -- Calculated age at load time
    SEX                 VARCHAR2(1)     NOT NULL,   -- M, F, U
    ETHNICITY           VARCHAR2(50)    NULL,       -- FDA standard ethnicity categories
    RACE                VARCHAR2(50)    NULL,       -- FDA standard race categories
    WEIGHT_KG           NUMBER(5,1)     NULL,
    HEIGHT_CM           NUMBER(5,1)     NULL,
    BMI                 NUMBER(4,1)     NULL,       -- Calculated BMI
    COUNTRY             VARCHAR2(3)     NULL,       -- ISO 3166-1 alpha-3
    STATE_PROVINCE      VARCHAR2(50)    NULL,
    SMOKING_STATUS      VARCHAR2(20)    NULL,       -- current, former, never
    CONSTRAINT PK_SAT_PAT_DEMO PRIMARY KEY (PATIENT_KEY, LOAD_DATE),
    CONSTRAINT FK_SAT_PAT_HUB FOREIGN KEY (PATIENT_KEY)
        REFERENCES HUB_PATIENT (PATIENT_KEY),
    CONSTRAINT CK_SAT_PAT_SEX CHECK (SEX IN ('M', 'F', 'U'))
)
TABLESPACE TBS_RAW_VAULT
PCTFREE 10
COMPRESS;

COMMENT ON TABLE SAT_PATIENT_DEMOGRAPHICS IS 'Satellite - Patient demographic attributes with full change history';

CREATE INDEX IDX_SAT_PAT_END ON SAT_PATIENT_DEMOGRAPHICS (PATIENT_KEY, LOAD_END_DATE)
    TABLESPACE TBS_RAW_VAULT_IDX;

CREATE INDEX IDX_SAT_PAT_HASHDIFF ON SAT_PATIENT_DEMOGRAPHICS (PATIENT_KEY, HASHDIFF)
    TABLESPACE TBS_RAW_VAULT_IDX;

-- ============================================================================
-- SAT_CLINICAL_TRIAL_DETAILS
-- Descriptive attributes for clinical trials. Captures trial status
-- changes, protocol amendments, and milestone updates.
-- ============================================================================

CREATE TABLE SAT_CLINICAL_TRIAL_DETAILS (
    TRIAL_KEY           RAW(16)         NOT NULL,   -- FK to HUB_CLINICAL_TRIAL
    LOAD_DATE           TIMESTAMP       NOT NULL,
    LOAD_END_DATE       TIMESTAMP       DEFAULT TO_TIMESTAMP('9999-12-31 23:59:59', 'YYYY-MM-DD HH24:MI:SS') NOT NULL,
    HASHDIFF            RAW(16)         NOT NULL,
    RECORD_SOURCE       VARCHAR2(100)   NOT NULL,
    -- Descriptive attributes
    TRIAL_TITLE         VARCHAR2(500)   NOT NULL,
    TRIAL_PHASE         VARCHAR2(20)    NOT NULL,   -- Phase I, II, III, IV
    TRIAL_STATUS        VARCHAR2(30)    NOT NULL,   -- recruiting, active, completed, etc.
    START_DATE          DATE            NULL,
    ESTIMATED_END_DATE  DATE            NULL,
    ACTUAL_END_DATE     DATE            NULL,
    SPONSOR             VARCHAR2(200)   NOT NULL,
    LEAD_INVESTIGATOR   VARCHAR2(200)   NULL,
    PROTOCOL_NUMBER     VARCHAR2(50)    NULL,
    PROTOCOL_VERSION    VARCHAR2(10)    NULL,
    TARGET_ENROLLMENT   NUMBER(6)       NULL,
    ACTUAL_ENROLLMENT   NUMBER(6)       NULL,
    THERAPEUTIC_AREA    VARCHAR2(100)   NULL,
    PRIMARY_ENDPOINT    VARCHAR2(500)   NULL,
    IND_NUMBER          VARCHAR2(20)    NULL,       -- Investigational New Drug number
    CONSTRAINT PK_SAT_TRIAL_DETAILS PRIMARY KEY (TRIAL_KEY, LOAD_DATE),
    CONSTRAINT FK_SAT_TRIAL_HUB FOREIGN KEY (TRIAL_KEY)
        REFERENCES HUB_CLINICAL_TRIAL (TRIAL_KEY),
    CONSTRAINT CK_TRIAL_PHASE CHECK (
        TRIAL_PHASE IN ('Phase I', 'Phase I/II', 'Phase II', 'Phase II/III',
                         'Phase III', 'Phase IV', 'Pre-clinical')
    ),
    CONSTRAINT CK_TRIAL_STATUS CHECK (
        TRIAL_STATUS IN ('not_yet_recruiting', 'recruiting', 'enrolling_by_invitation',
                         'active_not_recruiting', 'suspended', 'terminated',
                         'completed', 'withdrawn')
    )
)
TABLESPACE TBS_RAW_VAULT
PCTFREE 10
COMPRESS;

COMMENT ON TABLE SAT_CLINICAL_TRIAL_DETAILS IS 'Satellite - Clinical trial details with full change history';

CREATE INDEX IDX_SAT_TRIAL_END ON SAT_CLINICAL_TRIAL_DETAILS (TRIAL_KEY, LOAD_END_DATE)
    TABLESPACE TBS_RAW_VAULT_IDX;

CREATE INDEX IDX_SAT_TRIAL_STATUS ON SAT_CLINICAL_TRIAL_DETAILS (TRIAL_STATUS)
    TABLESPACE TBS_RAW_VAULT_IDX;

CREATE INDEX IDX_SAT_TRIAL_PHASE ON SAT_CLINICAL_TRIAL_DETAILS (TRIAL_PHASE)
    TABLESPACE TBS_RAW_VAULT_IDX;

-- ============================================================================
-- SAT_ADVERSE_EVENT_DETAILS
-- Descriptive attributes for adverse event reports. Captures severity
-- assessments, outcomes, and narrative descriptions.
-- ============================================================================

CREATE TABLE SAT_ADVERSE_EVENT_DETAILS (
    AE_KEY              RAW(16)         NOT NULL,   -- FK to HUB_ADVERSE_EVENT
    LOAD_DATE           TIMESTAMP       NOT NULL,
    LOAD_END_DATE       TIMESTAMP       DEFAULT TO_TIMESTAMP('9999-12-31 23:59:59', 'YYYY-MM-DD HH24:MI:SS') NOT NULL,
    HASHDIFF            RAW(16)         NOT NULL,
    RECORD_SOURCE       VARCHAR2(100)   NOT NULL,
    -- Descriptive attributes
    AE_TERM             VARCHAR2(250)   NOT NULL,   -- MedDRA preferred term
    AE_DESCRIPTION      CLOB            NULL,       -- Full narrative
    SEVERITY            VARCHAR2(20)    NOT NULL,   -- mild, moderate, severe, life_threatening, fatal
    SERIOUSNESS         VARCHAR2(1)     NOT NULL,   -- Y/N per FDA definition
    CAUSALITY           VARCHAR2(30)    NULL,       -- definite, probable, possible, unlikely, unrelated
    OUTCOME             VARCHAR2(50)    NOT NULL,   -- recovered, recovering, not_recovered, fatal, unknown
    ONSET_DATE          DATE            NOT NULL,
    RESOLUTION_DATE     DATE            NULL,
    REPORT_DATE         DATE            NOT NULL,
    REPORTER_TYPE       VARCHAR2(30)    NULL,       -- physician, pharmacist, consumer, other
    MEDDRA_PT_CODE      VARCHAR2(20)    NULL,       -- MedDRA Preferred Term code
    MEDDRA_SOC          VARCHAR2(100)   NULL,       -- MedDRA System Organ Class
    EXPECTEDNESS        VARCHAR2(20)    NULL,       -- expected, unexpected
    ACTION_TAKEN        VARCHAR2(50)    NULL,       -- dose_reduced, withdrawn, none, etc.
    CONSTRAINT PK_SAT_AE_DETAILS PRIMARY KEY (AE_KEY, LOAD_DATE),
    CONSTRAINT FK_SAT_AE_HUB FOREIGN KEY (AE_KEY)
        REFERENCES HUB_ADVERSE_EVENT (AE_KEY),
    CONSTRAINT CK_AE_SEVERITY CHECK (
        SEVERITY IN ('mild', 'moderate', 'severe', 'life_threatening', 'fatal')
    ),
    CONSTRAINT CK_AE_SERIOUS CHECK (SERIOUSNESS IN ('Y', 'N')),
    CONSTRAINT CK_AE_OUTCOME CHECK (
        OUTCOME IN ('recovered', 'recovering', 'not_recovered', 'recovered_with_sequelae',
                     'fatal', 'unknown')
    )
)
TABLESPACE TBS_RAW_VAULT
PCTFREE 10;

COMMENT ON TABLE SAT_ADVERSE_EVENT_DETAILS IS 'Satellite - Adverse event details with full change history';

CREATE INDEX IDX_SAT_AE_END ON SAT_ADVERSE_EVENT_DETAILS (AE_KEY, LOAD_END_DATE)
    TABLESPACE TBS_RAW_VAULT_IDX;

CREATE INDEX IDX_SAT_AE_SEVERITY ON SAT_ADVERSE_EVENT_DETAILS (SEVERITY)
    TABLESPACE TBS_RAW_VAULT_IDX;

CREATE INDEX IDX_SAT_AE_ONSET ON SAT_ADVERSE_EVENT_DETAILS (ONSET_DATE)
    TABLESPACE TBS_RAW_VAULT_IDX;

CREATE INDEX IDX_SAT_AE_MEDDRA ON SAT_ADVERSE_EVENT_DETAILS (MEDDRA_PT_CODE)
    TABLESPACE TBS_RAW_VAULT_IDX;

-- ============================================================================
-- SAT_DRUG_MANUFACTURING
-- Descriptive attributes for drug manufacturing lots. Tracks lot
-- information, manufacturing dates, and quality control data.
-- ============================================================================

CREATE TABLE SAT_DRUG_MANUFACTURING (
    DRUG_KEY            RAW(16)         NOT NULL,   -- FK to HUB_DRUG
    LOAD_DATE           TIMESTAMP       NOT NULL,
    LOAD_END_DATE       TIMESTAMP       DEFAULT TO_TIMESTAMP('9999-12-31 23:59:59', 'YYYY-MM-DD HH24:MI:SS') NOT NULL,
    HASHDIFF            RAW(16)         NOT NULL,
    RECORD_SOURCE       VARCHAR2(100)   NOT NULL,
    -- Descriptive attributes
    LOT_NUMBER          VARCHAR2(30)    NOT NULL,
    BATCH_SIZE          NUMBER(12,2)    NULL,       -- Quantity manufactured
    BATCH_SIZE_UNIT     VARCHAR2(20)    NULL,       -- kg, L, units
    MFG_DATE            DATE            NOT NULL,
    EXPIRY_DATE         DATE            NOT NULL,
    FACILITY_ID         VARCHAR2(20)    NOT NULL,   -- Manufacturing facility
    MFG_LINE            VARCHAR2(30)    NULL,       -- Production line
    QC_STATUS           VARCHAR2(20)    NOT NULL,   -- passed, failed, pending, quarantined
    QC_DATE             DATE            NULL,
    QC_ANALYST          VARCHAR2(100)   NULL,
    YIELD_PERCENT       NUMBER(5,2)     NULL,
    DEVIATION_FLAG      VARCHAR2(1)     DEFAULT 'N' NOT NULL,
    DEVIATION_ID        VARCHAR2(30)    NULL,
    RELEASE_DATE        DATE            NULL,       -- Date lot was released for distribution
    STORAGE_CONDITIONS  VARCHAR2(100)   NULL,       -- e.g., '2-8C refrigerated'
    CONSTRAINT PK_SAT_DRUG_MFG PRIMARY KEY (DRUG_KEY, LOAD_DATE),
    CONSTRAINT FK_SAT_MFG_HUB FOREIGN KEY (DRUG_KEY)
        REFERENCES HUB_DRUG (DRUG_KEY),
    CONSTRAINT CK_MFG_QC CHECK (
        QC_STATUS IN ('passed', 'failed', 'pending', 'quarantined')
    ),
    CONSTRAINT CK_MFG_DEV CHECK (DEVIATION_FLAG IN ('Y', 'N'))
)
TABLESPACE TBS_RAW_VAULT
PCTFREE 10;

COMMENT ON TABLE SAT_DRUG_MANUFACTURING IS 'Satellite - Drug manufacturing lot details with full change history';

CREATE INDEX IDX_SAT_MFG_END ON SAT_DRUG_MANUFACTURING (DRUG_KEY, LOAD_END_DATE)
    TABLESPACE TBS_RAW_VAULT_IDX;

CREATE INDEX IDX_SAT_MFG_LOT ON SAT_DRUG_MANUFACTURING (LOT_NUMBER)
    TABLESPACE TBS_RAW_VAULT_IDX;

CREATE INDEX IDX_SAT_MFG_EXPIRY ON SAT_DRUG_MANUFACTURING (EXPIRY_DATE)
    TABLESPACE TBS_RAW_VAULT_IDX;

CREATE INDEX IDX_SAT_MFG_QC ON SAT_DRUG_MANUFACTURING (QC_STATUS)
    TABLESPACE TBS_RAW_VAULT_IDX;

CREATE INDEX IDX_SAT_MFG_FACILITY ON SAT_DRUG_MANUFACTURING (FACILITY_ID)
    TABLESPACE TBS_RAW_VAULT_IDX;

-- ============================================================================
-- ETL_ERROR_LOG - Supporting table for satellite load error tracking
-- ============================================================================

CREATE TABLE ETL_ERROR_LOG (
    ERROR_ID            NUMBER          GENERATED ALWAYS AS IDENTITY NOT NULL,
    PROCEDURE_NAME      VARCHAR2(100)   NOT NULL,
    ERROR_CODE          NUMBER          NOT NULL,
    ERROR_MESSAGE       VARCHAR2(4000)  NOT NULL,
    ERROR_CONTEXT       VARCHAR2(4000)  NULL,
    SOURCE_TABLE        VARCHAR2(100)   NULL,
    TARGET_TABLE        VARCHAR2(100)   NULL,
    RECORD_KEY          VARCHAR2(200)   NULL,
    BATCH_ID            NUMBER          NULL,
    ERROR_TIMESTAMP     TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL,
    CONSTRAINT PK_ETL_ERROR_LOG PRIMARY KEY (ERROR_ID)
)
TABLESPACE TBS_STAGING;

CREATE INDEX IDX_ETL_ERR_TS ON ETL_ERROR_LOG (ERROR_TIMESTAMP)
    TABLESPACE TBS_STAGING;

CREATE INDEX IDX_ETL_ERR_PROC ON ETL_ERROR_LOG (PROCEDURE_NAME)
    TABLESPACE TBS_STAGING;

-- ============================================================================
-- Grants
-- ============================================================================

GRANT SELECT, INSERT ON SAT_DRUG_DETAILS TO ROLE_ETL;
GRANT SELECT, INSERT ON SAT_PATIENT_DEMOGRAPHICS TO ROLE_ETL;
GRANT SELECT, INSERT ON SAT_CLINICAL_TRIAL_DETAILS TO ROLE_ETL;
GRANT SELECT, INSERT ON SAT_ADVERSE_EVENT_DETAILS TO ROLE_ETL;
GRANT SELECT, INSERT ON SAT_DRUG_MANUFACTURING TO ROLE_ETL;
GRANT SELECT, INSERT ON ETL_ERROR_LOG TO ROLE_ETL;
GRANT SELECT, INSERT, UPDATE ON SAT_DRUG_DETAILS TO ROLE_ETL;
GRANT SELECT, INSERT, UPDATE ON SAT_PATIENT_DEMOGRAPHICS TO ROLE_ETL;
GRANT SELECT, INSERT, UPDATE ON SAT_CLINICAL_TRIAL_DETAILS TO ROLE_ETL;
GRANT SELECT, INSERT, UPDATE ON SAT_ADVERSE_EVENT_DETAILS TO ROLE_ETL;
GRANT SELECT, INSERT, UPDATE ON SAT_DRUG_MANUFACTURING TO ROLE_ETL;
