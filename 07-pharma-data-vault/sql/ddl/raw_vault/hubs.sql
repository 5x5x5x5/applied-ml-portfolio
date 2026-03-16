/*******************************************************************************
 * PharmaDataVault - Raw Vault Hub Tables DDL
 *
 * Data Vault 2.0 Hub tables for pharmaceutical data warehouse.
 * Hubs store unique business keys and represent core business concepts.
 *
 * Conventions:
 *   - Hash keys (MD5) used as surrogate keys for distribution and joining
 *   - Business keys stored as natural keys from source systems
 *   - LOAD_DATE: timestamp when record first entered the vault
 *   - RECORD_SOURCE: identifier for the originating source system
 *
 * Oracle PL/SQL compatible DDL
 ******************************************************************************/

-- ============================================================================
-- SEQUENCES
-- ============================================================================

CREATE SEQUENCE SEQ_HUB_DRUG START WITH 1 INCREMENT BY 1 NOCACHE;
CREATE SEQUENCE SEQ_HUB_PATIENT START WITH 1 INCREMENT BY 1 NOCACHE;
CREATE SEQUENCE SEQ_HUB_CLINICAL_TRIAL START WITH 1 INCREMENT BY 1 NOCACHE;
CREATE SEQUENCE SEQ_HUB_FACILITY START WITH 1 INCREMENT BY 1 NOCACHE;
CREATE SEQUENCE SEQ_HUB_ADVERSE_EVENT START WITH 1 INCREMENT BY 1 NOCACHE;

-- ============================================================================
-- HUB_DRUG
-- Central hub for all drug/product records across source systems.
-- Business Key: NDC (National Drug Code) - the FDA standard drug identifier.
-- ============================================================================

CREATE TABLE HUB_DRUG (
    DRUG_KEY            RAW(16)         NOT NULL,   -- MD5 hash of DRUG_NDC
    DRUG_NDC            VARCHAR2(13)    NOT NULL,   -- NDC in 5-4-2 format
    LOAD_DATE           TIMESTAMP       NOT NULL,   -- First seen in vault
    RECORD_SOURCE       VARCHAR2(100)   NOT NULL,   -- Source system identifier
    CONSTRAINT PK_HUB_DRUG PRIMARY KEY (DRUG_KEY),
    CONSTRAINT UK_HUB_DRUG_BK UNIQUE (DRUG_NDC)
)
TABLESPACE TBS_RAW_VAULT
PCTFREE 5
NOLOGGING;

COMMENT ON TABLE HUB_DRUG IS 'Data Vault 2.0 Hub - Unique drugs identified by National Drug Code (NDC)';
COMMENT ON COLUMN HUB_DRUG.DRUG_KEY IS 'MD5 hash surrogate key derived from DRUG_NDC';
COMMENT ON COLUMN HUB_DRUG.DRUG_NDC IS 'FDA National Drug Code in 5-4-2 format (business key)';
COMMENT ON COLUMN HUB_DRUG.LOAD_DATE IS 'UTC timestamp when this business key first entered the vault';
COMMENT ON COLUMN HUB_DRUG.RECORD_SOURCE IS 'Source system that first provided this business key';

CREATE INDEX IDX_HUB_DRUG_NDC ON HUB_DRUG (DRUG_NDC)
    TABLESPACE TBS_RAW_VAULT_IDX;

CREATE INDEX IDX_HUB_DRUG_LOAD_DATE ON HUB_DRUG (LOAD_DATE)
    TABLESPACE TBS_RAW_VAULT_IDX;

-- ============================================================================
-- HUB_PATIENT
-- Central hub for all patient records.
-- Business Key: MRN (Medical Record Number) - unique patient identifier.
-- ============================================================================

CREATE TABLE HUB_PATIENT (
    PATIENT_KEY         RAW(16)         NOT NULL,   -- MD5 hash of PATIENT_MRN
    PATIENT_MRN         VARCHAR2(20)    NOT NULL,   -- Medical Record Number
    LOAD_DATE           TIMESTAMP       NOT NULL,
    RECORD_SOURCE       VARCHAR2(100)   NOT NULL,
    CONSTRAINT PK_HUB_PATIENT PRIMARY KEY (PATIENT_KEY),
    CONSTRAINT UK_HUB_PATIENT_BK UNIQUE (PATIENT_MRN)
)
TABLESPACE TBS_RAW_VAULT
PCTFREE 5
NOLOGGING;

COMMENT ON TABLE HUB_PATIENT IS 'Data Vault 2.0 Hub - Unique patients identified by Medical Record Number';
COMMENT ON COLUMN HUB_PATIENT.PATIENT_KEY IS 'MD5 hash surrogate key derived from PATIENT_MRN';
COMMENT ON COLUMN HUB_PATIENT.PATIENT_MRN IS 'Medical Record Number (business key)';

CREATE INDEX IDX_HUB_PATIENT_MRN ON HUB_PATIENT (PATIENT_MRN)
    TABLESPACE TBS_RAW_VAULT_IDX;

CREATE INDEX IDX_HUB_PATIENT_LOAD_DATE ON HUB_PATIENT (LOAD_DATE)
    TABLESPACE TBS_RAW_VAULT_IDX;

-- ============================================================================
-- HUB_CLINICAL_TRIAL
-- Central hub for clinical trials.
-- Business Key: NCT ID from ClinicalTrials.gov (e.g., NCT00000001).
-- ============================================================================

CREATE TABLE HUB_CLINICAL_TRIAL (
    TRIAL_KEY           RAW(16)         NOT NULL,   -- MD5 hash of TRIAL_NCT_ID
    TRIAL_NCT_ID        VARCHAR2(15)    NOT NULL,   -- ClinicalTrials.gov NCT ID
    LOAD_DATE           TIMESTAMP       NOT NULL,
    RECORD_SOURCE       VARCHAR2(100)   NOT NULL,
    CONSTRAINT PK_HUB_CLINICAL_TRIAL PRIMARY KEY (TRIAL_KEY),
    CONSTRAINT UK_HUB_TRIAL_BK UNIQUE (TRIAL_NCT_ID)
)
TABLESPACE TBS_RAW_VAULT
PCTFREE 5
NOLOGGING;

COMMENT ON TABLE HUB_CLINICAL_TRIAL IS 'Data Vault 2.0 Hub - Clinical trials identified by ClinicalTrials.gov NCT ID';
COMMENT ON COLUMN HUB_CLINICAL_TRIAL.TRIAL_NCT_ID IS 'NCT identifier from ClinicalTrials.gov (business key)';

CREATE INDEX IDX_HUB_TRIAL_NCT ON HUB_CLINICAL_TRIAL (TRIAL_NCT_ID)
    TABLESPACE TBS_RAW_VAULT_IDX;

CREATE INDEX IDX_HUB_TRIAL_LOAD_DATE ON HUB_CLINICAL_TRIAL (LOAD_DATE)
    TABLESPACE TBS_RAW_VAULT_IDX;

-- ============================================================================
-- HUB_FACILITY
-- Central hub for facilities (manufacturing plants, clinical sites, labs).
-- Business Key: Internal facility identifier.
-- ============================================================================

CREATE TABLE HUB_FACILITY (
    FACILITY_KEY        RAW(16)         NOT NULL,   -- MD5 hash of FACILITY_ID
    FACILITY_ID         VARCHAR2(20)    NOT NULL,   -- Internal facility code
    LOAD_DATE           TIMESTAMP       NOT NULL,
    RECORD_SOURCE       VARCHAR2(100)   NOT NULL,
    CONSTRAINT PK_HUB_FACILITY PRIMARY KEY (FACILITY_KEY),
    CONSTRAINT UK_HUB_FACILITY_BK UNIQUE (FACILITY_ID)
)
TABLESPACE TBS_RAW_VAULT
PCTFREE 5
NOLOGGING;

COMMENT ON TABLE HUB_FACILITY IS 'Data Vault 2.0 Hub - Facilities (mfg plants, clinical sites, labs)';
COMMENT ON COLUMN HUB_FACILITY.FACILITY_ID IS 'Internal facility identifier (business key)';

CREATE INDEX IDX_HUB_FACILITY_ID ON HUB_FACILITY (FACILITY_ID)
    TABLESPACE TBS_RAW_VAULT_IDX;

CREATE INDEX IDX_HUB_FACILITY_LOAD_DATE ON HUB_FACILITY (LOAD_DATE)
    TABLESPACE TBS_RAW_VAULT_IDX;

-- ============================================================================
-- HUB_ADVERSE_EVENT
-- Central hub for adverse event reports.
-- Business Key: AE Report ID (FDA MedWatch or internal safety report ID).
-- ============================================================================

CREATE TABLE HUB_ADVERSE_EVENT (
    AE_KEY              RAW(16)         NOT NULL,   -- MD5 hash of AE_REPORT_ID
    AE_REPORT_ID        VARCHAR2(30)    NOT NULL,   -- FDA MedWatch or internal ID
    LOAD_DATE           TIMESTAMP       NOT NULL,
    RECORD_SOURCE       VARCHAR2(100)   NOT NULL,
    CONSTRAINT PK_HUB_ADVERSE_EVENT PRIMARY KEY (AE_KEY),
    CONSTRAINT UK_HUB_AE_BK UNIQUE (AE_REPORT_ID)
)
TABLESPACE TBS_RAW_VAULT
PCTFREE 5
NOLOGGING;

COMMENT ON TABLE HUB_ADVERSE_EVENT IS 'Data Vault 2.0 Hub - Adverse event reports (FDA MedWatch / internal safety)';
COMMENT ON COLUMN HUB_ADVERSE_EVENT.AE_REPORT_ID IS 'Adverse event report identifier (business key)';

CREATE INDEX IDX_HUB_AE_REPORT ON HUB_ADVERSE_EVENT (AE_REPORT_ID)
    TABLESPACE TBS_RAW_VAULT_IDX;

CREATE INDEX IDX_HUB_AE_LOAD_DATE ON HUB_ADVERSE_EVENT (LOAD_DATE)
    TABLESPACE TBS_RAW_VAULT_IDX;

-- ============================================================================
-- Grants for ETL role
-- ============================================================================

GRANT SELECT, INSERT ON HUB_DRUG TO ROLE_ETL;
GRANT SELECT, INSERT ON HUB_PATIENT TO ROLE_ETL;
GRANT SELECT, INSERT ON HUB_CLINICAL_TRIAL TO ROLE_ETL;
GRANT SELECT, INSERT ON HUB_FACILITY TO ROLE_ETL;
GRANT SELECT, INSERT ON HUB_ADVERSE_EVENT TO ROLE_ETL;

GRANT SELECT ON SEQ_HUB_DRUG TO ROLE_ETL;
GRANT SELECT ON SEQ_HUB_PATIENT TO ROLE_ETL;
GRANT SELECT ON SEQ_HUB_CLINICAL_TRIAL TO ROLE_ETL;
GRANT SELECT ON SEQ_HUB_FACILITY TO ROLE_ETL;
GRANT SELECT ON SEQ_HUB_ADVERSE_EVENT TO ROLE_ETL;
