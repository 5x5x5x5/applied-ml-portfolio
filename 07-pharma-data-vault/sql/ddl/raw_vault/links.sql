/*******************************************************************************
 * PharmaDataVault - Raw Vault Link Tables DDL
 *
 * Data Vault 2.0 Link tables representing relationships between Hubs.
 * Links capture the associations/transactions between business concepts.
 *
 * Conventions:
 *   - Link hash key = MD5 of concatenated hub business keys
 *   - Foreign keys reference hub hash keys
 *   - Links are insert-only; relationships are never updated
 *   - Effectivity satellites track the validity period of relationships
 *
 * Oracle PL/SQL compatible DDL
 ******************************************************************************/

-- ============================================================================
-- LNK_PATIENT_TRIAL
-- Represents patient enrollment in a clinical trial.
-- Grain: one row per unique (patient, trial) combination.
-- ============================================================================

CREATE TABLE LNK_PATIENT_TRIAL (
    PATIENT_TRIAL_KEY   RAW(16)         NOT NULL,   -- MD5(PATIENT_MRN || '|' || TRIAL_NCT_ID)
    PATIENT_KEY         RAW(16)         NOT NULL,   -- FK to HUB_PATIENT
    TRIAL_KEY           RAW(16)         NOT NULL,   -- FK to HUB_CLINICAL_TRIAL
    LOAD_DATE           TIMESTAMP       NOT NULL,
    RECORD_SOURCE       VARCHAR2(100)   NOT NULL,
    CONSTRAINT PK_LNK_PATIENT_TRIAL PRIMARY KEY (PATIENT_TRIAL_KEY),
    CONSTRAINT FK_LPT_PATIENT FOREIGN KEY (PATIENT_KEY)
        REFERENCES HUB_PATIENT (PATIENT_KEY),
    CONSTRAINT FK_LPT_TRIAL FOREIGN KEY (TRIAL_KEY)
        REFERENCES HUB_CLINICAL_TRIAL (TRIAL_KEY),
    CONSTRAINT UK_LNK_PT_COMBO UNIQUE (PATIENT_KEY, TRIAL_KEY)
)
TABLESPACE TBS_RAW_VAULT
PCTFREE 5
NOLOGGING;

COMMENT ON TABLE LNK_PATIENT_TRIAL IS 'Data Vault 2.0 Link - Patient enrollment in clinical trial';

CREATE INDEX IDX_LPT_PATIENT ON LNK_PATIENT_TRIAL (PATIENT_KEY)
    TABLESPACE TBS_RAW_VAULT_IDX;

CREATE INDEX IDX_LPT_TRIAL ON LNK_PATIENT_TRIAL (TRIAL_KEY)
    TABLESPACE TBS_RAW_VAULT_IDX;

CREATE INDEX IDX_LPT_LOAD_DATE ON LNK_PATIENT_TRIAL (LOAD_DATE)
    TABLESPACE TBS_RAW_VAULT_IDX;

-- ============================================================================
-- LNK_TRIAL_DRUG
-- Represents a drug being used/investigated in a clinical trial.
-- Grain: one row per unique (trial, drug) combination.
-- ============================================================================

CREATE TABLE LNK_TRIAL_DRUG (
    TRIAL_DRUG_KEY      RAW(16)         NOT NULL,   -- MD5(TRIAL_NCT_ID || '|' || DRUG_NDC)
    TRIAL_KEY           RAW(16)         NOT NULL,   -- FK to HUB_CLINICAL_TRIAL
    DRUG_KEY            RAW(16)         NOT NULL,   -- FK to HUB_DRUG
    LOAD_DATE           TIMESTAMP       NOT NULL,
    RECORD_SOURCE       VARCHAR2(100)   NOT NULL,
    CONSTRAINT PK_LNK_TRIAL_DRUG PRIMARY KEY (TRIAL_DRUG_KEY),
    CONSTRAINT FK_LTD_TRIAL FOREIGN KEY (TRIAL_KEY)
        REFERENCES HUB_CLINICAL_TRIAL (TRIAL_KEY),
    CONSTRAINT FK_LTD_DRUG FOREIGN KEY (DRUG_KEY)
        REFERENCES HUB_DRUG (DRUG_KEY),
    CONSTRAINT UK_LNK_TD_COMBO UNIQUE (TRIAL_KEY, DRUG_KEY)
)
TABLESPACE TBS_RAW_VAULT
PCTFREE 5
NOLOGGING;

COMMENT ON TABLE LNK_TRIAL_DRUG IS 'Data Vault 2.0 Link - Drug investigated in clinical trial';

CREATE INDEX IDX_LTD_TRIAL ON LNK_TRIAL_DRUG (TRIAL_KEY)
    TABLESPACE TBS_RAW_VAULT_IDX;

CREATE INDEX IDX_LTD_DRUG ON LNK_TRIAL_DRUG (DRUG_KEY)
    TABLESPACE TBS_RAW_VAULT_IDX;

-- ============================================================================
-- LNK_PATIENT_DRUG
-- Represents a drug prescribed/administered to a patient.
-- Grain: one row per unique (patient, drug) combination.
-- ============================================================================

CREATE TABLE LNK_PATIENT_DRUG (
    PATIENT_DRUG_KEY    RAW(16)         NOT NULL,   -- MD5(PATIENT_MRN || '|' || DRUG_NDC)
    PATIENT_KEY         RAW(16)         NOT NULL,   -- FK to HUB_PATIENT
    DRUG_KEY            RAW(16)         NOT NULL,   -- FK to HUB_DRUG
    LOAD_DATE           TIMESTAMP       NOT NULL,
    RECORD_SOURCE       VARCHAR2(100)   NOT NULL,
    CONSTRAINT PK_LNK_PATIENT_DRUG PRIMARY KEY (PATIENT_DRUG_KEY),
    CONSTRAINT FK_LPD_PATIENT FOREIGN KEY (PATIENT_KEY)
        REFERENCES HUB_PATIENT (PATIENT_KEY),
    CONSTRAINT FK_LPD_DRUG FOREIGN KEY (DRUG_KEY)
        REFERENCES HUB_DRUG (DRUG_KEY),
    CONSTRAINT UK_LNK_PD_COMBO UNIQUE (PATIENT_KEY, DRUG_KEY)
)
TABLESPACE TBS_RAW_VAULT
PCTFREE 5
NOLOGGING;

COMMENT ON TABLE LNK_PATIENT_DRUG IS 'Data Vault 2.0 Link - Drug prescribed/administered to patient';

CREATE INDEX IDX_LPD_PATIENT ON LNK_PATIENT_DRUG (PATIENT_KEY)
    TABLESPACE TBS_RAW_VAULT_IDX;

CREATE INDEX IDX_LPD_DRUG ON LNK_PATIENT_DRUG (DRUG_KEY)
    TABLESPACE TBS_RAW_VAULT_IDX;

-- ============================================================================
-- LNK_DRUG_ADVERSE_EVENT
-- Represents an adverse event associated with a drug.
-- Grain: one row per unique (drug, adverse event) combination.
-- ============================================================================

CREATE TABLE LNK_DRUG_ADVERSE_EVENT (
    DRUG_AE_KEY         RAW(16)         NOT NULL,   -- MD5(DRUG_NDC || '|' || AE_REPORT_ID)
    DRUG_KEY            RAW(16)         NOT NULL,   -- FK to HUB_DRUG
    AE_KEY              RAW(16)         NOT NULL,   -- FK to HUB_ADVERSE_EVENT
    PATIENT_KEY         RAW(16)         NULL,        -- Optional FK to HUB_PATIENT (reporter)
    LOAD_DATE           TIMESTAMP       NOT NULL,
    RECORD_SOURCE       VARCHAR2(100)   NOT NULL,
    CONSTRAINT PK_LNK_DRUG_AE PRIMARY KEY (DRUG_AE_KEY),
    CONSTRAINT FK_LDAE_DRUG FOREIGN KEY (DRUG_KEY)
        REFERENCES HUB_DRUG (DRUG_KEY),
    CONSTRAINT FK_LDAE_AE FOREIGN KEY (AE_KEY)
        REFERENCES HUB_ADVERSE_EVENT (AE_KEY),
    CONSTRAINT FK_LDAE_PATIENT FOREIGN KEY (PATIENT_KEY)
        REFERENCES HUB_PATIENT (PATIENT_KEY),
    CONSTRAINT UK_LNK_DAE_COMBO UNIQUE (DRUG_KEY, AE_KEY)
)
TABLESPACE TBS_RAW_VAULT
PCTFREE 5
NOLOGGING;

COMMENT ON TABLE LNK_DRUG_ADVERSE_EVENT IS 'Data Vault 2.0 Link - Adverse event associated with a drug';

CREATE INDEX IDX_LDAE_DRUG ON LNK_DRUG_ADVERSE_EVENT (DRUG_KEY)
    TABLESPACE TBS_RAW_VAULT_IDX;

CREATE INDEX IDX_LDAE_AE ON LNK_DRUG_ADVERSE_EVENT (AE_KEY)
    TABLESPACE TBS_RAW_VAULT_IDX;

CREATE INDEX IDX_LDAE_PATIENT ON LNK_DRUG_ADVERSE_EVENT (PATIENT_KEY)
    TABLESPACE TBS_RAW_VAULT_IDX;

-- ============================================================================
-- LNK_TRIAL_FACILITY
-- Represents a clinical trial being conducted at a facility/site.
-- Grain: one row per unique (trial, facility) combination.
-- ============================================================================

CREATE TABLE LNK_TRIAL_FACILITY (
    TRIAL_FACILITY_KEY  RAW(16)         NOT NULL,   -- MD5(TRIAL_NCT_ID || '|' || FACILITY_ID)
    TRIAL_KEY           RAW(16)         NOT NULL,   -- FK to HUB_CLINICAL_TRIAL
    FACILITY_KEY        RAW(16)         NOT NULL,   -- FK to HUB_FACILITY
    LOAD_DATE           TIMESTAMP       NOT NULL,
    RECORD_SOURCE       VARCHAR2(100)   NOT NULL,
    CONSTRAINT PK_LNK_TRIAL_FACILITY PRIMARY KEY (TRIAL_FACILITY_KEY),
    CONSTRAINT FK_LTF_TRIAL FOREIGN KEY (TRIAL_KEY)
        REFERENCES HUB_CLINICAL_TRIAL (TRIAL_KEY),
    CONSTRAINT FK_LTF_FACILITY FOREIGN KEY (FACILITY_KEY)
        REFERENCES HUB_FACILITY (FACILITY_KEY),
    CONSTRAINT UK_LNK_TF_COMBO UNIQUE (TRIAL_KEY, FACILITY_KEY)
)
TABLESPACE TBS_RAW_VAULT
PCTFREE 5
NOLOGGING;

COMMENT ON TABLE LNK_TRIAL_FACILITY IS 'Data Vault 2.0 Link - Clinical trial conducted at facility/site';

CREATE INDEX IDX_LTF_TRIAL ON LNK_TRIAL_FACILITY (TRIAL_KEY)
    TABLESPACE TBS_RAW_VAULT_IDX;

CREATE INDEX IDX_LTF_FACILITY ON LNK_TRIAL_FACILITY (FACILITY_KEY)
    TABLESPACE TBS_RAW_VAULT_IDX;

-- ============================================================================
-- Effectivity Satellite for LNK_PATIENT_TRIAL
-- Tracks when a patient's enrollment in a trial is active/inactive.
-- ============================================================================

CREATE TABLE ESAT_PATIENT_TRIAL (
    PATIENT_TRIAL_KEY   RAW(16)         NOT NULL,
    LOAD_DATE           TIMESTAMP       NOT NULL,
    LOAD_END_DATE       TIMESTAMP       DEFAULT TO_TIMESTAMP('9999-12-31', 'YYYY-MM-DD') NOT NULL,
    RECORD_SOURCE       VARCHAR2(100)   NOT NULL,
    CONSTRAINT PK_ESAT_PT PRIMARY KEY (PATIENT_TRIAL_KEY, LOAD_DATE),
    CONSTRAINT FK_ESAT_PT FOREIGN KEY (PATIENT_TRIAL_KEY)
        REFERENCES LNK_PATIENT_TRIAL (PATIENT_TRIAL_KEY)
)
TABLESPACE TBS_RAW_VAULT
PCTFREE 10;

COMMENT ON TABLE ESAT_PATIENT_TRIAL IS 'Effectivity satellite - tracks active/inactive enrollment periods';

CREATE INDEX IDX_ESAT_PT_END ON ESAT_PATIENT_TRIAL (LOAD_END_DATE)
    TABLESPACE TBS_RAW_VAULT_IDX;

-- ============================================================================
-- Grants for ETL role
-- ============================================================================

GRANT SELECT, INSERT ON LNK_PATIENT_TRIAL TO ROLE_ETL;
GRANT SELECT, INSERT ON LNK_TRIAL_DRUG TO ROLE_ETL;
GRANT SELECT, INSERT ON LNK_PATIENT_DRUG TO ROLE_ETL;
GRANT SELECT, INSERT ON LNK_DRUG_ADVERSE_EVENT TO ROLE_ETL;
GRANT SELECT, INSERT ON LNK_TRIAL_FACILITY TO ROLE_ETL;
GRANT SELECT, INSERT, UPDATE ON ESAT_PATIENT_TRIAL TO ROLE_ETL;
