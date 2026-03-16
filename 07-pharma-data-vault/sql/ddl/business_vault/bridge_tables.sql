/*******************************************************************************
 * PharmaDataVault - Business Vault Bridge Tables DDL
 *
 * Bridge tables pre-compute many-to-many traversals between hubs through
 * links. They denormalize the link paths to avoid expensive multi-hop
 * joins at query time.
 *
 * Bridge tables are rebuilt periodically (daily) and are snapshot-based
 * with an effectivity window.
 *
 * Oracle PL/SQL compatible DDL
 ******************************************************************************/

-- ============================================================================
-- BRIDGE_PATIENT_TRIAL_DRUG
-- Pre-computed traversal: Patient -> Trial -> Drug
-- Enables direct join from patient to the drugs used in their trials.
-- ============================================================================

CREATE TABLE BRIDGE_PATIENT_TRIAL_DRUG (
    BRIDGE_PTD_KEY      RAW(16)         NOT NULL,
    PATIENT_KEY         RAW(16)         NOT NULL,
    TRIAL_KEY           RAW(16)         NOT NULL,
    DRUG_KEY            RAW(16)         NOT NULL,
    -- Link keys for lineage / audit
    PATIENT_TRIAL_KEY   RAW(16)         NOT NULL,
    TRIAL_DRUG_KEY      RAW(16)         NOT NULL,
    -- Effectivity
    SNAPSHOT_DATE       TIMESTAMP       NOT NULL,
    LOAD_DATE           TIMESTAMP       NOT NULL,
    CONSTRAINT PK_BRIDGE_PTD PRIMARY KEY (BRIDGE_PTD_KEY),
    CONSTRAINT FK_BR_PTD_PAT FOREIGN KEY (PATIENT_KEY)
        REFERENCES HUB_PATIENT (PATIENT_KEY),
    CONSTRAINT FK_BR_PTD_TRIAL FOREIGN KEY (TRIAL_KEY)
        REFERENCES HUB_CLINICAL_TRIAL (TRIAL_KEY),
    CONSTRAINT FK_BR_PTD_DRUG FOREIGN KEY (DRUG_KEY)
        REFERENCES HUB_DRUG (DRUG_KEY),
    CONSTRAINT FK_BR_PTD_LPT FOREIGN KEY (PATIENT_TRIAL_KEY)
        REFERENCES LNK_PATIENT_TRIAL (PATIENT_TRIAL_KEY),
    CONSTRAINT FK_BR_PTD_LTD FOREIGN KEY (TRIAL_DRUG_KEY)
        REFERENCES LNK_TRIAL_DRUG (TRIAL_DRUG_KEY),
    CONSTRAINT UK_BRIDGE_PTD UNIQUE (PATIENT_KEY, TRIAL_KEY, DRUG_KEY, SNAPSHOT_DATE)
)
TABLESPACE TBS_BUSINESS_VAULT
PCTFREE 5
COMPRESS;

COMMENT ON TABLE BRIDGE_PATIENT_TRIAL_DRUG IS 'Bridge table - Patient-Trial-Drug traversal for direct access';

CREATE INDEX IDX_BR_PTD_PAT ON BRIDGE_PATIENT_TRIAL_DRUG (PATIENT_KEY)
    TABLESPACE TBS_BUSINESS_VAULT_IDX;

CREATE INDEX IDX_BR_PTD_DRUG ON BRIDGE_PATIENT_TRIAL_DRUG (DRUG_KEY)
    TABLESPACE TBS_BUSINESS_VAULT_IDX;

CREATE INDEX IDX_BR_PTD_SNAP ON BRIDGE_PATIENT_TRIAL_DRUG (SNAPSHOT_DATE)
    TABLESPACE TBS_BUSINESS_VAULT_IDX;

-- ============================================================================
-- BRIDGE_DRUG_AE_PATIENT
-- Pre-computed traversal: Drug -> Adverse Event -> Patient
-- Enables pharmacovigilance queries joining drug to AE to patient.
-- ============================================================================

CREATE TABLE BRIDGE_DRUG_AE_PATIENT (
    BRIDGE_DAP_KEY      RAW(16)         NOT NULL,
    DRUG_KEY            RAW(16)         NOT NULL,
    AE_KEY              RAW(16)         NOT NULL,
    PATIENT_KEY         RAW(16)         NULL,       -- Optional (some AEs have no patient link)
    -- Link key for audit
    DRUG_AE_KEY         RAW(16)         NOT NULL,
    -- Effectivity
    SNAPSHOT_DATE       TIMESTAMP       NOT NULL,
    LOAD_DATE           TIMESTAMP       NOT NULL,
    CONSTRAINT PK_BRIDGE_DAP PRIMARY KEY (BRIDGE_DAP_KEY),
    CONSTRAINT FK_BR_DAP_DRUG FOREIGN KEY (DRUG_KEY)
        REFERENCES HUB_DRUG (DRUG_KEY),
    CONSTRAINT FK_BR_DAP_AE FOREIGN KEY (AE_KEY)
        REFERENCES HUB_ADVERSE_EVENT (AE_KEY),
    CONSTRAINT FK_BR_DAP_PAT FOREIGN KEY (PATIENT_KEY)
        REFERENCES HUB_PATIENT (PATIENT_KEY),
    CONSTRAINT FK_BR_DAP_LDAE FOREIGN KEY (DRUG_AE_KEY)
        REFERENCES LNK_DRUG_ADVERSE_EVENT (DRUG_AE_KEY),
    CONSTRAINT UK_BRIDGE_DAP UNIQUE (DRUG_KEY, AE_KEY, SNAPSHOT_DATE)
)
TABLESPACE TBS_BUSINESS_VAULT
PCTFREE 5
COMPRESS;

COMMENT ON TABLE BRIDGE_DRUG_AE_PATIENT IS 'Bridge table - Drug-AE-Patient traversal for pharmacovigilance';

CREATE INDEX IDX_BR_DAP_DRUG ON BRIDGE_DRUG_AE_PATIENT (DRUG_KEY)
    TABLESPACE TBS_BUSINESS_VAULT_IDX;

CREATE INDEX IDX_BR_DAP_AE ON BRIDGE_DRUG_AE_PATIENT (AE_KEY)
    TABLESPACE TBS_BUSINESS_VAULT_IDX;

CREATE INDEX IDX_BR_DAP_PAT ON BRIDGE_DRUG_AE_PATIENT (PATIENT_KEY)
    TABLESPACE TBS_BUSINESS_VAULT_IDX;

-- ============================================================================
-- BRIDGE_TRIAL_FACILITY_DRUG
-- Pre-computed traversal: Trial -> Facility + Trial -> Drug
-- Enables queries about what drugs are tested at which facilities.
-- ============================================================================

CREATE TABLE BRIDGE_TRIAL_FACILITY_DRUG (
    BRIDGE_TFD_KEY      RAW(16)         NOT NULL,
    TRIAL_KEY           RAW(16)         NOT NULL,
    FACILITY_KEY        RAW(16)         NOT NULL,
    DRUG_KEY            RAW(16)         NOT NULL,
    -- Link keys for audit
    TRIAL_FACILITY_KEY  RAW(16)         NOT NULL,
    TRIAL_DRUG_KEY      RAW(16)         NOT NULL,
    -- Effectivity
    SNAPSHOT_DATE       TIMESTAMP       NOT NULL,
    LOAD_DATE           TIMESTAMP       NOT NULL,
    CONSTRAINT PK_BRIDGE_TFD PRIMARY KEY (BRIDGE_TFD_KEY),
    CONSTRAINT FK_BR_TFD_TRIAL FOREIGN KEY (TRIAL_KEY)
        REFERENCES HUB_CLINICAL_TRIAL (TRIAL_KEY),
    CONSTRAINT FK_BR_TFD_FAC FOREIGN KEY (FACILITY_KEY)
        REFERENCES HUB_FACILITY (FACILITY_KEY),
    CONSTRAINT FK_BR_TFD_DRUG FOREIGN KEY (DRUG_KEY)
        REFERENCES HUB_DRUG (DRUG_KEY),
    CONSTRAINT UK_BRIDGE_TFD UNIQUE (TRIAL_KEY, FACILITY_KEY, DRUG_KEY, SNAPSHOT_DATE)
)
TABLESPACE TBS_BUSINESS_VAULT
PCTFREE 5
COMPRESS;

COMMENT ON TABLE BRIDGE_TRIAL_FACILITY_DRUG IS 'Bridge table - Trial-Facility-Drug traversal';

-- ============================================================================
-- PL/SQL Procedure: Populate BRIDGE_PATIENT_TRIAL_DRUG
-- ============================================================================

CREATE OR REPLACE PROCEDURE SP_POPULATE_BRIDGE_PTD (
    p_snapshot_date     IN TIMESTAMP DEFAULT SYSTIMESTAMP,
    p_rows_inserted     OUT NUMBER
)
AS
    v_proc_name     CONSTANT VARCHAR2(100) := 'SP_POPULATE_BRIDGE_PTD';
    v_count         NUMBER := 0;
BEGIN
    -- Delete existing rows for this snapshot date (full refresh per snapshot)
    DELETE FROM BRIDGE_PATIENT_TRIAL_DRUG
    WHERE SNAPSHOT_DATE = p_snapshot_date;

    -- Insert bridge rows by traversing active links
    INSERT INTO BRIDGE_PATIENT_TRIAL_DRUG (
        BRIDGE_PTD_KEY, PATIENT_KEY, TRIAL_KEY, DRUG_KEY,
        PATIENT_TRIAL_KEY, TRIAL_DRUG_KEY,
        SNAPSHOT_DATE, LOAD_DATE
    )
    SELECT
        STANDARD_HASH(
            lpt.PATIENT_KEY || lpt.TRIAL_KEY || ltd.DRUG_KEY
            || TO_CHAR(p_snapshot_date, 'YYYYMMDDHH24MISS'),
            'MD5'
        ) AS BRIDGE_PTD_KEY,
        lpt.PATIENT_KEY,
        lpt.TRIAL_KEY,
        ltd.DRUG_KEY,
        lpt.PATIENT_TRIAL_KEY,
        ltd.TRIAL_DRUG_KEY,
        p_snapshot_date,
        SYSTIMESTAMP
    FROM LNK_PATIENT_TRIAL lpt
    INNER JOIN LNK_TRIAL_DRUG ltd
        ON lpt.TRIAL_KEY = ltd.TRIAL_KEY
    -- Only include active enrollments (effectivity satellite check)
    WHERE EXISTS (
        SELECT 1
        FROM ESAT_PATIENT_TRIAL esat
        WHERE esat.PATIENT_TRIAL_KEY = lpt.PATIENT_TRIAL_KEY
          AND esat.LOAD_DATE <= p_snapshot_date
          AND esat.LOAD_END_DATE > p_snapshot_date
    );

    v_count := SQL%ROWCOUNT;
    p_rows_inserted := v_count;

    COMMIT;

EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        INSERT INTO ETL_ERROR_LOG (PROCEDURE_NAME, ERROR_CODE, ERROR_MESSAGE, TARGET_TABLE)
        VALUES (v_proc_name, SQLCODE, SQLERRM, 'BRIDGE_PATIENT_TRIAL_DRUG');
        COMMIT;
        RAISE;
END SP_POPULATE_BRIDGE_PTD;
/

-- ============================================================================
-- PL/SQL Procedure: Populate BRIDGE_DRUG_AE_PATIENT
-- ============================================================================

CREATE OR REPLACE PROCEDURE SP_POPULATE_BRIDGE_DAP (
    p_snapshot_date     IN TIMESTAMP DEFAULT SYSTIMESTAMP,
    p_rows_inserted     OUT NUMBER
)
AS
    v_proc_name     CONSTANT VARCHAR2(100) := 'SP_POPULATE_BRIDGE_DAP';
BEGIN
    DELETE FROM BRIDGE_DRUG_AE_PATIENT
    WHERE SNAPSHOT_DATE = p_snapshot_date;

    INSERT INTO BRIDGE_DRUG_AE_PATIENT (
        BRIDGE_DAP_KEY, DRUG_KEY, AE_KEY, PATIENT_KEY,
        DRUG_AE_KEY, SNAPSHOT_DATE, LOAD_DATE
    )
    SELECT
        STANDARD_HASH(
            ldae.DRUG_KEY || ldae.AE_KEY
            || TO_CHAR(p_snapshot_date, 'YYYYMMDDHH24MISS'),
            'MD5'
        ),
        ldae.DRUG_KEY,
        ldae.AE_KEY,
        ldae.PATIENT_KEY,      -- May be NULL
        ldae.DRUG_AE_KEY,
        p_snapshot_date,
        SYSTIMESTAMP
    FROM LNK_DRUG_ADVERSE_EVENT ldae
    WHERE ldae.LOAD_DATE <= p_snapshot_date;

    p_rows_inserted := SQL%ROWCOUNT;
    COMMIT;

EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        INSERT INTO ETL_ERROR_LOG (PROCEDURE_NAME, ERROR_CODE, ERROR_MESSAGE, TARGET_TABLE)
        VALUES (v_proc_name, SQLCODE, SQLERRM, 'BRIDGE_DRUG_AE_PATIENT');
        COMMIT;
        RAISE;
END SP_POPULATE_BRIDGE_DAP;
/

-- ============================================================================
-- PL/SQL Procedure: Populate BRIDGE_TRIAL_FACILITY_DRUG
-- ============================================================================

CREATE OR REPLACE PROCEDURE SP_POPULATE_BRIDGE_TFD (
    p_snapshot_date     IN TIMESTAMP DEFAULT SYSTIMESTAMP,
    p_rows_inserted     OUT NUMBER
)
AS
    v_proc_name     CONSTANT VARCHAR2(100) := 'SP_POPULATE_BRIDGE_TFD';
BEGIN
    DELETE FROM BRIDGE_TRIAL_FACILITY_DRUG
    WHERE SNAPSHOT_DATE = p_snapshot_date;

    INSERT INTO BRIDGE_TRIAL_FACILITY_DRUG (
        BRIDGE_TFD_KEY, TRIAL_KEY, FACILITY_KEY, DRUG_KEY,
        TRIAL_FACILITY_KEY, TRIAL_DRUG_KEY,
        SNAPSHOT_DATE, LOAD_DATE
    )
    SELECT
        STANDARD_HASH(
            ltf.TRIAL_KEY || ltf.FACILITY_KEY || ltd.DRUG_KEY
            || TO_CHAR(p_snapshot_date, 'YYYYMMDDHH24MISS'),
            'MD5'
        ),
        ltf.TRIAL_KEY,
        ltf.FACILITY_KEY,
        ltd.DRUG_KEY,
        ltf.TRIAL_FACILITY_KEY,
        ltd.TRIAL_DRUG_KEY,
        p_snapshot_date,
        SYSTIMESTAMP
    FROM LNK_TRIAL_FACILITY ltf
    INNER JOIN LNK_TRIAL_DRUG ltd
        ON ltf.TRIAL_KEY = ltd.TRIAL_KEY
    WHERE ltf.LOAD_DATE <= p_snapshot_date
      AND ltd.LOAD_DATE <= p_snapshot_date;

    p_rows_inserted := SQL%ROWCOUNT;
    COMMIT;

EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        INSERT INTO ETL_ERROR_LOG (PROCEDURE_NAME, ERROR_CODE, ERROR_MESSAGE, TARGET_TABLE)
        VALUES (v_proc_name, SQLCODE, SQLERRM, 'BRIDGE_TRIAL_FACILITY_DRUG');
        COMMIT;
        RAISE;
END SP_POPULATE_BRIDGE_TFD;
/

-- ============================================================================
-- Grants
-- ============================================================================

GRANT SELECT, INSERT, UPDATE, DELETE ON BRIDGE_PATIENT_TRIAL_DRUG TO ROLE_ETL;
GRANT SELECT, INSERT, UPDATE, DELETE ON BRIDGE_DRUG_AE_PATIENT TO ROLE_ETL;
GRANT SELECT, INSERT, UPDATE, DELETE ON BRIDGE_TRIAL_FACILITY_DRUG TO ROLE_ETL;

GRANT SELECT ON BRIDGE_PATIENT_TRIAL_DRUG TO ROLE_ANALYST;
GRANT SELECT ON BRIDGE_DRUG_AE_PATIENT TO ROLE_ANALYST;
GRANT SELECT ON BRIDGE_TRIAL_FACILITY_DRUG TO ROLE_ANALYST;
