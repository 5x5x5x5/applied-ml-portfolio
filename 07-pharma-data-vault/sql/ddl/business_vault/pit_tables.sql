/*******************************************************************************
 * PharmaDataVault - Business Vault Point-in-Time (PIT) Tables DDL
 *
 * PIT tables provide efficient point-in-time lookups by pre-joining
 * the LOAD_DATE timelines of multiple satellites for a given hub.
 * This eliminates costly equi-temporal joins at query time.
 *
 * Each PIT row stores the hub key, a snapshot date, and the corresponding
 * LOAD_DATE for each satellite that was current at that snapshot date.
 *
 * Oracle PL/SQL compatible DDL
 ******************************************************************************/

-- ============================================================================
-- PIT_DRUG
-- Point-in-Time table for HUB_DRUG, referencing:
--   SAT_DRUG_DETAILS, SAT_DRUG_MANUFACTURING
-- ============================================================================

CREATE TABLE PIT_DRUG (
    PIT_DRUG_KEY            RAW(16)     NOT NULL,   -- MD5(DRUG_KEY || SNAPSHOT_DATE)
    DRUG_KEY                RAW(16)     NOT NULL,   -- FK to HUB_DRUG
    SNAPSHOT_DATE           TIMESTAMP   NOT NULL,   -- The point-in-time date
    -- Satellite load dates current as of SNAPSHOT_DATE
    SAT_DRUG_DETAILS_LDTS       TIMESTAMP   NOT NULL,
    SAT_DRUG_MFG_LDTS           TIMESTAMP   NOT NULL,
    CONSTRAINT PK_PIT_DRUG PRIMARY KEY (PIT_DRUG_KEY),
    CONSTRAINT FK_PIT_DRUG_HUB FOREIGN KEY (DRUG_KEY)
        REFERENCES HUB_DRUG (DRUG_KEY),
    CONSTRAINT FK_PIT_DRUG_SAT_DET FOREIGN KEY (DRUG_KEY, SAT_DRUG_DETAILS_LDTS)
        REFERENCES SAT_DRUG_DETAILS (DRUG_KEY, LOAD_DATE),
    CONSTRAINT FK_PIT_DRUG_SAT_MFG FOREIGN KEY (DRUG_KEY, SAT_DRUG_MFG_LDTS)
        REFERENCES SAT_DRUG_MANUFACTURING (DRUG_KEY, LOAD_DATE),
    CONSTRAINT UK_PIT_DRUG UNIQUE (DRUG_KEY, SNAPSHOT_DATE)
)
TABLESPACE TBS_BUSINESS_VAULT
PCTFREE 5
COMPRESS;

COMMENT ON TABLE PIT_DRUG IS 'Point-in-Time table for HUB_DRUG - pre-computed satellite timeline joins';
COMMENT ON COLUMN PIT_DRUG.SNAPSHOT_DATE IS 'Point in time for which satellite versions are resolved';
COMMENT ON COLUMN PIT_DRUG.SAT_DRUG_DETAILS_LDTS IS 'LOAD_DATE of SAT_DRUG_DETAILS record current at SNAPSHOT_DATE';
COMMENT ON COLUMN PIT_DRUG.SAT_DRUG_MFG_LDTS IS 'LOAD_DATE of SAT_DRUG_MANUFACTURING record current at SNAPSHOT_DATE';

CREATE INDEX IDX_PIT_DRUG_SNAP ON PIT_DRUG (SNAPSHOT_DATE)
    TABLESPACE TBS_BUSINESS_VAULT_IDX;

-- ============================================================================
-- PIT_PATIENT
-- Point-in-Time table for HUB_PATIENT, referencing:
--   SAT_PATIENT_DEMOGRAPHICS
-- ============================================================================

CREATE TABLE PIT_PATIENT (
    PIT_PATIENT_KEY         RAW(16)     NOT NULL,
    PATIENT_KEY             RAW(16)     NOT NULL,
    SNAPSHOT_DATE           TIMESTAMP   NOT NULL,
    SAT_PAT_DEMO_LDTS      TIMESTAMP   NOT NULL,
    CONSTRAINT PK_PIT_PATIENT PRIMARY KEY (PIT_PATIENT_KEY),
    CONSTRAINT FK_PIT_PAT_HUB FOREIGN KEY (PATIENT_KEY)
        REFERENCES HUB_PATIENT (PATIENT_KEY),
    CONSTRAINT FK_PIT_PAT_SAT FOREIGN KEY (PATIENT_KEY, SAT_PAT_DEMO_LDTS)
        REFERENCES SAT_PATIENT_DEMOGRAPHICS (PATIENT_KEY, LOAD_DATE),
    CONSTRAINT UK_PIT_PATIENT UNIQUE (PATIENT_KEY, SNAPSHOT_DATE)
)
TABLESPACE TBS_BUSINESS_VAULT
PCTFREE 5
COMPRESS;

COMMENT ON TABLE PIT_PATIENT IS 'Point-in-Time table for HUB_PATIENT';

CREATE INDEX IDX_PIT_PAT_SNAP ON PIT_PATIENT (SNAPSHOT_DATE)
    TABLESPACE TBS_BUSINESS_VAULT_IDX;

-- ============================================================================
-- PIT_CLINICAL_TRIAL
-- Point-in-Time table for HUB_CLINICAL_TRIAL, referencing:
--   SAT_CLINICAL_TRIAL_DETAILS
-- ============================================================================

CREATE TABLE PIT_CLINICAL_TRIAL (
    PIT_TRIAL_KEY           RAW(16)     NOT NULL,
    TRIAL_KEY               RAW(16)     NOT NULL,
    SNAPSHOT_DATE           TIMESTAMP   NOT NULL,
    SAT_TRIAL_DETAILS_LDTS  TIMESTAMP   NOT NULL,
    CONSTRAINT PK_PIT_TRIAL PRIMARY KEY (PIT_TRIAL_KEY),
    CONSTRAINT FK_PIT_TRIAL_HUB FOREIGN KEY (TRIAL_KEY)
        REFERENCES HUB_CLINICAL_TRIAL (TRIAL_KEY),
    CONSTRAINT FK_PIT_TRIAL_SAT FOREIGN KEY (TRIAL_KEY, SAT_TRIAL_DETAILS_LDTS)
        REFERENCES SAT_CLINICAL_TRIAL_DETAILS (TRIAL_KEY, LOAD_DATE),
    CONSTRAINT UK_PIT_TRIAL UNIQUE (TRIAL_KEY, SNAPSHOT_DATE)
)
TABLESPACE TBS_BUSINESS_VAULT
PCTFREE 5
COMPRESS;

COMMENT ON TABLE PIT_CLINICAL_TRIAL IS 'Point-in-Time table for HUB_CLINICAL_TRIAL';

CREATE INDEX IDX_PIT_TRIAL_SNAP ON PIT_CLINICAL_TRIAL (SNAPSHOT_DATE)
    TABLESPACE TBS_BUSINESS_VAULT_IDX;

-- ============================================================================
-- PIT_ADVERSE_EVENT
-- Point-in-Time table for HUB_ADVERSE_EVENT, referencing:
--   SAT_ADVERSE_EVENT_DETAILS
-- ============================================================================

CREATE TABLE PIT_ADVERSE_EVENT (
    PIT_AE_KEY              RAW(16)     NOT NULL,
    AE_KEY                  RAW(16)     NOT NULL,
    SNAPSHOT_DATE           TIMESTAMP   NOT NULL,
    SAT_AE_DETAILS_LDTS    TIMESTAMP   NOT NULL,
    CONSTRAINT PK_PIT_AE PRIMARY KEY (PIT_AE_KEY),
    CONSTRAINT FK_PIT_AE_HUB FOREIGN KEY (AE_KEY)
        REFERENCES HUB_ADVERSE_EVENT (AE_KEY),
    CONSTRAINT FK_PIT_AE_SAT FOREIGN KEY (AE_KEY, SAT_AE_DETAILS_LDTS)
        REFERENCES SAT_ADVERSE_EVENT_DETAILS (AE_KEY, LOAD_DATE),
    CONSTRAINT UK_PIT_AE UNIQUE (AE_KEY, SNAPSHOT_DATE)
)
TABLESPACE TBS_BUSINESS_VAULT
PCTFREE 5
COMPRESS;

COMMENT ON TABLE PIT_ADVERSE_EVENT IS 'Point-in-Time table for HUB_ADVERSE_EVENT';

-- ============================================================================
-- PL/SQL Procedure: Populate PIT_DRUG
-- Called during ETL to refresh PIT table after satellite loads.
-- ============================================================================

CREATE OR REPLACE PROCEDURE SP_POPULATE_PIT_DRUG (
    p_snapshot_date     IN TIMESTAMP DEFAULT SYSTIMESTAMP,
    p_rows_inserted     OUT NUMBER
)
AS
    v_proc_name     CONSTANT VARCHAR2(100) := 'SP_POPULATE_PIT_DRUG';
    v_count         NUMBER := 0;
BEGIN
    /*
     * For each drug in HUB_DRUG, find the satellite record current as of
     * the snapshot date by selecting the MAX(LOAD_DATE) <= snapshot date.
     * Ghost record handling: if no satellite record exists, use a sentinel
     * date of 0001-01-01 to indicate a "ghost" (missing satellite row).
     */

    MERGE INTO PIT_DRUG pit
    USING (
        SELECT
            STANDARD_HASH(h.DRUG_KEY || TO_CHAR(p_snapshot_date, 'YYYYMMDDHH24MISS'), 'MD5') AS PIT_DRUG_KEY,
            h.DRUG_KEY,
            p_snapshot_date AS SNAPSHOT_DATE,
            NVL(
                (SELECT MAX(sd.LOAD_DATE)
                 FROM SAT_DRUG_DETAILS sd
                 WHERE sd.DRUG_KEY = h.DRUG_KEY
                   AND sd.LOAD_DATE <= p_snapshot_date),
                TO_TIMESTAMP('0001-01-01', 'YYYY-MM-DD')
            ) AS SAT_DRUG_DETAILS_LDTS,
            NVL(
                (SELECT MAX(sm.LOAD_DATE)
                 FROM SAT_DRUG_MANUFACTURING sm
                 WHERE sm.DRUG_KEY = h.DRUG_KEY
                   AND sm.LOAD_DATE <= p_snapshot_date),
                TO_TIMESTAMP('0001-01-01', 'YYYY-MM-DD')
            ) AS SAT_DRUG_MFG_LDTS
        FROM HUB_DRUG h
        WHERE h.LOAD_DATE <= p_snapshot_date
    ) src
    ON (pit.DRUG_KEY = src.DRUG_KEY AND pit.SNAPSHOT_DATE = src.SNAPSHOT_DATE)
    WHEN MATCHED THEN
        UPDATE SET
            pit.SAT_DRUG_DETAILS_LDTS = src.SAT_DRUG_DETAILS_LDTS,
            pit.SAT_DRUG_MFG_LDTS = src.SAT_DRUG_MFG_LDTS
    WHEN NOT MATCHED THEN
        INSERT (PIT_DRUG_KEY, DRUG_KEY, SNAPSHOT_DATE,
                SAT_DRUG_DETAILS_LDTS, SAT_DRUG_MFG_LDTS)
        VALUES (src.PIT_DRUG_KEY, src.DRUG_KEY, src.SNAPSHOT_DATE,
                src.SAT_DRUG_DETAILS_LDTS, src.SAT_DRUG_MFG_LDTS);

    v_count := SQL%ROWCOUNT;
    p_rows_inserted := v_count;

    COMMIT;

EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        INSERT INTO ETL_ERROR_LOG (PROCEDURE_NAME, ERROR_CODE, ERROR_MESSAGE, TARGET_TABLE)
        VALUES (v_proc_name, SQLCODE, SQLERRM, 'PIT_DRUG');
        COMMIT;
        RAISE;
END SP_POPULATE_PIT_DRUG;
/

-- ============================================================================
-- PL/SQL Procedure: Populate PIT_PATIENT
-- ============================================================================

CREATE OR REPLACE PROCEDURE SP_POPULATE_PIT_PATIENT (
    p_snapshot_date     IN TIMESTAMP DEFAULT SYSTIMESTAMP,
    p_rows_inserted     OUT NUMBER
)
AS
    v_proc_name     CONSTANT VARCHAR2(100) := 'SP_POPULATE_PIT_PATIENT';
BEGIN
    MERGE INTO PIT_PATIENT pit
    USING (
        SELECT
            STANDARD_HASH(h.PATIENT_KEY || TO_CHAR(p_snapshot_date, 'YYYYMMDDHH24MISS'), 'MD5') AS PIT_PATIENT_KEY,
            h.PATIENT_KEY,
            p_snapshot_date AS SNAPSHOT_DATE,
            NVL(
                (SELECT MAX(sd.LOAD_DATE)
                 FROM SAT_PATIENT_DEMOGRAPHICS sd
                 WHERE sd.PATIENT_KEY = h.PATIENT_KEY
                   AND sd.LOAD_DATE <= p_snapshot_date),
                TO_TIMESTAMP('0001-01-01', 'YYYY-MM-DD')
            ) AS SAT_PAT_DEMO_LDTS
        FROM HUB_PATIENT h
        WHERE h.LOAD_DATE <= p_snapshot_date
    ) src
    ON (pit.PATIENT_KEY = src.PATIENT_KEY AND pit.SNAPSHOT_DATE = src.SNAPSHOT_DATE)
    WHEN MATCHED THEN
        UPDATE SET pit.SAT_PAT_DEMO_LDTS = src.SAT_PAT_DEMO_LDTS
    WHEN NOT MATCHED THEN
        INSERT (PIT_PATIENT_KEY, PATIENT_KEY, SNAPSHOT_DATE, SAT_PAT_DEMO_LDTS)
        VALUES (src.PIT_PATIENT_KEY, src.PATIENT_KEY, src.SNAPSHOT_DATE, src.SAT_PAT_DEMO_LDTS);

    p_rows_inserted := SQL%ROWCOUNT;
    COMMIT;

EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        INSERT INTO ETL_ERROR_LOG (PROCEDURE_NAME, ERROR_CODE, ERROR_MESSAGE, TARGET_TABLE)
        VALUES (v_proc_name, SQLCODE, SQLERRM, 'PIT_PATIENT');
        COMMIT;
        RAISE;
END SP_POPULATE_PIT_PATIENT;
/

-- ============================================================================
-- PL/SQL Procedure: Populate PIT_CLINICAL_TRIAL
-- ============================================================================

CREATE OR REPLACE PROCEDURE SP_POPULATE_PIT_CLINICAL_TRIAL (
    p_snapshot_date     IN TIMESTAMP DEFAULT SYSTIMESTAMP,
    p_rows_inserted     OUT NUMBER
)
AS
    v_proc_name     CONSTANT VARCHAR2(100) := 'SP_POPULATE_PIT_CLINICAL_TRIAL';
BEGIN
    MERGE INTO PIT_CLINICAL_TRIAL pit
    USING (
        SELECT
            STANDARD_HASH(h.TRIAL_KEY || TO_CHAR(p_snapshot_date, 'YYYYMMDDHH24MISS'), 'MD5') AS PIT_TRIAL_KEY,
            h.TRIAL_KEY,
            p_snapshot_date AS SNAPSHOT_DATE,
            NVL(
                (SELECT MAX(sd.LOAD_DATE)
                 FROM SAT_CLINICAL_TRIAL_DETAILS sd
                 WHERE sd.TRIAL_KEY = h.TRIAL_KEY
                   AND sd.LOAD_DATE <= p_snapshot_date),
                TO_TIMESTAMP('0001-01-01', 'YYYY-MM-DD')
            ) AS SAT_TRIAL_DETAILS_LDTS
        FROM HUB_CLINICAL_TRIAL h
        WHERE h.LOAD_DATE <= p_snapshot_date
    ) src
    ON (pit.TRIAL_KEY = src.TRIAL_KEY AND pit.SNAPSHOT_DATE = src.SNAPSHOT_DATE)
    WHEN MATCHED THEN
        UPDATE SET pit.SAT_TRIAL_DETAILS_LDTS = src.SAT_TRIAL_DETAILS_LDTS
    WHEN NOT MATCHED THEN
        INSERT (PIT_TRIAL_KEY, TRIAL_KEY, SNAPSHOT_DATE, SAT_TRIAL_DETAILS_LDTS)
        VALUES (src.PIT_TRIAL_KEY, src.TRIAL_KEY, src.SNAPSHOT_DATE, src.SAT_TRIAL_DETAILS_LDTS);

    p_rows_inserted := SQL%ROWCOUNT;
    COMMIT;

EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        INSERT INTO ETL_ERROR_LOG (PROCEDURE_NAME, ERROR_CODE, ERROR_MESSAGE, TARGET_TABLE)
        VALUES (v_proc_name, SQLCODE, SQLERRM, 'PIT_CLINICAL_TRIAL');
        COMMIT;
        RAISE;
END SP_POPULATE_PIT_CLINICAL_TRIAL;
/

-- ============================================================================
-- Grants
-- ============================================================================

GRANT SELECT, INSERT, UPDATE ON PIT_DRUG TO ROLE_ETL;
GRANT SELECT, INSERT, UPDATE ON PIT_PATIENT TO ROLE_ETL;
GRANT SELECT, INSERT, UPDATE ON PIT_CLINICAL_TRIAL TO ROLE_ETL;
GRANT SELECT, INSERT, UPDATE ON PIT_ADVERSE_EVENT TO ROLE_ETL;

GRANT SELECT ON PIT_DRUG TO ROLE_ANALYST;
GRANT SELECT ON PIT_PATIENT TO ROLE_ANALYST;
GRANT SELECT ON PIT_CLINICAL_TRIAL TO ROLE_ANALYST;
GRANT SELECT ON PIT_ADVERSE_EVENT TO ROLE_ANALYST;
