/*******************************************************************************
 * PharmaDataVault - Populate Clinical Data Mart from Raw Vault
 *
 * PL/SQL procedures to transform raw vault data into star schema data mart.
 * Handles:
 *   - SCD Type 2 dimension management
 *   - Complex multi-hub/link/satellite joins
 *   - Incremental fact table loading
 *   - Aggregate pre-computation
 *   - Materialized view refresh
 *
 * Oracle PL/SQL compatible
 ******************************************************************************/

CREATE OR REPLACE PACKAGE PKG_POPULATE_CLINICAL_MART
AS
    PROCEDURE REFRESH_DIM_DRUG (
        p_rows_inserted     OUT NUMBER,
        p_rows_updated      OUT NUMBER
    );

    PROCEDURE REFRESH_DIM_PATIENT (
        p_rows_inserted     OUT NUMBER,
        p_rows_updated      OUT NUMBER
    );

    PROCEDURE LOAD_FACT_ENROLLMENT (
        p_load_from_date    IN DATE DEFAULT NULL,
        p_rows_loaded       OUT NUMBER
    );

    PROCEDURE LOAD_FACT_ADVERSE_EVENTS (
        p_load_from_date    IN DATE DEFAULT NULL,
        p_rows_loaded       OUT NUMBER
    );

    PROCEDURE REFRESH_ALL_MVS;

    PROCEDURE FULL_MART_REFRESH (
        p_status            OUT VARCHAR2
    );

END PKG_POPULATE_CLINICAL_MART;
/

CREATE OR REPLACE PACKAGE BODY PKG_POPULATE_CLINICAL_MART
AS
    gc_package_name     CONSTANT VARCHAR2(100) := 'PKG_POPULATE_CLINICAL_MART';
    gc_end_of_time      CONSTANT DATE := TO_DATE('9999-12-31', 'YYYY-MM-DD');
    gc_end_of_time_ts   CONSTANT TIMESTAMP := TO_TIMESTAMP('9999-12-31 23:59:59', 'YYYY-MM-DD HH24:MI:SS');

    PROCEDURE log_error (
        p_proc IN VARCHAR2, p_code IN NUMBER, p_msg IN VARCHAR2,
        p_target IN VARCHAR2 DEFAULT NULL, p_batch IN NUMBER DEFAULT NULL
    ) AS
        PRAGMA AUTONOMOUS_TRANSACTION;
    BEGIN
        INSERT INTO ETL_ERROR_LOG (
            PROCEDURE_NAME, ERROR_CODE, ERROR_MESSAGE, TARGET_TABLE, BATCH_ID
        ) VALUES (p_proc, p_code, SUBSTR(p_msg, 1, 4000), p_target, p_batch);
        COMMIT;
    END log_error;

    /*-----------------------------------------------------------------------
     * REFRESH_DIM_DRUG
     *
     * SCD Type 2 dimension refresh from raw vault.
     * Joins HUB_DRUG -> SAT_DRUG_DETAILS (current record).
     * Detects changes via ROW_HASH comparison and creates new dimension
     * rows while end-dating the prior version.
     *-----------------------------------------------------------------------*/
    PROCEDURE REFRESH_DIM_DRUG (
        p_rows_inserted     OUT NUMBER,
        p_rows_updated      OUT NUMBER
    )
    AS
        v_proc_name CONSTANT VARCHAR2(100) := gc_package_name || '.REFRESH_DIM_DRUG';
        v_inserted  NUMBER := 0;
        v_updated   NUMBER := 0;
        v_today     DATE := TRUNC(SYSDATE);

        CURSOR c_vault_drugs IS
            SELECT
                hd.DRUG_KEY                 AS VAULT_KEY,
                hd.DRUG_NDC,
                sd.DRUG_NAME,
                sd.GENERIC_NAME,
                sd.MANUFACTURER,
                sd.DRUG_FORM,
                sd.STRENGTH,
                sd.ROUTE_OF_ADMIN,
                sd.DEA_SCHEDULE,
                sd.THERAPEUTIC_CLASS,
                sd.APPROVAL_DATE,
                sd.NDA_NUMBER,
                sd.HASHDIFF                 AS SOURCE_HASH
            FROM HUB_DRUG hd
            INNER JOIN SAT_DRUG_DETAILS sd
                ON hd.DRUG_KEY = sd.DRUG_KEY
                AND sd.LOAD_END_DATE = gc_end_of_time_ts  -- Current version only
            ORDER BY hd.DRUG_NDC;

        v_dim_hash  RAW(16);
    BEGIN
        log_error(v_proc_name, 0, 'Starting DIM_DRUG refresh', 'DIM_DRUG');

        FOR rec IN c_vault_drugs LOOP
            -- Check if this drug currently exists in the dimension
            BEGIN
                SELECT ROW_HASH
                INTO v_dim_hash
                FROM DIM_DRUG
                WHERE DRUG_VAULT_KEY = rec.VAULT_KEY
                  AND IS_CURRENT = 'Y';

                -- Exists: check for changes
                IF v_dim_hash != rec.SOURCE_HASH THEN
                    -- End-date the current dimension record
                    UPDATE DIM_DRUG
                    SET EXPIRATION_DATE = v_today - 1,
                        IS_CURRENT = 'N'
                    WHERE DRUG_VAULT_KEY = rec.VAULT_KEY
                      AND IS_CURRENT = 'Y';
                    v_updated := v_updated + 1;

                    -- Insert new version
                    INSERT INTO DIM_DRUG (
                        DRUG_VAULT_KEY, DRUG_NDC, DRUG_NAME, GENERIC_NAME,
                        MANUFACTURER, DRUG_FORM, STRENGTH, ROUTE_OF_ADMIN,
                        DEA_SCHEDULE, THERAPEUTIC_CLASS, APPROVAL_DATE, NDA_NUMBER,
                        EFFECTIVE_DATE, EXPIRATION_DATE, IS_CURRENT, ROW_HASH
                    ) VALUES (
                        rec.VAULT_KEY, rec.DRUG_NDC, rec.DRUG_NAME, rec.GENERIC_NAME,
                        rec.MANUFACTURER, rec.DRUG_FORM, rec.STRENGTH,
                        rec.ROUTE_OF_ADMIN, rec.DEA_SCHEDULE, rec.THERAPEUTIC_CLASS,
                        rec.APPROVAL_DATE, rec.NDA_NUMBER,
                        v_today, gc_end_of_time, 'Y', rec.SOURCE_HASH
                    );
                    v_inserted := v_inserted + 1;
                END IF;
                -- Else: no change, skip

            EXCEPTION
                WHEN NO_DATA_FOUND THEN
                    -- New drug: insert first dimension record
                    INSERT INTO DIM_DRUG (
                        DRUG_VAULT_KEY, DRUG_NDC, DRUG_NAME, GENERIC_NAME,
                        MANUFACTURER, DRUG_FORM, STRENGTH, ROUTE_OF_ADMIN,
                        DEA_SCHEDULE, THERAPEUTIC_CLASS, APPROVAL_DATE, NDA_NUMBER,
                        EFFECTIVE_DATE, EXPIRATION_DATE, IS_CURRENT, ROW_HASH
                    ) VALUES (
                        rec.VAULT_KEY, rec.DRUG_NDC, rec.DRUG_NAME, rec.GENERIC_NAME,
                        rec.MANUFACTURER, rec.DRUG_FORM, rec.STRENGTH,
                        rec.ROUTE_OF_ADMIN, rec.DEA_SCHEDULE, rec.THERAPEUTIC_CLASS,
                        rec.APPROVAL_DATE, rec.NDA_NUMBER,
                        v_today, gc_end_of_time, 'Y', rec.SOURCE_HASH
                    );
                    v_inserted := v_inserted + 1;
            END;
        END LOOP;

        p_rows_inserted := v_inserted;
        p_rows_updated := v_updated;
        COMMIT;

        log_error(v_proc_name, 0,
            'DIM_DRUG refresh complete: ' || v_inserted || ' inserted, '
            || v_updated || ' type-2 updates', 'DIM_DRUG');

    EXCEPTION
        WHEN OTHERS THEN
            ROLLBACK;
            log_error(v_proc_name, SQLCODE, SQLERRM, 'DIM_DRUG');
            RAISE;
    END REFRESH_DIM_DRUG;

    /*-----------------------------------------------------------------------
     * REFRESH_DIM_PATIENT
     *
     * SCD Type 2 dimension refresh for patients.
     * Derives age groups and BMI categories from raw vault data.
     *-----------------------------------------------------------------------*/
    PROCEDURE REFRESH_DIM_PATIENT (
        p_rows_inserted     OUT NUMBER,
        p_rows_updated      OUT NUMBER
    )
    AS
        v_proc_name CONSTANT VARCHAR2(100) := gc_package_name || '.REFRESH_DIM_PATIENT';
        v_inserted  NUMBER := 0;
        v_updated   NUMBER := 0;
        v_today     DATE := TRUNC(SYSDATE);
    BEGIN
        log_error(v_proc_name, 0, 'Starting DIM_PATIENT refresh', 'DIM_PATIENT');

        -- Use MERGE for efficiency
        MERGE INTO DIM_PATIENT dim
        USING (
            SELECT
                hp.PATIENT_KEY                  AS VAULT_KEY,
                hp.PATIENT_MRN,
                -- Derive age group from age
                CASE
                    WHEN sp.AGE_AT_LOAD < 18 THEN 'Pediatric (<18)'
                    WHEN sp.AGE_AT_LOAD BETWEEN 18 AND 25 THEN '18-25'
                    WHEN sp.AGE_AT_LOAD BETWEEN 26 AND 35 THEN '26-35'
                    WHEN sp.AGE_AT_LOAD BETWEEN 36 AND 45 THEN '36-45'
                    WHEN sp.AGE_AT_LOAD BETWEEN 46 AND 55 THEN '46-55'
                    WHEN sp.AGE_AT_LOAD BETWEEN 56 AND 65 THEN '56-65'
                    WHEN sp.AGE_AT_LOAD > 65 THEN 'Senior (>65)'
                    ELSE 'Unknown'
                END AS AGE_GROUP,
                sp.SEX,
                sp.ETHNICITY,
                sp.RACE,
                sp.COUNTRY,
                sp.STATE_PROVINCE,
                -- Derive BMI category
                CASE
                    WHEN sp.BMI < 18.5 THEN 'Underweight'
                    WHEN sp.BMI BETWEEN 18.5 AND 24.9 THEN 'Normal'
                    WHEN sp.BMI BETWEEN 25.0 AND 29.9 THEN 'Overweight'
                    WHEN sp.BMI >= 30.0 THEN 'Obese'
                    ELSE 'Unknown'
                END AS BMI_CATEGORY,
                sp.SMOKING_STATUS,
                sp.HASHDIFF AS SOURCE_HASH
            FROM HUB_PATIENT hp
            INNER JOIN SAT_PATIENT_DEMOGRAPHICS sp
                ON hp.PATIENT_KEY = sp.PATIENT_KEY
                AND sp.LOAD_END_DATE = gc_end_of_time_ts
        ) src
        ON (dim.PATIENT_VAULT_KEY = src.VAULT_KEY AND dim.IS_CURRENT = 'Y')
        WHEN MATCHED THEN
            UPDATE SET
                dim.EXPIRATION_DATE = CASE
                    WHEN dim.ROW_HASH != src.SOURCE_HASH THEN v_today - 1
                    ELSE dim.EXPIRATION_DATE
                END,
                dim.IS_CURRENT = CASE
                    WHEN dim.ROW_HASH != src.SOURCE_HASH THEN 'N'
                    ELSE 'Y'
                END
            WHERE dim.ROW_HASH != src.SOURCE_HASH
        WHEN NOT MATCHED THEN
            INSERT (
                PATIENT_VAULT_KEY, PATIENT_MRN, AGE_GROUP, SEX, ETHNICITY,
                RACE, COUNTRY, STATE_PROVINCE, BMI_CATEGORY, SMOKING_STATUS,
                EFFECTIVE_DATE, EXPIRATION_DATE, IS_CURRENT, ROW_HASH
            ) VALUES (
                src.VAULT_KEY, src.PATIENT_MRN, src.AGE_GROUP, src.SEX,
                src.ETHNICITY, src.RACE, src.COUNTRY, src.STATE_PROVINCE,
                src.BMI_CATEGORY, src.SMOKING_STATUS,
                v_today, gc_end_of_time, 'Y', src.SOURCE_HASH
            );

        v_inserted := SQL%ROWCOUNT;

        -- Insert new versions for type-2 changes (records that were expired above)
        INSERT INTO DIM_PATIENT (
            PATIENT_VAULT_KEY, PATIENT_MRN, AGE_GROUP, SEX, ETHNICITY,
            RACE, COUNTRY, STATE_PROVINCE, BMI_CATEGORY, SMOKING_STATUS,
            EFFECTIVE_DATE, EXPIRATION_DATE, IS_CURRENT, ROW_HASH
        )
        SELECT
            hp.PATIENT_KEY,
            hp.PATIENT_MRN,
            CASE
                WHEN sp.AGE_AT_LOAD < 18 THEN 'Pediatric (<18)'
                WHEN sp.AGE_AT_LOAD BETWEEN 18 AND 25 THEN '18-25'
                WHEN sp.AGE_AT_LOAD BETWEEN 26 AND 35 THEN '26-35'
                WHEN sp.AGE_AT_LOAD BETWEEN 36 AND 45 THEN '36-45'
                WHEN sp.AGE_AT_LOAD BETWEEN 46 AND 55 THEN '46-55'
                WHEN sp.AGE_AT_LOAD BETWEEN 56 AND 65 THEN '56-65'
                WHEN sp.AGE_AT_LOAD > 65 THEN 'Senior (>65)'
                ELSE 'Unknown'
            END,
            sp.SEX,
            sp.ETHNICITY,
            sp.RACE,
            sp.COUNTRY,
            sp.STATE_PROVINCE,
            CASE
                WHEN sp.BMI < 18.5 THEN 'Underweight'
                WHEN sp.BMI BETWEEN 18.5 AND 24.9 THEN 'Normal'
                WHEN sp.BMI BETWEEN 25.0 AND 29.9 THEN 'Overweight'
                WHEN sp.BMI >= 30.0 THEN 'Obese'
                ELSE 'Unknown'
            END,
            sp.SMOKING_STATUS,
            v_today,
            gc_end_of_time,
            'Y',
            sp.HASHDIFF
        FROM HUB_PATIENT hp
        INNER JOIN SAT_PATIENT_DEMOGRAPHICS sp
            ON hp.PATIENT_KEY = sp.PATIENT_KEY
            AND sp.LOAD_END_DATE = gc_end_of_time_ts
        WHERE EXISTS (
            SELECT 1
            FROM DIM_PATIENT d
            WHERE d.PATIENT_VAULT_KEY = hp.PATIENT_KEY
              AND d.IS_CURRENT = 'N'
              AND d.EXPIRATION_DATE = v_today - 1
        )
        AND NOT EXISTS (
            SELECT 1
            FROM DIM_PATIENT d
            WHERE d.PATIENT_VAULT_KEY = hp.PATIENT_KEY
              AND d.IS_CURRENT = 'Y'
        );

        v_updated := SQL%ROWCOUNT;

        p_rows_inserted := v_inserted;
        p_rows_updated := v_updated;
        COMMIT;

    EXCEPTION
        WHEN OTHERS THEN
            ROLLBACK;
            log_error(v_proc_name, SQLCODE, SQLERRM, 'DIM_PATIENT');
            RAISE;
    END REFRESH_DIM_PATIENT;

    /*-----------------------------------------------------------------------
     * LOAD_FACT_ENROLLMENT
     *
     * Populates FACT_TRIAL_ENROLLMENT by joining across:
     *   HUB_PATIENT -> LNK_PATIENT_TRIAL -> HUB_CLINICAL_TRIAL
     *   -> LNK_TRIAL_DRUG -> HUB_DRUG
     *   -> LNK_TRIAL_FACILITY -> HUB_FACILITY
     * Plus satellite tables for descriptive attributes.
     *
     * Supports incremental loading via p_load_from_date parameter.
     *-----------------------------------------------------------------------*/
    PROCEDURE LOAD_FACT_ENROLLMENT (
        p_load_from_date    IN DATE DEFAULT NULL,
        p_rows_loaded       OUT NUMBER
    )
    AS
        v_proc_name     CONSTANT VARCHAR2(100) := gc_package_name || '.LOAD_FACT_ENROLLMENT';
        v_from_date     DATE;
        v_batch_id      NUMBER;
        v_loaded        NUMBER := 0;
    BEGIN
        -- Determine load start date (incremental or full)
        IF p_load_from_date IS NULL THEN
            -- Full reload: truncate and reload
            EXECUTE IMMEDIATE 'TRUNCATE TABLE FACT_TRIAL_ENROLLMENT';
            v_from_date := TO_DATE('1900-01-01', 'YYYY-MM-DD');
        ELSE
            v_from_date := p_load_from_date;
        END IF;

        -- Create batch tracking record
        INSERT INTO ETL_BATCH_CONTROL (BATCH_TYPE, STATUS)
        VALUES ('FACT_ENROLLMENT', 'RUNNING')
        RETURNING BATCH_ID INTO v_batch_id;

        log_error(v_proc_name, 0,
            'Starting FACT_TRIAL_ENROLLMENT load from ' || TO_CHAR(v_from_date, 'YYYY-MM-DD'),
            'FACT_TRIAL_ENROLLMENT', v_batch_id);

        -- Insert fact rows from vault joins
        INSERT INTO FACT_TRIAL_ENROLLMENT (
            DIM_PATIENT_KEY, DIM_DRUG_KEY, DIM_FACILITY_KEY,
            ENROLL_DATE_KEY, WITHDRAW_DATE_KEY,
            TRIAL_NCT_ID, TRIAL_PHASE, TRIAL_STATUS, ARM_NAME,
            ENROLLMENT_DURATION_DAYS, TOTAL_VISITS, COMPLETED_VISITS,
            PROTOCOL_DEVIATIONS, SCREEN_FAILURE_FLAG, COMPLETED_FLAG,
            WITHDRAWN_FLAG, ETL_BATCH_ID
        )
        SELECT
            dp.DIM_PATIENT_KEY,
            dd.DIM_DRUG_KEY,
            df.DIM_FACILITY_KEY,
            -- Enrollment date key (YYYYMMDD)
            TO_NUMBER(TO_CHAR(st.START_DATE, 'YYYYMMDD'))   AS ENROLL_DATE_KEY,
            -- Withdrawal date key (NULL if still enrolled)
            CASE
                WHEN esat.LOAD_END_DATE < gc_end_of_time_ts
                THEN TO_NUMBER(TO_CHAR(CAST(esat.LOAD_END_DATE AS DATE), 'YYYYMMDD'))
                ELSE NULL
            END AS WITHDRAW_DATE_KEY,
            hct.TRIAL_NCT_ID,
            st.TRIAL_PHASE,
            st.TRIAL_STATUS,
            NULL,  -- ARM_NAME: would come from a dedicated satellite
            -- Enrollment duration
            CASE
                WHEN esat.LOAD_END_DATE < gc_end_of_time_ts
                THEN CAST(esat.LOAD_END_DATE AS DATE) - st.START_DATE
                ELSE TRUNC(SYSDATE) - st.START_DATE
            END AS ENROLLMENT_DURATION_DAYS,
            NULL,   -- TOTAL_VISITS: would come from visit satellite
            NULL,   -- COMPLETED_VISITS
            0,      -- PROTOCOL_DEVIATIONS
            'N',    -- SCREEN_FAILURE_FLAG
            CASE WHEN st.TRIAL_STATUS = 'completed' THEN 'Y' ELSE 'N' END,
            CASE
                WHEN esat.LOAD_END_DATE < gc_end_of_time_ts THEN 'Y'
                ELSE 'N'
            END AS WITHDRAWN_FLAG,
            v_batch_id
        FROM LNK_PATIENT_TRIAL lpt
        -- Join to hubs
        INNER JOIN HUB_PATIENT hp
            ON lpt.PATIENT_KEY = hp.PATIENT_KEY
        INNER JOIN HUB_CLINICAL_TRIAL hct
            ON lpt.TRIAL_KEY = hct.TRIAL_KEY
        -- Join to effectivity satellite for enrollment window
        INNER JOIN ESAT_PATIENT_TRIAL esat
            ON lpt.PATIENT_TRIAL_KEY = esat.PATIENT_TRIAL_KEY
            AND esat.LOAD_DATE = (
                SELECT MAX(e2.LOAD_DATE)
                FROM ESAT_PATIENT_TRIAL e2
                WHERE e2.PATIENT_TRIAL_KEY = esat.PATIENT_TRIAL_KEY
            )
        -- Join to trial satellite for descriptive attributes
        INNER JOIN SAT_CLINICAL_TRIAL_DETAILS st
            ON hct.TRIAL_KEY = st.TRIAL_KEY
            AND st.LOAD_END_DATE = gc_end_of_time_ts
        -- Join to drug via trial-drug link
        INNER JOIN LNK_TRIAL_DRUG ltd
            ON hct.TRIAL_KEY = ltd.TRIAL_KEY
        INNER JOIN HUB_DRUG hd
            ON ltd.DRUG_KEY = hd.DRUG_KEY
        -- Join to dimensions (current versions)
        INNER JOIN DIM_PATIENT dp
            ON hp.PATIENT_KEY = dp.PATIENT_VAULT_KEY
            AND dp.IS_CURRENT = 'Y'
        INNER JOIN DIM_DRUG dd
            ON hd.DRUG_KEY = dd.DRUG_VAULT_KEY
            AND dd.IS_CURRENT = 'Y'
        -- Left join to facility (optional)
        LEFT JOIN (
            SELECT DISTINCT ltf.TRIAL_KEY, ltf.FACILITY_KEY
            FROM LNK_TRIAL_FACILITY ltf
        ) tf ON hct.TRIAL_KEY = tf.TRIAL_KEY
        LEFT JOIN DIM_FACILITY df
            ON tf.FACILITY_KEY = df.FACILITY_VAULT_KEY
            AND df.IS_CURRENT = 'Y'
        -- Incremental filter
        WHERE lpt.LOAD_DATE >= v_from_date
        -- Avoid duplicates on incremental loads
        AND NOT EXISTS (
            SELECT 1
            FROM FACT_TRIAL_ENROLLMENT fe
            WHERE fe.DIM_PATIENT_KEY = dp.DIM_PATIENT_KEY
              AND fe.DIM_DRUG_KEY = dd.DIM_DRUG_KEY
              AND fe.TRIAL_NCT_ID = hct.TRIAL_NCT_ID
        );

        v_loaded := SQL%ROWCOUNT;

        -- Update batch control
        UPDATE ETL_BATCH_CONTROL
        SET ROWS_LOADED = v_loaded,
            STATUS = 'SUCCESS',
            END_TIMESTAMP = SYSTIMESTAMP
        WHERE BATCH_ID = v_batch_id;

        p_rows_loaded := v_loaded;
        COMMIT;

        log_error(v_proc_name, 0,
            'FACT_TRIAL_ENROLLMENT loaded ' || v_loaded || ' rows',
            'FACT_TRIAL_ENROLLMENT', v_batch_id);

    EXCEPTION
        WHEN OTHERS THEN
            ROLLBACK;
            UPDATE ETL_BATCH_CONTROL
            SET STATUS = 'FAILED', END_TIMESTAMP = SYSTIMESTAMP, ERROR_MESSAGE = SQLERRM
            WHERE BATCH_ID = v_batch_id;
            COMMIT;
            log_error(v_proc_name, SQLCODE, SQLERRM, 'FACT_TRIAL_ENROLLMENT', v_batch_id);
            RAISE;
    END LOAD_FACT_ENROLLMENT;

    /*-----------------------------------------------------------------------
     * LOAD_FACT_ADVERSE_EVENTS
     *
     * Populates FACT_ADVERSE_EVENTS from vault.
     * Includes severity score derivation and time-to-onset calculation.
     *-----------------------------------------------------------------------*/
    PROCEDURE LOAD_FACT_ADVERSE_EVENTS (
        p_load_from_date    IN DATE DEFAULT NULL,
        p_rows_loaded       OUT NUMBER
    )
    AS
        v_proc_name     CONSTANT VARCHAR2(100) := gc_package_name || '.LOAD_FACT_AE';
        v_from_date     DATE;
        v_batch_id      NUMBER;
        v_loaded        NUMBER := 0;
    BEGIN
        IF p_load_from_date IS NULL THEN
            EXECUTE IMMEDIATE 'TRUNCATE TABLE FACT_ADVERSE_EVENTS';
            v_from_date := TO_DATE('1900-01-01', 'YYYY-MM-DD');
        ELSE
            v_from_date := p_load_from_date;
        END IF;

        INSERT INTO ETL_BATCH_CONTROL (BATCH_TYPE, STATUS)
        VALUES ('FACT_AE', 'RUNNING')
        RETURNING BATCH_ID INTO v_batch_id;

        INSERT INTO FACT_ADVERSE_EVENTS (
            DIM_PATIENT_KEY, DIM_DRUG_KEY, DIM_FACILITY_KEY,
            ONSET_DATE_KEY, REPORT_DATE_KEY, RESOLUTION_DATE_KEY,
            AE_REPORT_ID, AE_TERM, MEDDRA_SOC, SEVERITY, SERIOUSNESS,
            CAUSALITY, OUTCOME, TRIAL_NCT_ID,
            SEVERITY_SCORE, DURATION_DAYS, TIME_TO_ONSET_DAYS,
            REPORT_LAG_DAYS, ETL_BATCH_ID
        )
        SELECT
            dp.DIM_PATIENT_KEY,
            dd.DIM_DRUG_KEY,
            NULL,   -- DIM_FACILITY_KEY
            TO_NUMBER(TO_CHAR(sa.ONSET_DATE, 'YYYYMMDD')),
            TO_NUMBER(TO_CHAR(sa.REPORT_DATE, 'YYYYMMDD')),
            CASE
                WHEN sa.RESOLUTION_DATE IS NOT NULL
                THEN TO_NUMBER(TO_CHAR(sa.RESOLUTION_DATE, 'YYYYMMDD'))
                ELSE NULL
            END,
            hae.AE_REPORT_ID,
            sa.AE_TERM,
            sa.MEDDRA_SOC,
            sa.SEVERITY,
            sa.SERIOUSNESS,
            sa.CAUSALITY,
            sa.OUTCOME,
            NULL,   -- TRIAL_NCT_ID: join through patient-trial if needed
            -- Severity score derivation
            CASE sa.SEVERITY
                WHEN 'mild'             THEN 1
                WHEN 'moderate'         THEN 2
                WHEN 'severe'           THEN 3
                WHEN 'life_threatening' THEN 4
                WHEN 'fatal'            THEN 5
                ELSE 0
            END AS SEVERITY_SCORE,
            -- Duration: onset to resolution
            CASE
                WHEN sa.RESOLUTION_DATE IS NOT NULL
                THEN sa.RESOLUTION_DATE - sa.ONSET_DATE
                ELSE NULL
            END AS DURATION_DAYS,
            -- Time to onset: would need drug start date from patient-drug link
            NULL AS TIME_TO_ONSET_DAYS,
            -- Report lag: onset to report
            sa.REPORT_DATE - sa.ONSET_DATE AS REPORT_LAG_DAYS,
            v_batch_id
        FROM LNK_DRUG_ADVERSE_EVENT ldae
        INNER JOIN HUB_ADVERSE_EVENT hae
            ON ldae.AE_KEY = hae.AE_KEY
        INNER JOIN HUB_DRUG hd
            ON ldae.DRUG_KEY = hd.DRUG_KEY
        INNER JOIN SAT_ADVERSE_EVENT_DETAILS sa
            ON hae.AE_KEY = sa.AE_KEY
            AND sa.LOAD_END_DATE = gc_end_of_time_ts
        INNER JOIN DIM_DRUG dd
            ON hd.DRUG_KEY = dd.DRUG_VAULT_KEY
            AND dd.IS_CURRENT = 'Y'
        LEFT JOIN HUB_PATIENT hp
            ON ldae.PATIENT_KEY = hp.PATIENT_KEY
        LEFT JOIN DIM_PATIENT dp
            ON hp.PATIENT_KEY = dp.PATIENT_VAULT_KEY
            AND dp.IS_CURRENT = 'Y'
        WHERE ldae.LOAD_DATE >= v_from_date
        AND NOT EXISTS (
            SELECT 1
            FROM FACT_ADVERSE_EVENTS fae
            WHERE fae.AE_REPORT_ID = hae.AE_REPORT_ID
              AND fae.DIM_DRUG_KEY = dd.DIM_DRUG_KEY
        );

        v_loaded := SQL%ROWCOUNT;

        UPDATE ETL_BATCH_CONTROL
        SET ROWS_LOADED = v_loaded, STATUS = 'SUCCESS', END_TIMESTAMP = SYSTIMESTAMP
        WHERE BATCH_ID = v_batch_id;

        p_rows_loaded := v_loaded;
        COMMIT;

    EXCEPTION
        WHEN OTHERS THEN
            ROLLBACK;
            UPDATE ETL_BATCH_CONTROL
            SET STATUS = 'FAILED', END_TIMESTAMP = SYSTIMESTAMP, ERROR_MESSAGE = SQLERRM
            WHERE BATCH_ID = v_batch_id;
            COMMIT;
            log_error(v_proc_name, SQLCODE, SQLERRM, 'FACT_ADVERSE_EVENTS', v_batch_id);
            RAISE;
    END LOAD_FACT_ADVERSE_EVENTS;

    /*-----------------------------------------------------------------------
     * REFRESH_ALL_MVS
     * Refreshes all materialized views in the clinical mart.
     *-----------------------------------------------------------------------*/
    PROCEDURE REFRESH_ALL_MVS
    AS
        v_proc_name CONSTANT VARCHAR2(100) := gc_package_name || '.REFRESH_ALL_MVS';
    BEGIN
        log_error(v_proc_name, 0, 'Starting materialized view refresh', 'MV_*');

        DBMS_MVIEW.REFRESH('MV_AE_DRUG_SEVERITY', 'C');
        DBMS_MVIEW.REFRESH('MV_ENROLLMENT_SUMMARY', 'C');
        DBMS_MVIEW.REFRESH('MV_MFG_QUALITY', 'C');

        log_error(v_proc_name, 0, 'All materialized views refreshed', 'MV_*');

    EXCEPTION
        WHEN OTHERS THEN
            log_error(v_proc_name, SQLCODE, SQLERRM, 'MV_*');
            RAISE;
    END REFRESH_ALL_MVS;

    /*-----------------------------------------------------------------------
     * FULL_MART_REFRESH
     * Orchestrates a complete mart refresh in the correct order.
     *-----------------------------------------------------------------------*/
    PROCEDURE FULL_MART_REFRESH (
        p_status    OUT VARCHAR2
    )
    AS
        v_proc_name CONSTANT VARCHAR2(100) := gc_package_name || '.FULL_MART_REFRESH';
        v_ins NUMBER; v_upd NUMBER; v_loaded NUMBER;
    BEGIN
        log_error(v_proc_name, 0, 'Starting full clinical mart refresh', '*');

        -- Step 1: Refresh dimensions
        REFRESH_DIM_DRUG(v_ins, v_upd);
        REFRESH_DIM_PATIENT(v_ins, v_upd);

        -- Step 2: Load fact tables
        LOAD_FACT_ENROLLMENT(NULL, v_loaded);
        LOAD_FACT_ADVERSE_EVENTS(NULL, v_loaded);

        -- Step 3: Refresh materialized views
        REFRESH_ALL_MVS;

        p_status := 'SUCCESS';
        log_error(v_proc_name, 0, 'Full clinical mart refresh completed successfully', '*');

    EXCEPTION
        WHEN OTHERS THEN
            p_status := 'FAILED: ' || SQLERRM;
            log_error(v_proc_name, SQLCODE, SQLERRM, '*');
            RAISE;
    END FULL_MART_REFRESH;

END PKG_POPULATE_CLINICAL_MART;
/
