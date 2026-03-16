/*******************************************************************************
 * PharmaDataVault - PL/SQL Procedures: Load Satellite Tables
 *
 * Satellite loading follows the Data Vault 2.0 pattern:
 *   1. Compare incoming hashdiff with latest satellite hashdiff
 *   2. Only insert if hashdiff differs (data has changed)
 *   3. Close out previous record by setting LOAD_END_DATE
 *   4. Handle ghost records (hub exists but no satellite data)
 *   5. Maintain full history via insert-only pattern
 *
 * Oracle PL/SQL compatible
 ******************************************************************************/

-- ============================================================================
-- PKG_LOAD_SATELLITES: Package for all satellite loading procedures
-- ============================================================================

CREATE OR REPLACE PACKAGE PKG_LOAD_SATELLITES
AS
    PROCEDURE LOAD_SAT_DRUG_DETAILS (
        p_batch_id          IN NUMBER,
        p_rows_inserted     OUT NUMBER,
        p_rows_unchanged    OUT NUMBER
    );

    PROCEDURE LOAD_SAT_PATIENT_DEMOGRAPHICS (
        p_batch_id          IN NUMBER,
        p_rows_inserted     OUT NUMBER,
        p_rows_unchanged    OUT NUMBER
    );

    PROCEDURE LOAD_SAT_CLINICAL_TRIAL_DETAILS (
        p_batch_id          IN NUMBER,
        p_rows_inserted     OUT NUMBER,
        p_rows_unchanged    OUT NUMBER
    );

    PROCEDURE LOAD_SAT_ADVERSE_EVENT_DETAILS (
        p_batch_id          IN NUMBER,
        p_rows_inserted     OUT NUMBER,
        p_rows_unchanged    OUT NUMBER
    );

    PROCEDURE LOAD_SAT_DRUG_MANUFACTURING (
        p_batch_id          IN NUMBER,
        p_rows_inserted     OUT NUMBER,
        p_rows_unchanged    OUT NUMBER
    );

END PKG_LOAD_SATELLITES;
/

CREATE OR REPLACE PACKAGE BODY PKG_LOAD_SATELLITES
AS
    gc_package_name     CONSTANT VARCHAR2(100) := 'PKG_LOAD_SATELLITES';
    gc_bulk_limit       CONSTANT PLS_INTEGER := 5000;
    gc_ghost_date       CONSTANT TIMESTAMP := TO_TIMESTAMP('0001-01-01', 'YYYY-MM-DD');
    gc_end_of_time      CONSTANT TIMESTAMP := TO_TIMESTAMP('9999-12-31 23:59:59', 'YYYY-MM-DD HH24:MI:SS');

    /*-----------------------------------------------------------------------
     * Autonomous error/progress logging
     *-----------------------------------------------------------------------*/
    PROCEDURE log_error (
        p_proc   IN VARCHAR2,
        p_code   IN NUMBER,
        p_msg    IN VARCHAR2,
        p_target IN VARCHAR2 DEFAULT NULL,
        p_key    IN VARCHAR2 DEFAULT NULL,
        p_batch  IN NUMBER DEFAULT NULL
    )
    AS
        PRAGMA AUTONOMOUS_TRANSACTION;
    BEGIN
        INSERT INTO ETL_ERROR_LOG (
            PROCEDURE_NAME, ERROR_CODE, ERROR_MESSAGE,
            SOURCE_TABLE, TARGET_TABLE, RECORD_KEY, BATCH_ID
        ) VALUES (
            p_proc, p_code, SUBSTR(p_msg, 1, 4000),
            'STG_*_CLEAN', p_target, p_key, p_batch
        );
        COMMIT;
    END log_error;

    /*-----------------------------------------------------------------------
     * LOAD_SAT_DRUG_DETAILS
     *
     * Loads drug descriptive attributes from staging into satellite.
     * Uses hashdiff comparison to detect changes. Only inserts new/changed
     * records. End-dates the previous current record.
     *-----------------------------------------------------------------------*/
    PROCEDURE LOAD_SAT_DRUG_DETAILS (
        p_batch_id          IN NUMBER,
        p_rows_inserted     OUT NUMBER,
        p_rows_unchanged    OUT NUMBER
    )
    AS
        v_proc_name     CONSTANT VARCHAR2(100) := gc_package_name || '.LOAD_SAT_DRUG_DETAILS';
        v_inserted      NUMBER := 0;
        v_unchanged     NUMBER := 0;
        v_load_date     TIMESTAMP := SYSTIMESTAMP;
        v_existing_diff RAW(16);

        CURSOR c_staged_drugs IS
            SELECT
                sc.DRUG_HASH_KEY,
                sc.DRUG_HASHDIFF,
                sc.RECORD_SOURCE,
                sc.DRUG_NAME,
                sc.GENERIC_NAME,
                sc.MANUFACTURER,
                sc.DRUG_FORM,
                sc.STRENGTH,
                sc.STRENGTH_UNIT,
                sc.ROUTE_OF_ADMIN,
                sc.DEA_SCHEDULE,
                sc.APPROVAL_DATE,
                sc.THERAPEUTIC_CLASS,
                sc.NDA_NUMBER
            FROM STG_DRUG_CLEAN sc
            WHERE sc.BATCH_ID = p_batch_id
            ORDER BY sc.DRUG_HASH_KEY;

    BEGIN
        log_error(v_proc_name, 0, 'Starting SAT_DRUG_DETAILS load for batch ' || p_batch_id,
                  'SAT_DRUG_DETAILS', NULL, p_batch_id);

        FOR rec IN c_staged_drugs LOOP
            BEGIN
                -- Look up the hashdiff of the current satellite record
                BEGIN
                    SELECT HASHDIFF
                    INTO v_existing_diff
                    FROM SAT_DRUG_DETAILS
                    WHERE DRUG_KEY = rec.DRUG_HASH_KEY
                      AND LOAD_END_DATE = gc_end_of_time;
                EXCEPTION
                    WHEN NO_DATA_FOUND THEN
                        v_existing_diff := NULL;  -- Ghost record or first load
                    WHEN TOO_MANY_ROWS THEN
                        -- Data integrity issue: multiple current records
                        log_error(v_proc_name, -20030,
                            'Multiple current records in SAT_DRUG_DETAILS',
                            'SAT_DRUG_DETAILS', 'DRUG_KEY=' || RAWTOHEX(rec.DRUG_HASH_KEY),
                            p_batch_id);
                        -- Take the latest one
                        SELECT HASHDIFF
                        INTO v_existing_diff
                        FROM (
                            SELECT HASHDIFF
                            FROM SAT_DRUG_DETAILS
                            WHERE DRUG_KEY = rec.DRUG_HASH_KEY
                              AND LOAD_END_DATE = gc_end_of_time
                            ORDER BY LOAD_DATE DESC
                        )
                        WHERE ROWNUM = 1;
                END;

                -- Compare hashdiffs: only insert if changed or new
                IF v_existing_diff IS NULL OR v_existing_diff != rec.DRUG_HASHDIFF THEN

                    -- End-date the previous current record
                    IF v_existing_diff IS NOT NULL THEN
                        UPDATE SAT_DRUG_DETAILS
                        SET LOAD_END_DATE = v_load_date
                        WHERE DRUG_KEY = rec.DRUG_HASH_KEY
                          AND LOAD_END_DATE = gc_end_of_time;
                    END IF;

                    -- Insert new satellite record
                    INSERT INTO SAT_DRUG_DETAILS (
                        DRUG_KEY, LOAD_DATE, LOAD_END_DATE, HASHDIFF, RECORD_SOURCE,
                        DRUG_NAME, GENERIC_NAME, MANUFACTURER, DRUG_FORM,
                        STRENGTH, STRENGTH_UNIT, ROUTE_OF_ADMIN, DEA_SCHEDULE,
                        APPROVAL_DATE, THERAPEUTIC_CLASS, NDA_NUMBER
                    ) VALUES (
                        rec.DRUG_HASH_KEY, v_load_date, gc_end_of_time,
                        rec.DRUG_HASHDIFF, rec.RECORD_SOURCE,
                        rec.DRUG_NAME, rec.GENERIC_NAME, rec.MANUFACTURER,
                        rec.DRUG_FORM, rec.STRENGTH, rec.STRENGTH_UNIT,
                        rec.ROUTE_OF_ADMIN, rec.DEA_SCHEDULE, rec.APPROVAL_DATE,
                        rec.THERAPEUTIC_CLASS, rec.NDA_NUMBER
                    );

                    v_inserted := v_inserted + 1;
                ELSE
                    v_unchanged := v_unchanged + 1;
                END IF;

            EXCEPTION
                WHEN OTHERS THEN
                    log_error(v_proc_name, SQLCODE,
                        SQLERRM || ' | ' || DBMS_UTILITY.FORMAT_ERROR_BACKTRACE,
                        'SAT_DRUG_DETAILS',
                        'DRUG_KEY=' || RAWTOHEX(rec.DRUG_HASH_KEY),
                        p_batch_id);
            END;

            -- Commit periodically
            IF MOD(v_inserted + v_unchanged, gc_bulk_limit) = 0 THEN
                COMMIT;
            END IF;
        END LOOP;

        p_rows_inserted := v_inserted;
        p_rows_unchanged := v_unchanged;
        COMMIT;

        log_error(v_proc_name, 0,
            'SAT_DRUG_DETAILS load complete: ' || v_inserted || ' inserted, '
            || v_unchanged || ' unchanged',
            'SAT_DRUG_DETAILS', NULL, p_batch_id);

    EXCEPTION
        WHEN OTHERS THEN
            ROLLBACK;
            log_error(v_proc_name, SQLCODE,
                SQLERRM || ' | ' || DBMS_UTILITY.FORMAT_ERROR_BACKTRACE,
                'SAT_DRUG_DETAILS', NULL, p_batch_id);
            RAISE;
    END LOAD_SAT_DRUG_DETAILS;

    /*-----------------------------------------------------------------------
     * LOAD_SAT_PATIENT_DEMOGRAPHICS
     *
     * Loads patient demographic data with change detection.
     * Includes ghost record handling for patients in hub without demographics.
     *-----------------------------------------------------------------------*/
    PROCEDURE LOAD_SAT_PATIENT_DEMOGRAPHICS (
        p_batch_id          IN NUMBER,
        p_rows_inserted     OUT NUMBER,
        p_rows_unchanged    OUT NUMBER
    )
    AS
        v_proc_name     CONSTANT VARCHAR2(100) := gc_package_name || '.LOAD_SAT_PAT_DEMO';
        v_inserted      NUMBER := 0;
        v_unchanged     NUMBER := 0;
        v_load_date     TIMESTAMP := SYSTIMESTAMP;
    BEGIN
        log_error(v_proc_name, 0, 'Starting SAT_PATIENT_DEMOGRAPHICS load', NULL, NULL, p_batch_id);

        -- Merge pattern: insert changed or new records, skip unchanged
        MERGE INTO SAT_PATIENT_DEMOGRAPHICS sat
        USING (
            SELECT
                sc.PATIENT_HASH_KEY AS PATIENT_KEY,
                v_load_date AS LOAD_DATE,
                gc_end_of_time AS LOAD_END_DATE,
                sc.PAT_HASHDIFF AS HASHDIFF,
                sc.RECORD_SOURCE,
                sc.DATE_OF_BIRTH,
                sc.AGE_AT_LOAD,
                sc.SEX,
                sc.ETHNICITY,
                sc.RACE,
                sc.WEIGHT_KG,
                sc.HEIGHT_CM,
                sc.BMI,
                sc.COUNTRY,
                sc.STATE_PROVINCE,
                sc.SMOKING_STATUS
            FROM STG_PATIENT_CLEAN sc
            WHERE sc.BATCH_ID = p_batch_id
        ) src
        ON (
            sat.PATIENT_KEY = src.PATIENT_KEY
            AND sat.LOAD_END_DATE = gc_end_of_time
            AND sat.HASHDIFF = src.HASHDIFF  -- Match means no change
        )
        WHEN NOT MATCHED THEN
            INSERT (
                PATIENT_KEY, LOAD_DATE, LOAD_END_DATE, HASHDIFF, RECORD_SOURCE,
                DATE_OF_BIRTH, AGE_AT_LOAD, SEX, ETHNICITY, RACE,
                WEIGHT_KG, HEIGHT_CM, BMI, COUNTRY, STATE_PROVINCE, SMOKING_STATUS
            ) VALUES (
                src.PATIENT_KEY, src.LOAD_DATE, src.LOAD_END_DATE, src.HASHDIFF,
                src.RECORD_SOURCE, src.DATE_OF_BIRTH, src.AGE_AT_LOAD, src.SEX,
                src.ETHNICITY, src.RACE, src.WEIGHT_KG, src.HEIGHT_CM, src.BMI,
                src.COUNTRY, src.STATE_PROVINCE, src.SMOKING_STATUS
            );

        v_inserted := SQL%ROWCOUNT;

        -- End-date previous records for patients that got new inserts
        UPDATE SAT_PATIENT_DEMOGRAPHICS old_sat
        SET old_sat.LOAD_END_DATE = v_load_date
        WHERE old_sat.LOAD_END_DATE = gc_end_of_time
          AND old_sat.LOAD_DATE < v_load_date
          AND EXISTS (
              SELECT 1
              FROM SAT_PATIENT_DEMOGRAPHICS new_sat
              WHERE new_sat.PATIENT_KEY = old_sat.PATIENT_KEY
                AND new_sat.LOAD_DATE = v_load_date
          );

        -- Ghost record handling: create ghost records for hub patients
        -- that have no satellite data at all
        INSERT INTO SAT_PATIENT_DEMOGRAPHICS (
            PATIENT_KEY, LOAD_DATE, LOAD_END_DATE, HASHDIFF, RECORD_SOURCE,
            SEX
        )
        SELECT
            hp.PATIENT_KEY,
            gc_ghost_date,
            gc_end_of_time,
            STANDARD_HASH('GHOST_RECORD', 'MD5'),
            'SYSTEM:GHOST',
            'U'   -- Unknown sex for ghost record
        FROM HUB_PATIENT hp
        WHERE NOT EXISTS (
            SELECT 1
            FROM SAT_PATIENT_DEMOGRAPHICS sat
            WHERE sat.PATIENT_KEY = hp.PATIENT_KEY
        );

        -- Count unchanged (total staged minus inserted)
        SELECT COUNT(*) - v_inserted
        INTO v_unchanged
        FROM STG_PATIENT_CLEAN
        WHERE BATCH_ID = p_batch_id;

        p_rows_inserted := v_inserted;
        p_rows_unchanged := v_unchanged;
        COMMIT;

        log_error(v_proc_name, 0,
            'SAT_PATIENT_DEMOGRAPHICS load complete: ' || v_inserted || ' inserted',
            'SAT_PATIENT_DEMOGRAPHICS', NULL, p_batch_id);

    EXCEPTION
        WHEN OTHERS THEN
            ROLLBACK;
            log_error(v_proc_name, SQLCODE, SQLERRM,
                'SAT_PATIENT_DEMOGRAPHICS', NULL, p_batch_id);
            RAISE;
    END LOAD_SAT_PATIENT_DEMOGRAPHICS;

    /*-----------------------------------------------------------------------
     * LOAD_SAT_CLINICAL_TRIAL_DETAILS
     *
     * Loads clinical trial descriptive data. Trial status changes are
     * critical for regulatory reporting, so every change is captured.
     *-----------------------------------------------------------------------*/
    PROCEDURE LOAD_SAT_CLINICAL_TRIAL_DETAILS (
        p_batch_id          IN NUMBER,
        p_rows_inserted     OUT NUMBER,
        p_rows_unchanged    OUT NUMBER
    )
    AS
        v_proc_name     CONSTANT VARCHAR2(100) := gc_package_name || '.LOAD_SAT_TRIAL';
        v_inserted      NUMBER := 0;
        v_unchanged     NUMBER := 0;
        v_load_date     TIMESTAMP := SYSTIMESTAMP;
        v_existing_diff RAW(16);

        CURSOR c_staged_trials IS
            SELECT
                sc.TRIAL_HASH_KEY,
                sc.TRIAL_HASHDIFF,
                sc.RECORD_SOURCE,
                sc.TRIAL_TITLE,
                sc.TRIAL_PHASE,
                sc.TRIAL_STATUS,
                sc.START_DATE,
                sc.ESTIMATED_END_DATE,
                sc.ACTUAL_END_DATE,
                sc.SPONSOR,
                sc.LEAD_INVESTIGATOR,
                sc.PROTOCOL_NUMBER,
                sc.PROTOCOL_VERSION,
                sc.TARGET_ENROLLMENT,
                sc.ACTUAL_ENROLLMENT,
                sc.THERAPEUTIC_AREA,
                sc.PRIMARY_ENDPOINT,
                sc.IND_NUMBER
            FROM STG_TRIAL_CLEAN sc
            WHERE sc.BATCH_ID = p_batch_id;

    BEGIN
        FOR rec IN c_staged_trials LOOP
            -- Look up current hashdiff
            BEGIN
                SELECT HASHDIFF
                INTO v_existing_diff
                FROM SAT_CLINICAL_TRIAL_DETAILS
                WHERE TRIAL_KEY = rec.TRIAL_HASH_KEY
                  AND LOAD_END_DATE = gc_end_of_time;
            EXCEPTION
                WHEN NO_DATA_FOUND THEN
                    v_existing_diff := NULL;
                WHEN TOO_MANY_ROWS THEN
                    SELECT HASHDIFF INTO v_existing_diff
                    FROM (
                        SELECT HASHDIFF FROM SAT_CLINICAL_TRIAL_DETAILS
                        WHERE TRIAL_KEY = rec.TRIAL_HASH_KEY
                          AND LOAD_END_DATE = gc_end_of_time
                        ORDER BY LOAD_DATE DESC
                    ) WHERE ROWNUM = 1;
            END;

            IF v_existing_diff IS NULL OR v_existing_diff != rec.TRIAL_HASHDIFF THEN
                -- End-date previous record
                IF v_existing_diff IS NOT NULL THEN
                    UPDATE SAT_CLINICAL_TRIAL_DETAILS
                    SET LOAD_END_DATE = v_load_date
                    WHERE TRIAL_KEY = rec.TRIAL_HASH_KEY
                      AND LOAD_END_DATE = gc_end_of_time;
                END IF;

                -- Insert new version
                INSERT INTO SAT_CLINICAL_TRIAL_DETAILS (
                    TRIAL_KEY, LOAD_DATE, LOAD_END_DATE, HASHDIFF, RECORD_SOURCE,
                    TRIAL_TITLE, TRIAL_PHASE, TRIAL_STATUS, START_DATE,
                    ESTIMATED_END_DATE, ACTUAL_END_DATE, SPONSOR, LEAD_INVESTIGATOR,
                    PROTOCOL_NUMBER, PROTOCOL_VERSION, TARGET_ENROLLMENT,
                    ACTUAL_ENROLLMENT, THERAPEUTIC_AREA, PRIMARY_ENDPOINT, IND_NUMBER
                ) VALUES (
                    rec.TRIAL_HASH_KEY, v_load_date, gc_end_of_time,
                    rec.TRIAL_HASHDIFF, rec.RECORD_SOURCE,
                    rec.TRIAL_TITLE, rec.TRIAL_PHASE, rec.TRIAL_STATUS,
                    rec.START_DATE, rec.ESTIMATED_END_DATE, rec.ACTUAL_END_DATE,
                    rec.SPONSOR, rec.LEAD_INVESTIGATOR, rec.PROTOCOL_NUMBER,
                    rec.PROTOCOL_VERSION, rec.TARGET_ENROLLMENT,
                    rec.ACTUAL_ENROLLMENT, rec.THERAPEUTIC_AREA,
                    rec.PRIMARY_ENDPOINT, rec.IND_NUMBER
                );

                v_inserted := v_inserted + 1;
            ELSE
                v_unchanged := v_unchanged + 1;
            END IF;
        END LOOP;

        p_rows_inserted := v_inserted;
        p_rows_unchanged := v_unchanged;
        COMMIT;

    EXCEPTION
        WHEN OTHERS THEN
            ROLLBACK;
            log_error(v_proc_name, SQLCODE, SQLERRM,
                'SAT_CLINICAL_TRIAL_DETAILS', NULL, p_batch_id);
            RAISE;
    END LOAD_SAT_CLINICAL_TRIAL_DETAILS;

    /*-----------------------------------------------------------------------
     * LOAD_SAT_ADVERSE_EVENT_DETAILS
     *
     * Loads adverse event details. Particularly important for
     * pharmacovigilance - every update to an AE report is captured.
     *-----------------------------------------------------------------------*/
    PROCEDURE LOAD_SAT_ADVERSE_EVENT_DETAILS (
        p_batch_id          IN NUMBER,
        p_rows_inserted     OUT NUMBER,
        p_rows_unchanged    OUT NUMBER
    )
    AS
        v_proc_name     CONSTANT VARCHAR2(100) := gc_package_name || '.LOAD_SAT_AE';
        v_inserted      NUMBER := 0;
        v_unchanged     NUMBER := 0;
        v_load_date     TIMESTAMP := SYSTIMESTAMP;
        v_existing_diff RAW(16);

        CURSOR c_staged_aes IS
            SELECT
                sc.AE_HASH_KEY,
                sc.AE_HASHDIFF,
                sc.RECORD_SOURCE,
                sc.AE_TERM,
                sc.AE_DESCRIPTION,
                sc.SEVERITY,
                sc.SERIOUSNESS,
                sc.CAUSALITY,
                sc.OUTCOME,
                sc.ONSET_DATE,
                sc.RESOLUTION_DATE,
                sc.REPORT_DATE,
                sc.REPORTER_TYPE,
                sc.MEDDRA_PT_CODE,
                sc.MEDDRA_SOC,
                sc.EXPECTEDNESS,
                sc.ACTION_TAKEN
            FROM STG_AE_CLEAN sc
            WHERE sc.BATCH_ID = p_batch_id;

    BEGIN
        FOR rec IN c_staged_aes LOOP
            BEGIN
                SELECT HASHDIFF INTO v_existing_diff
                FROM SAT_ADVERSE_EVENT_DETAILS
                WHERE AE_KEY = rec.AE_HASH_KEY
                  AND LOAD_END_DATE = gc_end_of_time;
            EXCEPTION
                WHEN NO_DATA_FOUND THEN v_existing_diff := NULL;
                WHEN TOO_MANY_ROWS THEN
                    SELECT HASHDIFF INTO v_existing_diff FROM (
                        SELECT HASHDIFF FROM SAT_ADVERSE_EVENT_DETAILS
                        WHERE AE_KEY = rec.AE_HASH_KEY
                          AND LOAD_END_DATE = gc_end_of_time
                        ORDER BY LOAD_DATE DESC
                    ) WHERE ROWNUM = 1;
            END;

            IF v_existing_diff IS NULL OR v_existing_diff != rec.AE_HASHDIFF THEN
                IF v_existing_diff IS NOT NULL THEN
                    UPDATE SAT_ADVERSE_EVENT_DETAILS
                    SET LOAD_END_DATE = v_load_date
                    WHERE AE_KEY = rec.AE_HASH_KEY
                      AND LOAD_END_DATE = gc_end_of_time;
                END IF;

                INSERT INTO SAT_ADVERSE_EVENT_DETAILS (
                    AE_KEY, LOAD_DATE, LOAD_END_DATE, HASHDIFF, RECORD_SOURCE,
                    AE_TERM, AE_DESCRIPTION, SEVERITY, SERIOUSNESS, CAUSALITY,
                    OUTCOME, ONSET_DATE, RESOLUTION_DATE, REPORT_DATE,
                    REPORTER_TYPE, MEDDRA_PT_CODE, MEDDRA_SOC, EXPECTEDNESS,
                    ACTION_TAKEN
                ) VALUES (
                    rec.AE_HASH_KEY, v_load_date, gc_end_of_time,
                    rec.AE_HASHDIFF, rec.RECORD_SOURCE,
                    rec.AE_TERM, rec.AE_DESCRIPTION, rec.SEVERITY,
                    rec.SERIOUSNESS, rec.CAUSALITY, rec.OUTCOME,
                    rec.ONSET_DATE, rec.RESOLUTION_DATE, rec.REPORT_DATE,
                    rec.REPORTER_TYPE, rec.MEDDRA_PT_CODE, rec.MEDDRA_SOC,
                    rec.EXPECTEDNESS, rec.ACTION_TAKEN
                );
                v_inserted := v_inserted + 1;
            ELSE
                v_unchanged := v_unchanged + 1;
            END IF;
        END LOOP;

        p_rows_inserted := v_inserted;
        p_rows_unchanged := v_unchanged;
        COMMIT;

    EXCEPTION
        WHEN OTHERS THEN
            ROLLBACK;
            log_error(v_proc_name, SQLCODE, SQLERRM,
                'SAT_ADVERSE_EVENT_DETAILS', NULL, p_batch_id);
            RAISE;
    END LOAD_SAT_ADVERSE_EVENT_DETAILS;

    /*-----------------------------------------------------------------------
     * LOAD_SAT_DRUG_MANUFACTURING
     *
     * Loads drug manufacturing lot data. QC status changes and lot
     * updates are tracked for GMP compliance and recall support.
     *-----------------------------------------------------------------------*/
    PROCEDURE LOAD_SAT_DRUG_MANUFACTURING (
        p_batch_id          IN NUMBER,
        p_rows_inserted     OUT NUMBER,
        p_rows_unchanged    OUT NUMBER
    )
    AS
        v_proc_name     CONSTANT VARCHAR2(100) := gc_package_name || '.LOAD_SAT_MFG';
        v_inserted      NUMBER := 0;
        v_unchanged     NUMBER := 0;
        v_load_date     TIMESTAMP := SYSTIMESTAMP;
        v_existing_diff RAW(16);

        CURSOR c_staged_mfg IS
            SELECT
                sc.DRUG_HASH_KEY,
                sc.MFG_HASHDIFF,
                sc.RECORD_SOURCE,
                sc.LOT_NUMBER,
                sc.BATCH_SIZE,
                sc.BATCH_SIZE_UNIT,
                sc.MFG_DATE,
                sc.EXPIRY_DATE,
                sc.FACILITY_ID,
                sc.MFG_LINE,
                sc.QC_STATUS,
                sc.QC_DATE,
                sc.QC_ANALYST,
                sc.YIELD_PERCENT,
                sc.DEVIATION_FLAG,
                sc.DEVIATION_ID,
                sc.RELEASE_DATE,
                sc.STORAGE_CONDITIONS
            FROM STG_MFG_CLEAN sc
            WHERE sc.BATCH_ID = p_batch_id;

    BEGIN
        FOR rec IN c_staged_mfg LOOP
            BEGIN
                SELECT HASHDIFF INTO v_existing_diff
                FROM SAT_DRUG_MANUFACTURING
                WHERE DRUG_KEY = rec.DRUG_HASH_KEY
                  AND LOAD_END_DATE = gc_end_of_time;
            EXCEPTION
                WHEN NO_DATA_FOUND THEN v_existing_diff := NULL;
                WHEN TOO_MANY_ROWS THEN
                    SELECT HASHDIFF INTO v_existing_diff FROM (
                        SELECT HASHDIFF FROM SAT_DRUG_MANUFACTURING
                        WHERE DRUG_KEY = rec.DRUG_HASH_KEY
                          AND LOAD_END_DATE = gc_end_of_time
                        ORDER BY LOAD_DATE DESC
                    ) WHERE ROWNUM = 1;
            END;

            IF v_existing_diff IS NULL OR v_existing_diff != rec.MFG_HASHDIFF THEN
                IF v_existing_diff IS NOT NULL THEN
                    UPDATE SAT_DRUG_MANUFACTURING
                    SET LOAD_END_DATE = v_load_date
                    WHERE DRUG_KEY = rec.DRUG_HASH_KEY
                      AND LOAD_END_DATE = gc_end_of_time;
                END IF;

                INSERT INTO SAT_DRUG_MANUFACTURING (
                    DRUG_KEY, LOAD_DATE, LOAD_END_DATE, HASHDIFF, RECORD_SOURCE,
                    LOT_NUMBER, BATCH_SIZE, BATCH_SIZE_UNIT, MFG_DATE, EXPIRY_DATE,
                    FACILITY_ID, MFG_LINE, QC_STATUS, QC_DATE, QC_ANALYST,
                    YIELD_PERCENT, DEVIATION_FLAG, DEVIATION_ID, RELEASE_DATE,
                    STORAGE_CONDITIONS
                ) VALUES (
                    rec.DRUG_HASH_KEY, v_load_date, gc_end_of_time,
                    rec.MFG_HASHDIFF, rec.RECORD_SOURCE,
                    rec.LOT_NUMBER, rec.BATCH_SIZE, rec.BATCH_SIZE_UNIT,
                    rec.MFG_DATE, rec.EXPIRY_DATE, rec.FACILITY_ID, rec.MFG_LINE,
                    rec.QC_STATUS, rec.QC_DATE, rec.QC_ANALYST, rec.YIELD_PERCENT,
                    rec.DEVIATION_FLAG, rec.DEVIATION_ID, rec.RELEASE_DATE,
                    rec.STORAGE_CONDITIONS
                );
                v_inserted := v_inserted + 1;
            ELSE
                v_unchanged := v_unchanged + 1;
            END IF;
        END LOOP;

        p_rows_inserted := v_inserted;
        p_rows_unchanged := v_unchanged;
        COMMIT;

    EXCEPTION
        WHEN OTHERS THEN
            ROLLBACK;
            log_error(v_proc_name, SQLCODE, SQLERRM,
                'SAT_DRUG_MANUFACTURING', NULL, p_batch_id);
            RAISE;
    END LOAD_SAT_DRUG_MANUFACTURING;

END PKG_LOAD_SATELLITES;
/
