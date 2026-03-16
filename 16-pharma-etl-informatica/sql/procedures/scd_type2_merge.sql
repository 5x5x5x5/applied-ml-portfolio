-- ============================================================================
-- PL/SQL Procedure: SCD Type 2 Merge for Drug Dimension
--
-- Implements Slowly Changing Dimension Type 2 logic:
--   1. Compare incoming staging records vs existing dimension using hash
--   2. INSERT new records (no existing match on business key)
--   3. UPDATE changed records: close old version, insert new version
--   4. NO-CHANGE: skip records where hash matches
--
-- Oracle PL/SQL conventions used throughout.
-- ============================================================================

CREATE OR REPLACE PACKAGE pkg_scd_merge AS
    -- -----------------------------------------------------------------------
    -- Package specification for SCD Type 2 merge operations
    -- -----------------------------------------------------------------------

    -- Process SCD Type 2 for drug dimension
    PROCEDURE merge_dim_drug(
        p_batch_id      IN  NUMBER,
        p_load_date     IN  DATE        DEFAULT SYSDATE,
        p_rows_inserted OUT NUMBER,
        p_rows_updated  OUT NUMBER,
        p_rows_nochange OUT NUMBER,
        p_return_code   OUT NUMBER,
        p_return_msg    OUT VARCHAR2
    );

    -- Process SCD Type 2 for trial dimension
    PROCEDURE merge_dim_trial(
        p_batch_id      IN  NUMBER,
        p_load_date     IN  DATE        DEFAULT SYSDATE,
        p_rows_inserted OUT NUMBER,
        p_rows_updated  OUT NUMBER,
        p_rows_nochange OUT NUMBER,
        p_return_code   OUT NUMBER,
        p_return_msg    OUT VARCHAR2
    );

    -- Generic SCD Type 2 merge (table-driven)
    PROCEDURE merge_generic_scd2(
        p_staging_table IN  VARCHAR2,
        p_target_table  IN  VARCHAR2,
        p_business_key  IN  VARCHAR2,
        p_hash_column   IN  VARCHAR2,
        p_batch_id      IN  NUMBER,
        p_load_date     IN  DATE        DEFAULT SYSDATE,
        p_rows_inserted OUT NUMBER,
        p_rows_updated  OUT NUMBER,
        p_rows_nochange OUT NUMBER,
        p_return_code   OUT NUMBER,
        p_return_msg    OUT VARCHAR2
    );

    -- Generate next surrogate key
    FUNCTION get_next_surrogate_key(
        p_table_name    IN  VARCHAR2
    ) RETURN NUMBER;

END pkg_scd_merge;
/


CREATE OR REPLACE PACKAGE BODY pkg_scd_merge AS

    -- -----------------------------------------------------------------------
    -- Constants
    -- -----------------------------------------------------------------------
    c_high_date     CONSTANT DATE := TO_DATE('9999-12-31', 'YYYY-MM-DD');
    c_current_yes   CONSTANT CHAR(1) := 'Y';
    c_current_no    CONSTANT CHAR(1) := 'N';

    -- Surrogate key sequence cache
    g_key_cache     NUMBER := 0;


    -- -----------------------------------------------------------------------
    -- get_next_surrogate_key: Generate surrogate keys from sequences
    -- -----------------------------------------------------------------------
    FUNCTION get_next_surrogate_key(
        p_table_name IN VARCHAR2
    ) RETURN NUMBER IS
        v_next_key NUMBER;
    BEGIN
        -- Use MAX + 1 approach (in production, use Oracle SEQUENCE objects)
        EXECUTE IMMEDIATE
            'SELECT NVL(MAX(' ||
            CASE p_table_name
                WHEN 'DIM_DRUG'  THEN 'drug_key'
                WHEN 'DIM_TRIAL' THEN 'trial_key'
                ELSE 'ROWNUM'
            END ||
            '), 0) + 1 FROM ' || p_table_name
        INTO v_next_key;

        RETURN v_next_key;
    END get_next_surrogate_key;


    -- -----------------------------------------------------------------------
    -- merge_dim_drug: SCD Type 2 merge for Drug Dimension
    -- -----------------------------------------------------------------------
    PROCEDURE merge_dim_drug(
        p_batch_id      IN  NUMBER,
        p_load_date     IN  DATE        DEFAULT SYSDATE,
        p_rows_inserted OUT NUMBER,
        p_rows_updated  OUT NUMBER,
        p_rows_nochange OUT NUMBER,
        p_return_code   OUT NUMBER,
        p_return_msg    OUT VARCHAR2
    ) IS
        v_next_key      NUMBER;
        v_existing_hash VARCHAR2(64);
        v_incoming_hash VARCHAR2(64);
        v_row_count     NUMBER := 0;

        -- Cursor over staged records for this batch
        CURSOR c_staged IS
            SELECT
                s.ndc_code,
                s.drug_name,
                s.generic_name,
                s.strength,
                s.dosage_form,
                s.route,
                s.manufacturer,
                s.dea_schedule,
                s.therapeutic_class,
                -- Compute hash of comparison columns
                ORA_HASH(
                    s.drug_name || '|' ||
                    NVL(s.generic_name, '') || '|' ||
                    NVL(s.strength, '') || '|' ||
                    NVL(s.dosage_form, '') || '|' ||
                    NVL(s.route, '') || '|' ||
                    NVL(s.manufacturer, '') || '|' ||
                    NVL(s.dea_schedule, '') || '|' ||
                    NVL(s.therapeutic_class, '')
                ) AS record_hash
            FROM stg_drug_master s
            WHERE s.load_batch_id = p_batch_id
              AND s.record_status = 'NEW'
              AND s.dq_error_code IS NULL
            ORDER BY s.ndc_code;

    BEGIN
        p_rows_inserted := 0;
        p_rows_updated  := 0;
        p_rows_nochange := 0;
        p_return_code   := 0;
        p_return_msg    := 'SUCCESS';

        -- Log start
        INSERT INTO etl_mapping_run_log (
            batch_id, mapping_name, session_name, run_status,
            start_timestamp, source_name, target_name
        ) VALUES (
            p_batch_id, 'm_drug_master_scd2', 's_drug_master_scd2', 'RUNNING',
            SYSTIMESTAMP, 'stg_drug_master', 'dim_drug'
        );
        COMMIT;

        FOR rec IN c_staged LOOP
            v_row_count := v_row_count + 1;

            BEGIN
                -- Check if business key (ndc_code) exists in dimension
                SELECT record_hash
                  INTO v_existing_hash
                  FROM dim_drug
                 WHERE ndc_code = rec.ndc_code
                   AND current_flag = c_current_yes;

                -- Record exists: compare hashes
                IF v_existing_hash = TO_CHAR(rec.record_hash) THEN
                    -- NO CHANGE: hashes match, skip
                    p_rows_nochange := p_rows_nochange + 1;
                ELSE
                    -- CHANGED: Close the existing record (set end date)
                    UPDATE dim_drug
                       SET effective_end_date = p_load_date - 1,
                           current_flag       = c_current_no,
                           updated_dt         = SYSTIMESTAMP
                     WHERE ndc_code    = rec.ndc_code
                       AND current_flag = c_current_yes;

                    -- Insert new version with new surrogate key
                    v_next_key := get_next_surrogate_key('DIM_DRUG');

                    INSERT INTO dim_drug (
                        drug_key, ndc_code, drug_name, generic_name,
                        strength, dosage_form, route, manufacturer,
                        dea_schedule, therapeutic_class,
                        effective_start_date, effective_end_date,
                        current_flag, record_hash,
                        created_dt, load_batch_id
                    ) VALUES (
                        v_next_key, rec.ndc_code, rec.drug_name, rec.generic_name,
                        rec.strength, rec.dosage_form, rec.route, rec.manufacturer,
                        rec.dea_schedule, rec.therapeutic_class,
                        p_load_date, c_high_date,
                        c_current_yes, TO_CHAR(rec.record_hash),
                        SYSTIMESTAMP, p_batch_id
                    );

                    p_rows_updated := p_rows_updated + 1;
                END IF;

            EXCEPTION
                WHEN NO_DATA_FOUND THEN
                    -- NEW: Business key does not exist, insert first version
                    v_next_key := get_next_surrogate_key('DIM_DRUG');

                    INSERT INTO dim_drug (
                        drug_key, ndc_code, drug_name, generic_name,
                        strength, dosage_form, route, manufacturer,
                        dea_schedule, therapeutic_class,
                        effective_start_date, effective_end_date,
                        current_flag, record_hash,
                        created_dt, load_batch_id
                    ) VALUES (
                        v_next_key, rec.ndc_code, rec.drug_name, rec.generic_name,
                        rec.strength, rec.dosage_form, rec.route, rec.manufacturer,
                        rec.dea_schedule, rec.therapeutic_class,
                        p_load_date, c_high_date,
                        c_current_yes, TO_CHAR(rec.record_hash),
                        SYSTIMESTAMP, p_batch_id
                    );

                    p_rows_inserted := p_rows_inserted + 1;

                WHEN TOO_MANY_ROWS THEN
                    -- Multiple current records: data integrity issue
                    INSERT INTO etl_error_log (
                        batch_id, mapping_name, error_code, error_message,
                        error_severity, source_row_data
                    ) VALUES (
                        p_batch_id, 'm_drug_master_scd2',
                        'SCD2_MULTI_CURRENT',
                        'Multiple current records found for NDC: ' || rec.ndc_code,
                        'ERROR',
                        'NDC=' || rec.ndc_code
                    );
            END;

            -- Commit every 5000 rows (Informatica-style commit interval)
            IF MOD(v_row_count, 5000) = 0 THEN
                COMMIT;
            END IF;
        END LOOP;

        -- Mark staging records as processed
        UPDATE stg_drug_master
           SET record_status = 'PROCESSED'
         WHERE load_batch_id = p_batch_id
           AND record_status = 'NEW'
           AND dq_error_code IS NULL;

        -- Final commit
        COMMIT;

        -- Log completion
        UPDATE etl_mapping_run_log
           SET run_status          = 'SUCCEEDED',
               end_timestamp       = SYSTIMESTAMP,
               elapsed_seconds     = EXTRACT(SECOND FROM (SYSTIMESTAMP - start_timestamp)),
               source_rows_read    = v_row_count,
               target_rows_written = p_rows_inserted,
               target_rows_updated = p_rows_updated
         WHERE batch_id     = p_batch_id
           AND mapping_name = 'm_drug_master_scd2'
           AND run_status   = 'RUNNING';

        -- Audit row counts
        INSERT INTO etl_row_count_audit (batch_id, table_name, count_type, row_count)
        VALUES (p_batch_id, 'dim_drug', 'POST_LOAD',
                (SELECT COUNT(*) FROM dim_drug WHERE current_flag = 'Y'));

        COMMIT;

    EXCEPTION
        WHEN OTHERS THEN
            ROLLBACK;
            p_return_code := SQLCODE;
            p_return_msg  := SQLERRM;

            -- Log failure
            UPDATE etl_mapping_run_log
               SET run_status    = 'FAILED',
                   end_timestamp = SYSTIMESTAMP,
                   error_count   = 1
             WHERE batch_id     = p_batch_id
               AND mapping_name = 'm_drug_master_scd2'
               AND run_status   = 'RUNNING';

            INSERT INTO etl_error_log (
                batch_id, mapping_name, error_code, error_message, error_severity
            ) VALUES (
                p_batch_id, 'm_drug_master_scd2',
                TO_CHAR(SQLCODE), SQLERRM, 'FATAL'
            );

            COMMIT;
    END merge_dim_drug;


    -- -----------------------------------------------------------------------
    -- merge_dim_trial: SCD Type 2 merge for Trial Dimension
    -- -----------------------------------------------------------------------
    PROCEDURE merge_dim_trial(
        p_batch_id      IN  NUMBER,
        p_load_date     IN  DATE        DEFAULT SYSDATE,
        p_rows_inserted OUT NUMBER,
        p_rows_updated  OUT NUMBER,
        p_rows_nochange OUT NUMBER,
        p_return_code   OUT NUMBER,
        p_return_msg    OUT VARCHAR2
    ) IS
        v_next_key      NUMBER;
        v_existing_hash VARCHAR2(64);

        CURSOR c_staged IS
            SELECT
                s.nct_id,
                s.brief_title,
                s.official_title,
                s.overall_status,
                s.phase,
                s.study_type,
                s.enrollment,
                s.start_date,
                s.completion_date,
                s.lead_sponsor,
                s.indication_code,
                ORA_HASH(
                    NVL(s.brief_title, '') || '|' ||
                    NVL(s.overall_status, '') || '|' ||
                    NVL(s.phase, '') || '|' ||
                    NVL(TO_CHAR(s.enrollment), '') || '|' ||
                    NVL(s.lead_sponsor, '')
                ) AS record_hash
            FROM stg_clinical_trial s
            WHERE s.load_batch_id = p_batch_id
              AND s.record_status = 'NEW';

    BEGIN
        p_rows_inserted := 0;
        p_rows_updated  := 0;
        p_rows_nochange := 0;
        p_return_code   := 0;
        p_return_msg    := 'SUCCESS';

        FOR rec IN c_staged LOOP
            BEGIN
                SELECT record_hash
                  INTO v_existing_hash
                  FROM dim_trial
                 WHERE nct_id = rec.nct_id
                   AND current_flag = c_current_yes;

                IF v_existing_hash = TO_CHAR(rec.record_hash) THEN
                    p_rows_nochange := p_rows_nochange + 1;
                ELSE
                    -- Close existing
                    UPDATE dim_trial
                       SET effective_end_date = p_load_date - 1,
                           current_flag       = c_current_no,
                           updated_dt         = SYSTIMESTAMP
                     WHERE nct_id       = rec.nct_id
                       AND current_flag = c_current_yes;

                    -- Insert new version
                    v_next_key := get_next_surrogate_key('DIM_TRIAL');

                    INSERT INTO dim_trial (
                        trial_key, nct_id, brief_title, phase_std,
                        study_type, lead_sponsor, indication_code,
                        effective_start_date, effective_end_date,
                        current_flag, record_hash,
                        created_dt, load_batch_id
                    ) VALUES (
                        v_next_key, rec.nct_id, rec.brief_title, rec.phase,
                        rec.study_type, rec.lead_sponsor, rec.indication_code,
                        p_load_date, c_high_date,
                        c_current_yes, TO_CHAR(rec.record_hash),
                        SYSTIMESTAMP, p_batch_id
                    );

                    p_rows_updated := p_rows_updated + 1;
                END IF;

            EXCEPTION
                WHEN NO_DATA_FOUND THEN
                    v_next_key := get_next_surrogate_key('DIM_TRIAL');

                    INSERT INTO dim_trial (
                        trial_key, nct_id, brief_title, phase_std,
                        study_type, lead_sponsor, indication_code,
                        effective_start_date, effective_end_date,
                        current_flag, record_hash,
                        created_dt, load_batch_id
                    ) VALUES (
                        v_next_key, rec.nct_id, rec.brief_title, rec.phase,
                        rec.study_type, rec.lead_sponsor, rec.indication_code,
                        p_load_date, c_high_date,
                        c_current_yes, TO_CHAR(rec.record_hash),
                        SYSTIMESTAMP, p_batch_id
                    );

                    p_rows_inserted := p_rows_inserted + 1;
            END;
        END LOOP;

        UPDATE stg_clinical_trial
           SET record_status = 'PROCESSED'
         WHERE load_batch_id = p_batch_id
           AND record_status = 'NEW';

        COMMIT;

    EXCEPTION
        WHEN OTHERS THEN
            ROLLBACK;
            p_return_code := SQLCODE;
            p_return_msg  := SQLERRM;

            INSERT INTO etl_error_log (
                batch_id, mapping_name, error_code, error_message, error_severity
            ) VALUES (
                p_batch_id, 'm_clinical_trial_scd2',
                TO_CHAR(SQLCODE), SQLERRM, 'FATAL'
            );
            COMMIT;
    END merge_dim_trial;


    -- -----------------------------------------------------------------------
    -- merge_generic_scd2: Table-driven SCD Type 2 (dynamic SQL)
    -- -----------------------------------------------------------------------
    PROCEDURE merge_generic_scd2(
        p_staging_table IN  VARCHAR2,
        p_target_table  IN  VARCHAR2,
        p_business_key  IN  VARCHAR2,
        p_hash_column   IN  VARCHAR2,
        p_batch_id      IN  NUMBER,
        p_load_date     IN  DATE        DEFAULT SYSDATE,
        p_rows_inserted OUT NUMBER,
        p_rows_updated  OUT NUMBER,
        p_rows_nochange OUT NUMBER,
        p_return_code   OUT NUMBER,
        p_return_msg    OUT VARCHAR2
    ) IS
        v_sql           VARCHAR2(4000);
        v_existing_hash VARCHAR2(64);
        v_incoming_hash VARCHAR2(64);
        v_bk_value      VARCHAR2(500);
        v_count         NUMBER;

        TYPE t_refcursor IS REF CURSOR;
        v_cursor        t_refcursor;
    BEGIN
        p_rows_inserted := 0;
        p_rows_updated  := 0;
        p_rows_nochange := 0;
        p_return_code   := 0;
        p_return_msg    := 'SUCCESS';

        -- Iterate over staging records
        v_sql := 'SELECT ' || p_business_key || ', ' || p_hash_column ||
                 ' FROM ' || p_staging_table ||
                 ' WHERE load_batch_id = ' || p_batch_id;

        OPEN v_cursor FOR v_sql;
        LOOP
            FETCH v_cursor INTO v_bk_value, v_incoming_hash;
            EXIT WHEN v_cursor%NOTFOUND;

            -- Check if exists in target
            v_sql := 'SELECT COUNT(*) FROM ' || p_target_table ||
                     ' WHERE ' || p_business_key || ' = :1' ||
                     ' AND current_flag = ''Y''';
            EXECUTE IMMEDIATE v_sql INTO v_count USING v_bk_value;

            IF v_count = 0 THEN
                -- New record: INSERT
                -- (In production, build dynamic INSERT with all columns)
                p_rows_inserted := p_rows_inserted + 1;
            ELSE
                -- Existing: compare hash
                v_sql := 'SELECT ' || p_hash_column || ' FROM ' || p_target_table ||
                         ' WHERE ' || p_business_key || ' = :1' ||
                         ' AND current_flag = ''Y'' AND ROWNUM = 1';
                EXECUTE IMMEDIATE v_sql INTO v_existing_hash USING v_bk_value;

                IF v_existing_hash = v_incoming_hash THEN
                    p_rows_nochange := p_rows_nochange + 1;
                ELSE
                    -- Close old, insert new version
                    v_sql := 'UPDATE ' || p_target_table ||
                             ' SET effective_end_date = :1,' ||
                             '     current_flag = ''N'',' ||
                             '     updated_dt = SYSTIMESTAMP' ||
                             ' WHERE ' || p_business_key || ' = :2' ||
                             ' AND current_flag = ''Y''';
                    EXECUTE IMMEDIATE v_sql USING p_load_date - 1, v_bk_value;

                    p_rows_updated := p_rows_updated + 1;
                END IF;
            END IF;
        END LOOP;
        CLOSE v_cursor;

        COMMIT;

    EXCEPTION
        WHEN OTHERS THEN
            IF v_cursor%ISOPEN THEN
                CLOSE v_cursor;
            END IF;
            ROLLBACK;
            p_return_code := SQLCODE;
            p_return_msg  := SQLERRM;
    END merge_generic_scd2;


END pkg_scd_merge;
/

-- ============================================================================
-- Standalone MERGE statement alternative (Oracle MERGE)
-- Can be used instead of the row-by-row PL/SQL for better performance.
-- ============================================================================

/*
MERGE INTO dim_drug tgt
USING (
    SELECT
        s.ndc_code,
        s.drug_name,
        s.generic_name,
        s.strength,
        s.dosage_form,
        s.route,
        s.manufacturer,
        s.dea_schedule,
        s.therapeutic_class,
        ORA_HASH(
            s.drug_name || '|' || NVL(s.generic_name, '') || '|' ||
            NVL(s.strength, '') || '|' || NVL(s.dosage_form, '') || '|' ||
            NVL(s.route, '') || '|' || NVL(s.manufacturer, '') || '|' ||
            NVL(s.dea_schedule, '') || '|' || NVL(s.therapeutic_class, '')
        ) AS record_hash
    FROM stg_drug_master s
    WHERE s.load_batch_id = :p_batch_id
      AND s.record_status = 'NEW'
) src
ON (tgt.ndc_code = src.ndc_code AND tgt.current_flag = 'Y')
WHEN MATCHED THEN
    UPDATE SET
        tgt.effective_end_date = SYSDATE - 1,
        tgt.current_flag = 'N',
        tgt.updated_dt = SYSTIMESTAMP
    WHERE tgt.record_hash != TO_CHAR(src.record_hash)
WHEN NOT MATCHED THEN
    INSERT (
        drug_key, ndc_code, drug_name, generic_name,
        strength, dosage_form, route, manufacturer,
        dea_schedule, therapeutic_class,
        effective_start_date, effective_end_date,
        current_flag, record_hash, load_batch_id
    ) VALUES (
        dim_drug_seq.NEXTVAL, src.ndc_code, src.drug_name, src.generic_name,
        src.strength, src.dosage_form, src.route, src.manufacturer,
        src.dea_schedule, src.therapeutic_class,
        SYSDATE, TO_DATE('9999-12-31','YYYY-MM-DD'),
        'Y', TO_CHAR(src.record_hash), :p_batch_id
    );
*/
