/*******************************************************************************
 * PharmaDataVault - PL/SQL Procedure: Load HUB_DRUG
 *
 * Loads unique drug business keys from clean staging into HUB_DRUG.
 * Follows Data Vault 2.0 hub loading pattern:
 *   - Check for existing business keys (insert-only, no updates)
 *   - Hash-based surrogate key generation
 *   - Full audit trail via ETL_BATCH_CONTROL and ETL_ERROR_LOG
 *   - Cursor-based processing with bulk operations for performance
 *
 * Oracle PL/SQL compatible
 ******************************************************************************/

CREATE OR REPLACE PACKAGE PKG_LOAD_HUB_DRUG
AS
    /*-----------------------------------------------------------------------
     * Package specification for HUB_DRUG loading operations.
     * Provides both single-batch and incremental load capabilities.
     *-----------------------------------------------------------------------*/

    -- Load hub from a specific staging batch
    PROCEDURE LOAD_FROM_BATCH (
        p_batch_id          IN NUMBER,
        p_rows_inserted     OUT NUMBER,
        p_rows_skipped      OUT NUMBER
    );

    -- Load hub from all unprocessed staging data
    PROCEDURE LOAD_INCREMENTAL (
        p_rows_inserted     OUT NUMBER,
        p_rows_skipped      OUT NUMBER
    );

    -- Verify hub integrity after load
    PROCEDURE VERIFY_HUB_INTEGRITY (
        p_batch_id          IN NUMBER,
        p_is_valid          OUT BOOLEAN,
        p_orphan_count      OUT NUMBER
    );

END PKG_LOAD_HUB_DRUG;
/

CREATE OR REPLACE PACKAGE BODY PKG_LOAD_HUB_DRUG
AS
    -- Package-level constants
    gc_package_name     CONSTANT VARCHAR2(100) := 'PKG_LOAD_HUB_DRUG';
    gc_bulk_limit       CONSTANT PLS_INTEGER := 5000;

    -- Package-level types for bulk collect
    TYPE t_hash_key_tab IS TABLE OF RAW(16);
    TYPE t_ndc_tab      IS TABLE OF VARCHAR2(13);
    TYPE t_source_tab   IS TABLE OF VARCHAR2(100);

    /*-----------------------------------------------------------------------
     * Internal procedure: Log an error to the ETL_ERROR_LOG table.
     *-----------------------------------------------------------------------*/
    PROCEDURE log_error (
        p_procedure_name    IN VARCHAR2,
        p_error_code        IN NUMBER,
        p_error_message     IN VARCHAR2,
        p_record_key        IN VARCHAR2 DEFAULT NULL,
        p_batch_id          IN NUMBER DEFAULT NULL
    )
    AS
        PRAGMA AUTONOMOUS_TRANSACTION;
    BEGIN
        INSERT INTO ETL_ERROR_LOG (
            PROCEDURE_NAME, ERROR_CODE, ERROR_MESSAGE,
            SOURCE_TABLE, TARGET_TABLE, RECORD_KEY, BATCH_ID
        ) VALUES (
            p_procedure_name, p_error_code, p_error_message,
            'STG_DRUG_CLEAN', 'HUB_DRUG', p_record_key, p_batch_id
        );
        COMMIT;
    END log_error;

    /*-----------------------------------------------------------------------
     * Internal procedure: Log ETL progress for monitoring.
     *-----------------------------------------------------------------------*/
    PROCEDURE log_progress (
        p_batch_id      IN NUMBER,
        p_message       IN VARCHAR2
    )
    AS
        PRAGMA AUTONOMOUS_TRANSACTION;
    BEGIN
        INSERT INTO ETL_ERROR_LOG (
            PROCEDURE_NAME, ERROR_CODE, ERROR_MESSAGE, BATCH_ID
        ) VALUES (
            gc_package_name || '.PROGRESS', 0, p_message, p_batch_id
        );
        COMMIT;
    END log_progress;

    /*-----------------------------------------------------------------------
     * LOAD_FROM_BATCH
     *
     * Loads new drug business keys from a specific staging batch into
     * HUB_DRUG. Uses anti-join to skip already-existing business keys.
     * Processes in chunks of gc_bulk_limit for memory efficiency.
     *
     * Parameters:
     *   p_batch_id      - The staging batch to process
     *   p_rows_inserted - OUT: count of new hub records inserted
     *   p_rows_skipped  - OUT: count of already-existing records skipped
     *-----------------------------------------------------------------------*/
    PROCEDURE LOAD_FROM_BATCH (
        p_batch_id          IN NUMBER,
        p_rows_inserted     OUT NUMBER,
        p_rows_skipped      OUT NUMBER
    )
    AS
        v_proc_name     CONSTANT VARCHAR2(100) := gc_package_name || '.LOAD_FROM_BATCH';
        v_inserted      NUMBER := 0;
        v_skipped       NUMBER := 0;
        v_total_stg     NUMBER;
        v_load_date     TIMESTAMP := SYSTIMESTAMP;

        -- Cursor: select distinct new drug NDCs not already in hub
        CURSOR c_new_drugs IS
            SELECT DISTINCT
                sc.DRUG_HASH_KEY,
                sc.DRUG_NDC,
                sc.RECORD_SOURCE
            FROM STG_DRUG_CLEAN sc
            WHERE sc.BATCH_ID = p_batch_id
              AND NOT EXISTS (
                  SELECT 1
                  FROM HUB_DRUG hd
                  WHERE hd.DRUG_KEY = sc.DRUG_HASH_KEY
              )
            ORDER BY sc.DRUG_NDC;

        l_hash_keys     t_hash_key_tab;
        l_ndcs          t_ndc_tab;
        l_sources       t_source_tab;
        l_errors        NUMBER;

    BEGIN
        log_progress(p_batch_id, 'Starting HUB_DRUG load for batch ' || p_batch_id);

        -- Count total staged records for this batch
        SELECT COUNT(DISTINCT DRUG_NDC)
        INTO v_total_stg
        FROM STG_DRUG_CLEAN
        WHERE BATCH_ID = p_batch_id;

        log_progress(p_batch_id,
            'Found ' || v_total_stg || ' distinct NDCs in batch ' || p_batch_id);

        -- Bulk insert loop
        OPEN c_new_drugs;
        LOOP
            FETCH c_new_drugs BULK COLLECT
                INTO l_hash_keys, l_ndcs, l_sources
                LIMIT gc_bulk_limit;

            EXIT WHEN l_hash_keys.COUNT = 0;

            -- Use FORALL for bulk insert with SAVE EXCEPTIONS
            BEGIN
                FORALL i IN 1..l_hash_keys.COUNT SAVE EXCEPTIONS
                    INSERT INTO HUB_DRUG (
                        DRUG_KEY, DRUG_NDC, LOAD_DATE, RECORD_SOURCE
                    ) VALUES (
                        l_hash_keys(i),
                        l_ndcs(i),
                        v_load_date,
                        l_sources(i)
                    );
            EXCEPTION
                WHEN OTHERS THEN
                    -- Handle bulk insert exceptions
                    l_errors := SQL%BULK_EXCEPTIONS.COUNT;
                    FOR j IN 1..l_errors LOOP
                        log_error(
                            p_procedure_name => v_proc_name,
                            p_error_code     => SQL%BULK_EXCEPTIONS(j).ERROR_CODE,
                            p_error_message  => 'Bulk insert error at index '
                                                || SQL%BULK_EXCEPTIONS(j).ERROR_INDEX,
                            p_record_key     => 'NDC='
                                                || l_ndcs(SQL%BULK_EXCEPTIONS(j).ERROR_INDEX),
                            p_batch_id       => p_batch_id
                        );
                    END LOOP;
            END;

            v_inserted := v_inserted + SQL%ROWCOUNT;

            -- Progress logging every gc_bulk_limit rows
            log_progress(p_batch_id,
                'Inserted ' || v_inserted || ' hub records so far');

        END LOOP;
        CLOSE c_new_drugs;

        v_skipped := v_total_stg - v_inserted;

        -- Update batch control with loaded count
        UPDATE ETL_BATCH_CONTROL
        SET ROWS_LOADED = v_inserted,
            STATUS = 'SUCCESS',
            END_TIMESTAMP = SYSTIMESTAMP
        WHERE BATCH_ID = p_batch_id;

        p_rows_inserted := v_inserted;
        p_rows_skipped := v_skipped;

        COMMIT;

        log_progress(p_batch_id,
            'HUB_DRUG load complete: ' || v_inserted || ' inserted, '
            || v_skipped || ' skipped (already existed)');

    EXCEPTION
        WHEN OTHERS THEN
            IF c_new_drugs%ISOPEN THEN
                CLOSE c_new_drugs;
            END IF;
            ROLLBACK;

            log_error(
                p_procedure_name => v_proc_name,
                p_error_code     => SQLCODE,
                p_error_message  => SQLERRM || ' | ' || DBMS_UTILITY.FORMAT_ERROR_BACKTRACE,
                p_batch_id       => p_batch_id
            );

            UPDATE ETL_BATCH_CONTROL
            SET STATUS = 'FAILED',
                END_TIMESTAMP = SYSTIMESTAMP,
                ERROR_MESSAGE = SQLERRM
            WHERE BATCH_ID = p_batch_id;
            COMMIT;

            RAISE;
    END LOAD_FROM_BATCH;

    /*-----------------------------------------------------------------------
     * LOAD_INCREMENTAL
     *
     * Loads all unprocessed staging batches into HUB_DRUG.
     * Identifies batches where ROWS_LOADED = 0 and STATUS = 'RUNNING'.
     *-----------------------------------------------------------------------*/
    PROCEDURE LOAD_INCREMENTAL (
        p_rows_inserted     OUT NUMBER,
        p_rows_skipped      OUT NUMBER
    )
    AS
        v_proc_name     CONSTANT VARCHAR2(100) := gc_package_name || '.LOAD_INCREMENTAL';
        v_batch_inserted NUMBER;
        v_batch_skipped  NUMBER;
        v_total_inserted NUMBER := 0;
        v_total_skipped  NUMBER := 0;

        CURSOR c_pending_batches IS
            SELECT BATCH_ID
            FROM ETL_BATCH_CONTROL
            WHERE BATCH_TYPE = 'DRUG'
              AND STATUS IN ('RUNNING', 'PARTIAL')
              AND ROWS_LOADED = 0
            ORDER BY BATCH_ID;

    BEGIN
        FOR batch_rec IN c_pending_batches LOOP
            BEGIN
                LOAD_FROM_BATCH(
                    p_batch_id      => batch_rec.BATCH_ID,
                    p_rows_inserted => v_batch_inserted,
                    p_rows_skipped  => v_batch_skipped
                );
                v_total_inserted := v_total_inserted + v_batch_inserted;
                v_total_skipped  := v_total_skipped + v_batch_skipped;
            EXCEPTION
                WHEN OTHERS THEN
                    -- Log but continue with next batch
                    log_error(
                        p_procedure_name => v_proc_name,
                        p_error_code     => SQLCODE,
                        p_error_message  => 'Batch ' || batch_rec.BATCH_ID
                                            || ' failed: ' || SQLERRM,
                        p_batch_id       => batch_rec.BATCH_ID
                    );
            END;
        END LOOP;

        p_rows_inserted := v_total_inserted;
        p_rows_skipped  := v_total_skipped;

    END LOAD_INCREMENTAL;

    /*-----------------------------------------------------------------------
     * VERIFY_HUB_INTEGRITY
     *
     * Post-load verification to ensure all staged business keys
     * now exist in HUB_DRUG. Identifies orphan staging records.
     *-----------------------------------------------------------------------*/
    PROCEDURE VERIFY_HUB_INTEGRITY (
        p_batch_id          IN NUMBER,
        p_is_valid          OUT BOOLEAN,
        p_orphan_count      OUT NUMBER
    )
    AS
        v_proc_name     CONSTANT VARCHAR2(100) := gc_package_name || '.VERIFY_HUB_INTEGRITY';
    BEGIN
        -- Check for staged NDCs that don't exist in hub
        SELECT COUNT(*)
        INTO p_orphan_count
        FROM STG_DRUG_CLEAN sc
        WHERE sc.BATCH_ID = p_batch_id
          AND NOT EXISTS (
              SELECT 1
              FROM HUB_DRUG hd
              WHERE hd.DRUG_KEY = sc.DRUG_HASH_KEY
          );

        IF p_orphan_count = 0 THEN
            p_is_valid := TRUE;
            log_progress(p_batch_id,
                'HUB_DRUG integrity check PASSED for batch ' || p_batch_id);
        ELSE
            p_is_valid := FALSE;
            log_error(
                p_procedure_name => v_proc_name,
                p_error_code     => -20020,
                p_error_message  => p_orphan_count
                                    || ' orphan NDCs found in staging after hub load',
                p_batch_id       => p_batch_id
            );
        END IF;

    EXCEPTION
        WHEN OTHERS THEN
            p_is_valid := FALSE;
            p_orphan_count := -1;
            log_error(
                p_procedure_name => v_proc_name,
                p_error_code     => SQLCODE,
                p_error_message  => SQLERRM,
                p_batch_id       => p_batch_id
            );
    END VERIFY_HUB_INTEGRITY;

END PKG_LOAD_HUB_DRUG;
/
