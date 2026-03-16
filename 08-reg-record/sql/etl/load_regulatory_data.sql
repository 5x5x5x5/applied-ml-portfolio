--------------------------------------------------------------------------------
-- RegRecord - ETL Procedures for Regulatory Data Loading
--
-- PL/SQL ETL pipeline:
--   1. Load regulatory submission data from external staging tables
--   2. Transform and standardize agency codes
--   3. MERGE (upsert) with existing records
--   4. Validate business rules
--   5. Generate load summary report
--
-- Designed for Control-M scheduling in UNIX environment
--------------------------------------------------------------------------------

-- ============================================================================
-- STAGING TABLES (populated by external data feeds)
-- ============================================================================

CREATE TABLE STG_REGULATORY_FEED (
    feed_id             NUMBER          GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    batch_id            VARCHAR2(50)    NOT NULL,
    source_system       VARCHAR2(50)    NOT NULL,
    drug_name           VARCHAR2(200),
    drug_ndc            VARCHAR2(50),
    submission_type     VARCHAR2(50),
    submission_status   VARCHAR2(50),
    agency_code         VARCHAR2(50),
    agency_name         VARCHAR2(200),
    tracking_number     VARCHAR2(100),
    submitted_date      VARCHAR2(30),
    approval_date       VARCHAR2(30),
    approval_type       VARCHAR2(50),
    conditions_text     CLOB,
    priority            VARCHAR2(30),
    assigned_to         VARCHAR2(100),
    raw_json            CLOB,
    load_timestamp      TIMESTAMP       DEFAULT SYSTIMESTAMP,
    process_status      VARCHAR2(20)    DEFAULT 'NEW',
    error_message       VARCHAR2(4000),
    CONSTRAINT chk_stg_status CHECK (
        process_status IN ('NEW', 'VALIDATED', 'TRANSFORMED', 'LOADED', 'ERROR', 'SKIPPED')
    )
);

CREATE INDEX idx_stg_batch ON STG_REGULATORY_FEED(batch_id);
CREATE INDEX idx_stg_status ON STG_REGULATORY_FEED(process_status);

-- ETL Load Log table
CREATE TABLE ETL_LOAD_LOG (
    log_id              NUMBER          GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    batch_id            VARCHAR2(50)    NOT NULL,
    step_name           VARCHAR2(100)   NOT NULL,
    step_status         VARCHAR2(20)    NOT NULL,
    records_processed   NUMBER          DEFAULT 0,
    records_success     NUMBER          DEFAULT 0,
    records_error       NUMBER          DEFAULT 0,
    records_skipped     NUMBER          DEFAULT 0,
    start_time          TIMESTAMP       NOT NULL,
    end_time            TIMESTAMP,
    duration_seconds    NUMBER,
    error_details       CLOB,
    run_by              VARCHAR2(100)   DEFAULT USER
);

-- ============================================================================
-- AGENCY CODE MAPPING TABLE (for standardization)
-- ============================================================================

CREATE TABLE AGENCY_CODE_MAPPING (
    source_system       VARCHAR2(50)    NOT NULL,
    source_code         VARCHAR2(50)    NOT NULL,
    standard_code       VARCHAR2(20)    NOT NULL,
    effective_from      DATE            DEFAULT SYSDATE,
    effective_to        DATE,
    CONSTRAINT pk_agency_mapping PRIMARY KEY (source_system, source_code),
    CONSTRAINT fk_agency_std FOREIGN KEY (standard_code) REFERENCES AGENCY(agency_code)
);

-- Seed some common mappings
INSERT INTO AGENCY_CODE_MAPPING VALUES ('EXTERNAL_A', 'US_FDA', 'FDA', SYSDATE, NULL);
INSERT INTO AGENCY_CODE_MAPPING VALUES ('EXTERNAL_A', 'EU_EMA', 'EMA', SYSDATE, NULL);
INSERT INTO AGENCY_CODE_MAPPING VALUES ('EXTERNAL_A', 'JP_PMDA', 'PMDA', SYSDATE, NULL);
INSERT INTO AGENCY_CODE_MAPPING VALUES ('EXTERNAL_A', 'CN_NMPA', 'NMPA', SYSDATE, NULL);
INSERT INTO AGENCY_CODE_MAPPING VALUES ('EXTERNAL_A', 'UK_MHRA', 'MHRA', SYSDATE, NULL);
INSERT INTO AGENCY_CODE_MAPPING VALUES ('EXTERNAL_B', 'FDA', 'FDA', SYSDATE, NULL);
INSERT INTO AGENCY_CODE_MAPPING VALUES ('EXTERNAL_B', 'EMEA', 'EMA', SYSDATE, NULL);
INSERT INTO AGENCY_CODE_MAPPING VALUES ('EXTERNAL_B', 'PMDA', 'PMDA', SYSDATE, NULL);
INSERT INTO AGENCY_CODE_MAPPING VALUES ('EXTERNAL_B', 'HEALTH_CANADA', 'HC', SYSDATE, NULL);
INSERT INTO AGENCY_CODE_MAPPING VALUES ('EXTERNAL_B', 'TGA_AU', 'TGA', SYSDATE, NULL);
COMMIT;

-- ============================================================================
-- ETL PACKAGE
-- ============================================================================

CREATE OR REPLACE PACKAGE PKG_ETL_REGULATORY AS

    -- Main ETL entry point
    PROCEDURE run_full_etl(
        p_batch_id      IN  VARCHAR2,
        p_run_by        IN  VARCHAR2 DEFAULT USER,
        p_success       OUT BOOLEAN,
        p_summary       OUT VARCHAR2
    );

    -- Step 1: Validate raw staging data
    PROCEDURE validate_staging_data(
        p_batch_id      IN  VARCHAR2,
        p_validated     OUT NUMBER,
        p_errors        OUT NUMBER
    );

    -- Step 2: Transform and standardize
    PROCEDURE transform_agency_codes(
        p_batch_id      IN  VARCHAR2,
        p_transformed   OUT NUMBER,
        p_errors        OUT NUMBER
    );

    -- Step 3: Merge into target tables
    PROCEDURE merge_submissions(
        p_batch_id      IN  VARCHAR2,
        p_inserted      OUT NUMBER,
        p_updated       OUT NUMBER,
        p_errors        OUT NUMBER
    );

    -- Step 4: Post-load validation
    PROCEDURE validate_business_rules(
        p_batch_id      IN  VARCHAR2,
        p_valid         OUT NUMBER,
        p_invalid       OUT NUMBER
    );

    -- Step 5: Generate load summary
    PROCEDURE generate_load_summary(
        p_batch_id      IN  VARCHAR2,
        p_cursor        OUT SYS_REFCURSOR
    );

    -- Utility: Log ETL step
    PROCEDURE log_step(
        p_batch_id      IN  VARCHAR2,
        p_step_name     IN  VARCHAR2,
        p_status        IN  VARCHAR2,
        p_processed     IN  NUMBER DEFAULT 0,
        p_success       IN  NUMBER DEFAULT 0,
        p_errors        IN  NUMBER DEFAULT 0,
        p_skipped       IN  NUMBER DEFAULT 0,
        p_start_time    IN  TIMESTAMP,
        p_error_details IN  CLOB DEFAULT NULL
    );

END PKG_ETL_REGULATORY;
/

CREATE OR REPLACE PACKAGE BODY PKG_ETL_REGULATORY AS

    -- ========================================================================
    -- UTILITY: Log ETL step
    -- ========================================================================
    PROCEDURE log_step(
        p_batch_id      IN  VARCHAR2,
        p_step_name     IN  VARCHAR2,
        p_status        IN  VARCHAR2,
        p_processed     IN  NUMBER DEFAULT 0,
        p_success       IN  NUMBER DEFAULT 0,
        p_errors        IN  NUMBER DEFAULT 0,
        p_skipped       IN  NUMBER DEFAULT 0,
        p_start_time    IN  TIMESTAMP,
        p_error_details IN  CLOB DEFAULT NULL
    ) IS
        PRAGMA AUTONOMOUS_TRANSACTION;
        v_duration  NUMBER;
    BEGIN
        v_duration := EXTRACT(SECOND FROM (SYSTIMESTAMP - p_start_time))
                    + EXTRACT(MINUTE FROM (SYSTIMESTAMP - p_start_time)) * 60
                    + EXTRACT(HOUR FROM (SYSTIMESTAMP - p_start_time)) * 3600;

        INSERT INTO ETL_LOAD_LOG (
            batch_id, step_name, step_status,
            records_processed, records_success, records_error, records_skipped,
            start_time, end_time, duration_seconds, error_details
        ) VALUES (
            p_batch_id, p_step_name, p_status,
            p_processed, p_success, p_errors, p_skipped,
            p_start_time, SYSTIMESTAMP, v_duration, p_error_details
        );
        COMMIT;
    END log_step;

    -- ========================================================================
    -- STEP 1: Validate staging data
    -- ========================================================================
    PROCEDURE validate_staging_data(
        p_batch_id      IN  VARCHAR2,
        p_validated     OUT NUMBER,
        p_errors        OUT NUMBER
    ) IS
        v_start     TIMESTAMP := SYSTIMESTAMP;
        v_valid     NUMBER := 0;
        v_err       NUMBER := 0;
        v_err_msg   VARCHAR2(4000);

        CURSOR c_staging IS
            SELECT feed_id, drug_name, drug_ndc, submission_type,
                   agency_code, tracking_number, submitted_date
            FROM STG_REGULATORY_FEED
            WHERE batch_id = p_batch_id
              AND process_status = 'NEW'
            FOR UPDATE;
    BEGIN
        DBMS_OUTPUT.PUT_LINE('Step 1: Validating staging data for batch ' || p_batch_id);

        FOR rec IN c_staging LOOP
            v_err_msg := NULL;

            -- Validate required fields
            IF rec.drug_name IS NULL AND rec.drug_ndc IS NULL THEN
                v_err_msg := 'Drug name or NDC is required';
            ELSIF rec.submission_type IS NULL THEN
                v_err_msg := 'Submission type is required';
            ELSIF rec.agency_code IS NULL THEN
                v_err_msg := 'Agency code is required';
            END IF;

            -- Validate submission type
            IF v_err_msg IS NULL AND rec.submission_type NOT IN (
                'NDA', 'ANDA', 'BLA', 'IND', 'sNDA', 'sBLA',
                'MAA', 'TYPE_II_VARIATION', 'TYPE_IB_VARIATION',
                'RENEWAL', 'ANNUAL_REPORT', 'PSUR', 'DSUR'
            ) THEN
                v_err_msg := 'Invalid submission type: ' || rec.submission_type;
            END IF;

            -- Validate date format
            IF v_err_msg IS NULL AND rec.submitted_date IS NOT NULL THEN
                BEGIN
                    -- Try to parse date
                    DECLARE
                        v_test_date DATE;
                    BEGIN
                        v_test_date := TO_DATE(rec.submitted_date, 'YYYY-MM-DD');
                    EXCEPTION
                        WHEN OTHERS THEN
                            BEGIN
                                v_test_date := TO_DATE(rec.submitted_date, 'MM/DD/YYYY');
                            EXCEPTION
                                WHEN OTHERS THEN
                                    v_err_msg := 'Invalid date format: ' || rec.submitted_date;
                            END;
                    END;
                END;
            END IF;

            -- Update record status
            IF v_err_msg IS NOT NULL THEN
                UPDATE STG_REGULATORY_FEED
                SET process_status = 'ERROR',
                    error_message = v_err_msg
                WHERE feed_id = rec.feed_id;
                v_err := v_err + 1;
            ELSE
                UPDATE STG_REGULATORY_FEED
                SET process_status = 'VALIDATED'
                WHERE feed_id = rec.feed_id;
                v_valid := v_valid + 1;
            END IF;

            -- Commit in batches
            IF MOD(v_valid + v_err, 500) = 0 THEN
                COMMIT;
            END IF;
        END LOOP;

        COMMIT;

        p_validated := v_valid;
        p_errors := v_err;

        log_step(p_batch_id, 'VALIDATE_STAGING', 'COMPLETE',
                 v_valid + v_err, v_valid, v_err, 0, v_start);

        DBMS_OUTPUT.PUT_LINE('  Validated: ' || v_valid || ', Errors: ' || v_err);
    END validate_staging_data;

    -- ========================================================================
    -- STEP 2: Transform and standardize agency codes
    -- ========================================================================
    PROCEDURE transform_agency_codes(
        p_batch_id      IN  VARCHAR2,
        p_transformed   OUT NUMBER,
        p_errors        OUT NUMBER
    ) IS
        v_start     TIMESTAMP := SYSTIMESTAMP;
        v_trans     NUMBER := 0;
        v_err       NUMBER := 0;
        v_std_code  VARCHAR2(20);

        CURSOR c_validated IS
            SELECT feed_id, source_system, agency_code, agency_name, submitted_date
            FROM STG_REGULATORY_FEED
            WHERE batch_id = p_batch_id
              AND process_status = 'VALIDATED'
            FOR UPDATE;
    BEGIN
        DBMS_OUTPUT.PUT_LINE('Step 2: Transforming agency codes for batch ' || p_batch_id);

        FOR rec IN c_validated LOOP
            v_std_code := NULL;

            -- Try to map using the mapping table
            BEGIN
                SELECT standard_code
                INTO v_std_code
                FROM AGENCY_CODE_MAPPING
                WHERE source_system = rec.source_system
                  AND source_code = rec.agency_code
                  AND (effective_to IS NULL OR effective_to >= SYSDATE);
            EXCEPTION
                WHEN NO_DATA_FOUND THEN
                    -- Try direct match against AGENCY table
                    BEGIN
                        SELECT agency_code
                        INTO v_std_code
                        FROM AGENCY
                        WHERE agency_code = UPPER(rec.agency_code)
                          AND active_flag = 'Y';
                    EXCEPTION
                        WHEN NO_DATA_FOUND THEN
                            -- Try matching by name
                            BEGIN
                                SELECT agency_code
                                INTO v_std_code
                                FROM AGENCY
                                WHERE UPPER(agency_name) LIKE '%' || UPPER(rec.agency_name) || '%'
                                  AND active_flag = 'Y'
                                  AND ROWNUM = 1;
                            EXCEPTION
                                WHEN NO_DATA_FOUND THEN
                                    v_std_code := NULL;
                            END;
                    END;
            END;

            IF v_std_code IS NOT NULL THEN
                -- Standardize the agency code and normalize date format
                UPDATE STG_REGULATORY_FEED
                SET agency_code = v_std_code,
                    process_status = 'TRANSFORMED',
                    -- Normalize submitted_date to YYYY-MM-DD
                    submitted_date = CASE
                        WHEN submitted_date IS NOT NULL THEN
                            TO_CHAR(
                                COALESCE(
                                    TO_DATE(submitted_date DEFAULT NULL ON CONVERSION ERROR, 'YYYY-MM-DD'),
                                    TO_DATE(submitted_date DEFAULT NULL ON CONVERSION ERROR, 'MM/DD/YYYY'),
                                    TO_DATE(submitted_date DEFAULT NULL ON CONVERSION ERROR, 'DD-MON-YYYY')
                                ),
                                'YYYY-MM-DD'
                            )
                        ELSE submitted_date
                    END
                WHERE feed_id = rec.feed_id;
                v_trans := v_trans + 1;
            ELSE
                UPDATE STG_REGULATORY_FEED
                SET process_status = 'ERROR',
                    error_message = 'Unable to map agency code: ' || rec.agency_code ||
                                    ' from source: ' || rec.source_system
                WHERE feed_id = rec.feed_id;
                v_err := v_err + 1;
            END IF;

            IF MOD(v_trans + v_err, 500) = 0 THEN
                COMMIT;
            END IF;
        END LOOP;

        COMMIT;

        p_transformed := v_trans;
        p_errors := v_err;

        log_step(p_batch_id, 'TRANSFORM_AGENCY', 'COMPLETE',
                 v_trans + v_err, v_trans, v_err, 0, v_start);

        DBMS_OUTPUT.PUT_LINE('  Transformed: ' || v_trans || ', Errors: ' || v_err);
    END transform_agency_codes;

    -- ========================================================================
    -- STEP 3: Merge into target tables (UPSERT)
    -- ========================================================================
    PROCEDURE merge_submissions(
        p_batch_id      IN  VARCHAR2,
        p_inserted      OUT NUMBER,
        p_updated       OUT NUMBER,
        p_errors        OUT NUMBER
    ) IS
        v_start         TIMESTAMP := SYSTIMESTAMP;
        v_ins           NUMBER := 0;
        v_upd           NUMBER := 0;
        v_err           NUMBER := 0;
        v_drug_id       NUMBER;
        v_submission_id NUMBER;
        v_err_msg       VARCHAR2(4000);

        CURSOR c_transformed IS
            SELECT feed_id, drug_name, drug_ndc, submission_type,
                   submission_status, agency_code, tracking_number,
                   submitted_date, approval_date, approval_type,
                   conditions_text, priority, assigned_to, source_system
            FROM STG_REGULATORY_FEED
            WHERE batch_id = p_batch_id
              AND process_status = 'TRANSFORMED'
            FOR UPDATE;
    BEGIN
        DBMS_OUTPUT.PUT_LINE('Step 3: Merging submissions for batch ' || p_batch_id);

        FOR rec IN c_transformed LOOP
            v_err_msg := NULL;

            BEGIN
                -- Resolve or create drug record
                BEGIN
                    SELECT drug_id INTO v_drug_id
                    FROM DRUG
                    WHERE (ndc_code = rec.drug_ndc AND rec.drug_ndc IS NOT NULL)
                       OR (drug_name = rec.drug_name AND rec.drug_ndc IS NULL)
                    FETCH FIRST 1 ROW ONLY;
                EXCEPTION
                    WHEN NO_DATA_FOUND THEN
                        INSERT INTO DRUG (drug_name, ndc_code, manufacturer, active_flag)
                        VALUES (rec.drug_name, rec.drug_ndc, 'IMPORTED_' || rec.source_system, 'Y')
                        RETURNING drug_id INTO v_drug_id;
                END;

                -- MERGE submission
                MERGE INTO REGULATORY_SUBMISSION rs
                USING (
                    SELECT
                        v_drug_id                                           AS drug_id,
                        rec.submission_type                                 AS submission_type,
                        NVL(rec.submission_status, 'DRAFT')                 AS status,
                        rec.agency_code                                     AS agency,
                        rec.tracking_number                                 AS tracking_number,
                        CASE
                            WHEN rec.submitted_date IS NOT NULL
                            THEN TO_TIMESTAMP(rec.submitted_date, 'YYYY-MM-DD')
                            ELSE NULL
                        END                                                 AS submitted_date,
                        NVL(rec.priority, 'STANDARD')                       AS priority,
                        rec.assigned_to                                     AS assigned_to
                    FROM DUAL
                ) src
                ON (rs.tracking_number = src.tracking_number AND src.tracking_number IS NOT NULL)
                WHEN MATCHED THEN
                    UPDATE SET
                        rs.status = src.status,
                        rs.priority = src.priority,
                        rs.assigned_to = NVL(src.assigned_to, rs.assigned_to),
                        rs.modified_by = 'ETL_' || p_batch_id,
                        rs.modified_date = SYSTIMESTAMP
                WHEN NOT MATCHED THEN
                    INSERT (drug_id, submission_type, status, agency,
                            tracking_number, submitted_date, priority,
                            assigned_to, created_by, modified_by)
                    VALUES (src.drug_id, src.submission_type, src.status, src.agency,
                            src.tracking_number, src.submitted_date, src.priority,
                            src.assigned_to, 'ETL_' || p_batch_id, 'ETL_' || p_batch_id);

                -- Track insert vs update
                IF SQL%ROWCOUNT > 0 THEN
                    -- Check if it was an insert by looking for the tracking number
                    BEGIN
                        SELECT id INTO v_submission_id
                        FROM REGULATORY_SUBMISSION
                        WHERE tracking_number = rec.tracking_number
                          AND created_by = 'ETL_' || p_batch_id
                          AND created_date >= v_start;
                        v_ins := v_ins + 1;
                    EXCEPTION
                        WHEN NO_DATA_FOUND THEN
                            v_upd := v_upd + 1;
                        WHEN TOO_MANY_ROWS THEN
                            v_ins := v_ins + 1;
                    END;
                END IF;

                -- Handle approval data if present
                IF rec.approval_date IS NOT NULL AND rec.approval_type IS NOT NULL THEN
                    SELECT id INTO v_submission_id
                    FROM REGULATORY_SUBMISSION
                    WHERE tracking_number = rec.tracking_number
                    FETCH FIRST 1 ROW ONLY;

                    MERGE INTO APPROVAL_RECORD ar
                    USING (
                        SELECT
                            v_submission_id                                     AS submission_id,
                            TO_DATE(rec.approval_date, 'YYYY-MM-DD')            AS approval_date,
                            rec.approval_type                                   AS approval_type,
                            rec.conditions_text                                 AS conditions
                        FROM DUAL
                    ) src
                    ON (ar.submission_id = src.submission_id AND ar.approval_type = src.approval_type)
                    WHEN MATCHED THEN
                        UPDATE SET
                            ar.conditions = src.conditions,
                            ar.modified_date = SYSTIMESTAMP
                    WHEN NOT MATCHED THEN
                        INSERT (submission_id, approval_date, approval_type, conditions, created_by)
                        VALUES (src.submission_id, src.approval_date, src.approval_type,
                                src.conditions, 'ETL_' || p_batch_id);
                END IF;

                -- Mark as loaded
                UPDATE STG_REGULATORY_FEED
                SET process_status = 'LOADED'
                WHERE feed_id = rec.feed_id;

            EXCEPTION
                WHEN OTHERS THEN
                    v_err_msg := SQLERRM;
                    v_err := v_err + 1;
                    UPDATE STG_REGULATORY_FEED
                    SET process_status = 'ERROR',
                        error_message = v_err_msg
                    WHERE feed_id = rec.feed_id;
            END;

            IF MOD(v_ins + v_upd + v_err, 500) = 0 THEN
                COMMIT;
            END IF;
        END LOOP;

        COMMIT;

        p_inserted := v_ins;
        p_updated := v_upd;
        p_errors := v_err;

        log_step(p_batch_id, 'MERGE_SUBMISSIONS', 'COMPLETE',
                 v_ins + v_upd + v_err, v_ins + v_upd, v_err, 0, v_start);

        DBMS_OUTPUT.PUT_LINE('  Inserted: ' || v_ins || ', Updated: ' || v_upd ||
                             ', Errors: ' || v_err);
    END merge_submissions;

    -- ========================================================================
    -- STEP 4: Post-load business rule validation
    -- ========================================================================
    PROCEDURE validate_business_rules(
        p_batch_id      IN  VARCHAR2,
        p_valid         OUT NUMBER,
        p_invalid       OUT NUMBER
    ) IS
        v_start     TIMESTAMP := SYSTIMESTAMP;
        v_valid     NUMBER := 0;
        v_invalid   NUMBER := 0;
        v_err_msg   VARCHAR2(4000);

        CURSOR c_loaded IS
            SELECT
                stg.feed_id,
                rs.id AS submission_id,
                rs.status,
                rs.submitted_date,
                rs.agency,
                rs.submission_type,
                stg.tracking_number
            FROM STG_REGULATORY_FEED stg
            JOIN REGULATORY_SUBMISSION rs ON rs.tracking_number = stg.tracking_number
            WHERE stg.batch_id = p_batch_id
              AND stg.process_status = 'LOADED';
    BEGIN
        DBMS_OUTPUT.PUT_LINE('Step 4: Validating business rules for batch ' || p_batch_id);

        FOR rec IN c_loaded LOOP
            v_err_msg := NULL;

            -- Rule 1: Submitted status must have a submitted_date
            IF rec.status = 'SUBMITTED' AND rec.submitted_date IS NULL THEN
                v_err_msg := 'SUBMITTED status requires submitted_date';
            END IF;

            -- Rule 2: Agency-specific submission type validation
            IF v_err_msg IS NULL THEN
                IF rec.agency = 'FDA' AND rec.submission_type IN ('MAA', 'TYPE_II_VARIATION', 'TYPE_IB_VARIATION') THEN
                    v_err_msg := 'Submission type ' || rec.submission_type || ' not valid for FDA';
                ELSIF rec.agency = 'EMA' AND rec.submission_type IN ('NDA', 'ANDA', 'BLA', 'IND') THEN
                    v_err_msg := 'Submission type ' || rec.submission_type || ' not valid for EMA';
                END IF;
            END IF;

            -- Rule 3: Check for duplicate submissions (same drug+type+agency within 30 days)
            IF v_err_msg IS NULL THEN
                DECLARE
                    v_dup_count NUMBER;
                BEGIN
                    SELECT COUNT(*) INTO v_dup_count
                    FROM REGULATORY_SUBMISSION rs2
                    JOIN REGULATORY_SUBMISSION rs1 ON rs1.id = rec.submission_id
                    WHERE rs2.drug_id = rs1.drug_id
                      AND rs2.submission_type = rs1.submission_type
                      AND rs2.agency = rs1.agency
                      AND rs2.id != rs1.id
                      AND ABS(EXTRACT(DAY FROM (rs2.created_date - rs1.created_date))) < 30;

                    IF v_dup_count > 0 THEN
                        v_err_msg := 'WARNING: Possible duplicate - ' || v_dup_count ||
                                     ' similar submissions found within 30 days';
                        -- This is a warning, not a hard error
                        UPDATE STG_REGULATORY_FEED
                        SET error_message = v_err_msg
                        WHERE feed_id = rec.feed_id;
                    END IF;
                END;
            END IF;

            IF v_err_msg IS NOT NULL AND SUBSTR(v_err_msg, 1, 7) != 'WARNING' THEN
                v_invalid := v_invalid + 1;
                UPDATE STG_REGULATORY_FEED
                SET error_message = v_err_msg
                WHERE feed_id = rec.feed_id;
            ELSE
                v_valid := v_valid + 1;
            END IF;
        END LOOP;

        COMMIT;

        p_valid := v_valid;
        p_invalid := v_invalid;

        log_step(p_batch_id, 'VALIDATE_BUSINESS_RULES', 'COMPLETE',
                 v_valid + v_invalid, v_valid, v_invalid, 0, v_start);

        DBMS_OUTPUT.PUT_LINE('  Valid: ' || v_valid || ', Invalid: ' || v_invalid);
    END validate_business_rules;

    -- ========================================================================
    -- STEP 5: Generate load summary
    -- ========================================================================
    PROCEDURE generate_load_summary(
        p_batch_id      IN  VARCHAR2,
        p_cursor        OUT SYS_REFCURSOR
    ) IS
    BEGIN
        OPEN p_cursor FOR
            SELECT
                p_batch_id                                          AS batch_id,
                COUNT(*)                                            AS total_records,
                SUM(CASE WHEN process_status = 'LOADED' THEN 1 ELSE 0 END)     AS loaded,
                SUM(CASE WHEN process_status = 'ERROR' THEN 1 ELSE 0 END)      AS errors,
                SUM(CASE WHEN process_status = 'SKIPPED' THEN 1 ELSE 0 END)    AS skipped,
                SUM(CASE WHEN process_status = 'NEW' THEN 1 ELSE 0 END)        AS unprocessed,
                MIN(load_timestamp)                                 AS first_record_time,
                MAX(load_timestamp)                                 AS last_record_time,
                COUNT(DISTINCT source_system)                       AS source_count,
                COUNT(DISTINCT agency_code)                         AS agency_count,
                LISTAGG(DISTINCT source_system, ', ')
                    WITHIN GROUP (ORDER BY source_system)           AS source_systems,
                LISTAGG(DISTINCT agency_code, ', ')
                    WITHIN GROUP (ORDER BY agency_code)             AS agencies
            FROM STG_REGULATORY_FEED
            WHERE batch_id = p_batch_id;

        DBMS_OUTPUT.PUT_LINE('Load summary generated for batch ' || p_batch_id);
    END generate_load_summary;

    -- ========================================================================
    -- MAIN ETL ORCHESTRATOR
    -- ========================================================================
    PROCEDURE run_full_etl(
        p_batch_id      IN  VARCHAR2,
        p_run_by        IN  VARCHAR2 DEFAULT USER,
        p_success       OUT BOOLEAN,
        p_summary       OUT VARCHAR2
    ) IS
        v_start         TIMESTAMP := SYSTIMESTAMP;
        v_validated     NUMBER := 0;
        v_transformed   NUMBER := 0;
        v_inserted      NUMBER := 0;
        v_updated       NUMBER := 0;
        v_valid         NUMBER := 0;
        v_val_errors    NUMBER := 0;
        v_trans_errors  NUMBER := 0;
        v_merge_errors  NUMBER := 0;
        v_rule_invalid  NUMBER := 0;
        v_total_errors  NUMBER := 0;
    BEGIN
        DBMS_OUTPUT.PUT_LINE('========================================');
        DBMS_OUTPUT.PUT_LINE('RegRecord ETL Pipeline');
        DBMS_OUTPUT.PUT_LINE('Batch: ' || p_batch_id);
        DBMS_OUTPUT.PUT_LINE('Started: ' || TO_CHAR(SYSTIMESTAMP, 'YYYY-MM-DD HH24:MI:SS'));
        DBMS_OUTPUT.PUT_LINE('Run by: ' || p_run_by);
        DBMS_OUTPUT.PUT_LINE('========================================');

        -- Step 1: Validate
        validate_staging_data(p_batch_id, v_validated, v_val_errors);

        -- Step 2: Transform
        transform_agency_codes(p_batch_id, v_transformed, v_trans_errors);

        -- Step 3: Merge
        merge_submissions(p_batch_id, v_inserted, v_updated, v_merge_errors);

        -- Step 4: Business rules
        validate_business_rules(p_batch_id, v_valid, v_rule_invalid);

        v_total_errors := v_val_errors + v_trans_errors + v_merge_errors + v_rule_invalid;

        -- Build summary
        p_summary := 'Batch ' || p_batch_id || ': ' ||
                     'Validated=' || v_validated || ', ' ||
                     'Transformed=' || v_transformed || ', ' ||
                     'Inserted=' || v_inserted || ', ' ||
                     'Updated=' || v_updated || ', ' ||
                     'Valid=' || v_valid || ', ' ||
                     'Errors=' || v_total_errors;

        p_success := (v_total_errors = 0);

        -- Log overall
        log_step(p_batch_id, 'FULL_ETL',
                 CASE WHEN v_total_errors = 0 THEN 'SUCCESS' ELSE 'PARTIAL' END,
                 v_validated + v_val_errors,
                 v_inserted + v_updated,
                 v_total_errors, 0, v_start);

        DBMS_OUTPUT.PUT_LINE('========================================');
        DBMS_OUTPUT.PUT_LINE('ETL Complete: ' || p_summary);
        DBMS_OUTPUT.PUT_LINE('Duration: ' ||
            ROUND(EXTRACT(SECOND FROM (SYSTIMESTAMP - v_start)) +
                  EXTRACT(MINUTE FROM (SYSTIMESTAMP - v_start)) * 60, 2) || 's');
        DBMS_OUTPUT.PUT_LINE('========================================');

    EXCEPTION
        WHEN OTHERS THEN
            p_success := FALSE;
            p_summary := 'ETL FAILED for batch ' || p_batch_id || ': ' || SQLERRM;

            log_step(p_batch_id, 'FULL_ETL', 'FAILED',
                     0, 0, 1, 0, v_start, SQLERRM);

            DBMS_OUTPUT.PUT_LINE('ETL FAILED: ' || SQLERRM);
            RAISE;
    END run_full_etl;

END PKG_ETL_REGULATORY;
/
