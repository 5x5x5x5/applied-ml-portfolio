/*******************************************************************************
 * PharmaDataVault - ETL Staging Procedures for Drug Data
 *
 * PL/SQL procedures to load, cleanse, validate, and deduplicate drug data
 * from CSV flat files into staging tables before vault loading.
 *
 * Process Flow:
 *   1. External table reads CSV files from OS directory
 *   2. SP_STAGE_DRUG_DATA loads into STG_DRUG_RAW with validation
 *   3. SP_CLEANSE_DRUG_DATA standardizes and deduplicates
 *   4. SP_VALIDATE_DRUG_DATA performs business rule checks
 *   5. Clean records promoted to STG_DRUG_CLEAN for vault loading
 *
 * Oracle PL/SQL compatible
 ******************************************************************************/

-- ============================================================================
-- Staging Tables
-- ============================================================================

CREATE TABLE STG_DRUG_RAW (
    STG_ID              NUMBER          GENERATED ALWAYS AS IDENTITY,
    BATCH_ID            NUMBER          NOT NULL,
    SOURCE_FILE         VARCHAR2(200)   NOT NULL,
    SOURCE_ROW_NUM      NUMBER          NOT NULL,
    -- Raw fields from CSV (all VARCHAR2 for initial load)
    RAW_NDC             VARCHAR2(50)    NULL,
    RAW_DRUG_NAME       VARCHAR2(500)   NULL,
    RAW_GENERIC_NAME    VARCHAR2(500)   NULL,
    RAW_MANUFACTURER    VARCHAR2(500)   NULL,
    RAW_FORM            VARCHAR2(100)   NULL,
    RAW_STRENGTH        VARCHAR2(100)   NULL,
    RAW_ROUTE           VARCHAR2(100)   NULL,
    RAW_DEA_SCHEDULE    VARCHAR2(20)    NULL,
    RAW_APPROVAL_DATE   VARCHAR2(50)    NULL,
    RAW_THERAPEUTIC     VARCHAR2(200)   NULL,
    RAW_NDA_NUMBER      VARCHAR2(50)    NULL,
    -- Staging metadata
    LOAD_TIMESTAMP      TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL,
    VALIDATION_STATUS   VARCHAR2(20)    DEFAULT 'PENDING' NOT NULL,  -- PENDING, VALID, INVALID
    ERROR_CODES         VARCHAR2(500)   NULL,
    CONSTRAINT PK_STG_DRUG_RAW PRIMARY KEY (STG_ID)
)
TABLESPACE TBS_STAGING;

CREATE TABLE STG_DRUG_CLEAN (
    STG_ID              NUMBER          NOT NULL,
    BATCH_ID            NUMBER          NOT NULL,
    -- Cleaned and typed fields
    DRUG_NDC            VARCHAR2(13)    NOT NULL,
    DRUG_NAME           VARCHAR2(200)   NOT NULL,
    GENERIC_NAME        VARCHAR2(200)   NULL,
    MANUFACTURER        VARCHAR2(200)   NOT NULL,
    DRUG_FORM           VARCHAR2(50)    NOT NULL,
    STRENGTH            VARCHAR2(50)    NOT NULL,
    STRENGTH_UNIT       VARCHAR2(20)    NULL,
    ROUTE_OF_ADMIN      VARCHAR2(50)    NULL,
    DEA_SCHEDULE        VARCHAR2(5)     NULL,
    APPROVAL_DATE       DATE            NULL,
    THERAPEUTIC_CLASS   VARCHAR2(100)   NULL,
    NDA_NUMBER          VARCHAR2(20)    NULL,
    -- Hash for vault loading
    DRUG_HASH_KEY       RAW(16)         NOT NULL,   -- MD5 of DRUG_NDC
    DRUG_HASHDIFF       RAW(16)         NOT NULL,   -- MD5 of all descriptive cols
    RECORD_SOURCE       VARCHAR2(100)   NOT NULL,
    LOAD_TIMESTAMP      TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL,
    CONSTRAINT PK_STG_DRUG_CLEAN PRIMARY KEY (STG_ID)
)
TABLESPACE TBS_STAGING;

CREATE INDEX IDX_STG_DRUG_CLEAN_NDC ON STG_DRUG_CLEAN (DRUG_NDC)
    TABLESPACE TBS_STAGING;

CREATE INDEX IDX_STG_DRUG_CLEAN_HASH ON STG_DRUG_CLEAN (DRUG_HASH_KEY)
    TABLESPACE TBS_STAGING;

-- ============================================================================
-- ETL Batch Tracking Table
-- ============================================================================

CREATE TABLE ETL_BATCH_CONTROL (
    BATCH_ID            NUMBER          GENERATED ALWAYS AS IDENTITY,
    BATCH_TYPE          VARCHAR2(50)    NOT NULL,   -- DRUG, PATIENT, TRIAL, AE, MFG
    SOURCE_FILE         VARCHAR2(200)   NULL,
    START_TIMESTAMP     TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL,
    END_TIMESTAMP       TIMESTAMP       NULL,
    STATUS              VARCHAR2(20)    DEFAULT 'RUNNING' NOT NULL,
    ROWS_READ           NUMBER          DEFAULT 0,
    ROWS_VALID          NUMBER          DEFAULT 0,
    ROWS_INVALID        NUMBER          DEFAULT 0,
    ROWS_LOADED         NUMBER          DEFAULT 0,
    ERROR_MESSAGE       VARCHAR2(4000)  NULL,
    CONSTRAINT PK_ETL_BATCH PRIMARY KEY (BATCH_ID),
    CONSTRAINT CK_BATCH_STATUS CHECK (STATUS IN ('RUNNING', 'SUCCESS', 'FAILED', 'PARTIAL'))
)
TABLESPACE TBS_STAGING;

-- ============================================================================
-- SP_STAGE_DRUG_DATA: Load raw drug data from external table/CSV
-- ============================================================================

CREATE OR REPLACE PROCEDURE SP_STAGE_DRUG_DATA (
    p_source_file       IN VARCHAR2,
    p_batch_id          OUT NUMBER,
    p_rows_loaded       OUT NUMBER
)
AS
    v_proc_name     CONSTANT VARCHAR2(100) := 'SP_STAGE_DRUG_DATA';
    v_batch_id      NUMBER;
    v_row_count     NUMBER := 0;
    v_error_count   NUMBER := 0;

    -- Cursor to read from external table (pre-configured for CSV directory)
    CURSOR c_ext_drug IS
        SELECT
            NDC_CODE,
            DRUG_NAME,
            GENERIC_NAME,
            MANUFACTURER,
            DOSAGE_FORM,
            STRENGTH,
            ROUTE,
            DEA_SCHEDULE,
            APPROVAL_DATE,
            THERAPEUTIC_CLASS,
            NDA_NUMBER,
            ROWNUM AS ROW_NUM
        FROM EXT_DRUG_FEED;

    -- Bulk collect types
    TYPE t_ndc_tab       IS TABLE OF VARCHAR2(50);
    TYPE t_name_tab      IS TABLE OF VARCHAR2(500);
    TYPE t_generic_tab   IS TABLE OF VARCHAR2(500);
    TYPE t_mfr_tab       IS TABLE OF VARCHAR2(500);
    TYPE t_form_tab      IS TABLE OF VARCHAR2(100);
    TYPE t_str_tab       IS TABLE OF VARCHAR2(100);
    TYPE t_route_tab     IS TABLE OF VARCHAR2(100);
    TYPE t_dea_tab       IS TABLE OF VARCHAR2(20);
    TYPE t_date_tab      IS TABLE OF VARCHAR2(50);
    TYPE t_ther_tab      IS TABLE OF VARCHAR2(200);
    TYPE t_nda_tab       IS TABLE OF VARCHAR2(50);
    TYPE t_rownum_tab    IS TABLE OF NUMBER;

    l_ndcs          t_ndc_tab;
    l_names         t_name_tab;
    l_generics      t_generic_tab;
    l_mfrs          t_mfr_tab;
    l_forms         t_form_tab;
    l_strengths     t_str_tab;
    l_routes        t_route_tab;
    l_deas          t_dea_tab;
    l_dates         t_date_tab;
    l_thers         t_ther_tab;
    l_ndas          t_nda_tab;
    l_rownums       t_rownum_tab;

    c_bulk_limit    CONSTANT PLS_INTEGER := 5000;

BEGIN
    -- Create batch record
    INSERT INTO ETL_BATCH_CONTROL (BATCH_TYPE, SOURCE_FILE, STATUS)
    VALUES ('DRUG', p_source_file, 'RUNNING')
    RETURNING BATCH_ID INTO v_batch_id;

    p_batch_id := v_batch_id;

    -- Bulk load from external table into staging
    OPEN c_ext_drug;
    LOOP
        FETCH c_ext_drug BULK COLLECT INTO
            l_ndcs, l_names, l_generics, l_mfrs,
            l_forms, l_strengths, l_routes, l_deas,
            l_dates, l_thers, l_ndas, l_rownums
        LIMIT c_bulk_limit;

        EXIT WHEN l_ndcs.COUNT = 0;

        FORALL i IN 1..l_ndcs.COUNT
            INSERT INTO STG_DRUG_RAW (
                BATCH_ID, SOURCE_FILE, SOURCE_ROW_NUM,
                RAW_NDC, RAW_DRUG_NAME, RAW_GENERIC_NAME,
                RAW_MANUFACTURER, RAW_FORM, RAW_STRENGTH,
                RAW_ROUTE, RAW_DEA_SCHEDULE, RAW_APPROVAL_DATE,
                RAW_THERAPEUTIC, RAW_NDA_NUMBER,
                VALIDATION_STATUS
            ) VALUES (
                v_batch_id, p_source_file, l_rownums(i),
                l_ndcs(i), l_names(i), l_generics(i),
                l_mfrs(i), l_forms(i), l_strengths(i),
                l_routes(i), l_deas(i), l_dates(i),
                l_thers(i), l_ndas(i),
                'PENDING'
            );

        v_row_count := v_row_count + l_ndcs.COUNT;
    END LOOP;
    CLOSE c_ext_drug;

    -- Update batch control
    UPDATE ETL_BATCH_CONTROL
    SET ROWS_READ = v_row_count
    WHERE BATCH_ID = v_batch_id;

    p_rows_loaded := v_row_count;
    COMMIT;

EXCEPTION
    WHEN OTHERS THEN
        IF c_ext_drug%ISOPEN THEN
            CLOSE c_ext_drug;
        END IF;
        ROLLBACK;

        UPDATE ETL_BATCH_CONTROL
        SET STATUS = 'FAILED',
            END_TIMESTAMP = SYSTIMESTAMP,
            ERROR_MESSAGE = SQLERRM
        WHERE BATCH_ID = v_batch_id;
        COMMIT;

        INSERT INTO ETL_ERROR_LOG (
            PROCEDURE_NAME, ERROR_CODE, ERROR_MESSAGE,
            SOURCE_TABLE, TARGET_TABLE, BATCH_ID
        ) VALUES (
            v_proc_name, SQLCODE, SQLERRM,
            'EXT_DRUG_FEED', 'STG_DRUG_RAW', v_batch_id
        );
        COMMIT;

        RAISE;
END SP_STAGE_DRUG_DATA;
/

-- ============================================================================
-- SP_CLEANSE_DRUG_DATA: Standardize and deduplicate drug staging data
-- ============================================================================

CREATE OR REPLACE PROCEDURE SP_CLEANSE_DRUG_DATA (
    p_batch_id          IN NUMBER,
    p_rows_cleaned      OUT NUMBER,
    p_rows_rejected     OUT NUMBER
)
AS
    v_proc_name     CONSTANT VARCHAR2(100) := 'SP_CLEANSE_DRUG_DATA';
    v_valid_count   NUMBER := 0;
    v_invalid_count NUMBER := 0;
    v_error_codes   VARCHAR2(500);
    v_ndc_clean     VARCHAR2(13);
    v_approval_dt   DATE;

    CURSOR c_raw_drugs IS
        SELECT STG_ID, RAW_NDC, RAW_DRUG_NAME, RAW_GENERIC_NAME,
               RAW_MANUFACTURER, RAW_FORM, RAW_STRENGTH, RAW_ROUTE,
               RAW_DEA_SCHEDULE, RAW_APPROVAL_DATE, RAW_THERAPEUTIC,
               RAW_NDA_NUMBER
        FROM STG_DRUG_RAW
        WHERE BATCH_ID = p_batch_id
          AND VALIDATION_STATUS = 'PENDING'
        FOR UPDATE;

BEGIN
    FOR rec IN c_raw_drugs LOOP
        v_error_codes := NULL;

        -- =====================================================================
        -- Validation 1: NDC format (must be 5-4-2 or 5-3-2 or numeric)
        -- =====================================================================
        BEGIN
            v_ndc_clean := REGEXP_REPLACE(TRIM(rec.RAW_NDC), '[^0-9-]', '');

            -- Standardize to 5-4-2 format
            IF REGEXP_LIKE(v_ndc_clean, '^\d{5}-\d{4}-\d{2}$') THEN
                NULL; -- Already in standard format
            ELSIF REGEXP_LIKE(v_ndc_clean, '^\d{5}-\d{3}-\d{2}$') THEN
                -- 5-3-2: pad segment 2 with leading zero
                v_ndc_clean := SUBSTR(v_ndc_clean, 1, 6) || '0' || SUBSTR(v_ndc_clean, 7);
            ELSIF REGEXP_LIKE(v_ndc_clean, '^\d{11}$') THEN
                -- Plain 11-digit: format as 5-4-2
                v_ndc_clean := SUBSTR(v_ndc_clean, 1, 5) || '-'
                            || SUBSTR(v_ndc_clean, 6, 4) || '-'
                            || SUBSTR(v_ndc_clean, 10, 2);
            ELSE
                v_error_codes := v_error_codes || 'INVALID_NDC;';
            END IF;
        EXCEPTION
            WHEN OTHERS THEN
                v_error_codes := v_error_codes || 'NDC_PARSE_ERROR;';
        END;

        -- =====================================================================
        -- Validation 2: Required fields
        -- =====================================================================
        IF TRIM(rec.RAW_DRUG_NAME) IS NULL THEN
            v_error_codes := v_error_codes || 'MISSING_DRUG_NAME;';
        END IF;

        IF TRIM(rec.RAW_MANUFACTURER) IS NULL THEN
            v_error_codes := v_error_codes || 'MISSING_MANUFACTURER;';
        END IF;

        IF TRIM(rec.RAW_FORM) IS NULL THEN
            v_error_codes := v_error_codes || 'MISSING_FORM;';
        END IF;

        IF TRIM(rec.RAW_STRENGTH) IS NULL THEN
            v_error_codes := v_error_codes || 'MISSING_STRENGTH;';
        END IF;

        -- =====================================================================
        -- Validation 3: Date parsing
        -- =====================================================================
        BEGIN
            IF rec.RAW_APPROVAL_DATE IS NOT NULL THEN
                -- Try multiple date formats
                BEGIN
                    v_approval_dt := TO_DATE(TRIM(rec.RAW_APPROVAL_DATE), 'YYYY-MM-DD');
                EXCEPTION
                    WHEN OTHERS THEN
                        BEGIN
                            v_approval_dt := TO_DATE(TRIM(rec.RAW_APPROVAL_DATE), 'MM/DD/YYYY');
                        EXCEPTION
                            WHEN OTHERS THEN
                                BEGIN
                                    v_approval_dt := TO_DATE(TRIM(rec.RAW_APPROVAL_DATE), 'DD-MON-YYYY');
                                EXCEPTION
                                    WHEN OTHERS THEN
                                        v_error_codes := v_error_codes || 'INVALID_APPROVAL_DATE;';
                                        v_approval_dt := NULL;
                                END;
                        END;
                END;
            ELSE
                v_approval_dt := NULL;
            END IF;
        END;

        -- =====================================================================
        -- Validation 4: DEA schedule validation
        -- =====================================================================
        IF rec.RAW_DEA_SCHEDULE IS NOT NULL
           AND UPPER(TRIM(rec.RAW_DEA_SCHEDULE)) NOT IN ('I', 'II', 'III', 'IV', 'V', 'N/A', '') THEN
            v_error_codes := v_error_codes || 'INVALID_DEA_SCHEDULE;';
        END IF;

        -- =====================================================================
        -- Route record based on validation results
        -- =====================================================================
        IF v_error_codes IS NULL THEN
            -- Valid: insert into clean staging
            UPDATE STG_DRUG_RAW
            SET VALIDATION_STATUS = 'VALID'
            WHERE CURRENT OF c_raw_drugs;

            INSERT INTO STG_DRUG_CLEAN (
                STG_ID, BATCH_ID,
                DRUG_NDC, DRUG_NAME, GENERIC_NAME, MANUFACTURER,
                DRUG_FORM, STRENGTH, STRENGTH_UNIT, ROUTE_OF_ADMIN,
                DEA_SCHEDULE, APPROVAL_DATE, THERAPEUTIC_CLASS, NDA_NUMBER,
                DRUG_HASH_KEY, DRUG_HASHDIFF, RECORD_SOURCE
            ) VALUES (
                rec.STG_ID, p_batch_id,
                v_ndc_clean,
                INITCAP(TRIM(rec.RAW_DRUG_NAME)),
                INITCAP(TRIM(rec.RAW_GENERIC_NAME)),
                UPPER(TRIM(rec.RAW_MANUFACTURER)),
                LOWER(TRIM(rec.RAW_FORM)),
                TRIM(rec.RAW_STRENGTH),
                REGEXP_SUBSTR(TRIM(rec.RAW_STRENGTH), '[a-zA-Z/]+$'),
                LOWER(TRIM(rec.RAW_ROUTE)),
                CASE
                    WHEN UPPER(TRIM(rec.RAW_DEA_SCHEDULE)) IN ('N/A', '')
                    THEN NULL
                    ELSE UPPER(TRIM(rec.RAW_DEA_SCHEDULE))
                END,
                v_approval_dt,
                TRIM(rec.RAW_THERAPEUTIC),
                TRIM(rec.RAW_NDA_NUMBER),
                -- Hash key: MD5 of business key
                STANDARD_HASH(v_ndc_clean, 'MD5'),
                -- Hashdiff: MD5 of all descriptive columns
                STANDARD_HASH(
                    NVL(INITCAP(TRIM(rec.RAW_DRUG_NAME)), '^^')
                    || '|' || NVL(INITCAP(TRIM(rec.RAW_GENERIC_NAME)), '^^')
                    || '|' || NVL(UPPER(TRIM(rec.RAW_MANUFACTURER)), '^^')
                    || '|' || NVL(LOWER(TRIM(rec.RAW_FORM)), '^^')
                    || '|' || NVL(TRIM(rec.RAW_STRENGTH), '^^')
                    || '|' || NVL(LOWER(TRIM(rec.RAW_ROUTE)), '^^')
                    || '|' || NVL(UPPER(TRIM(rec.RAW_DEA_SCHEDULE)), '^^')
                    || '|' || NVL(TO_CHAR(v_approval_dt, 'YYYY-MM-DD'), '^^')
                    || '|' || NVL(TRIM(rec.RAW_THERAPEUTIC), '^^')
                    || '|' || NVL(TRIM(rec.RAW_NDA_NUMBER), '^^'),
                    'MD5'
                ),
                'DRUG_FEED:' || TO_CHAR(SYSDATE, 'YYYY-MM-DD')
            );

            v_valid_count := v_valid_count + 1;
        ELSE
            -- Invalid: mark and log errors
            UPDATE STG_DRUG_RAW
            SET VALIDATION_STATUS = 'INVALID',
                ERROR_CODES = v_error_codes
            WHERE CURRENT OF c_raw_drugs;

            INSERT INTO ETL_ERROR_LOG (
                PROCEDURE_NAME, ERROR_CODE, ERROR_MESSAGE,
                SOURCE_TABLE, TARGET_TABLE, RECORD_KEY, BATCH_ID
            ) VALUES (
                v_proc_name, -20001, v_error_codes,
                'STG_DRUG_RAW', 'STG_DRUG_CLEAN',
                'STG_ID=' || rec.STG_ID || ',NDC=' || rec.RAW_NDC,
                p_batch_id
            );

            v_invalid_count := v_invalid_count + 1;
        END IF;

        -- Commit in batches to avoid long transactions
        IF MOD(v_valid_count + v_invalid_count, 5000) = 0 THEN
            COMMIT;
        END IF;
    END LOOP;

    -- =========================================================================
    -- Deduplication: keep latest record per NDC within this batch
    -- =========================================================================
    DELETE FROM STG_DRUG_CLEAN a
    WHERE a.BATCH_ID = p_batch_id
      AND a.STG_ID < (
          SELECT MAX(b.STG_ID)
          FROM STG_DRUG_CLEAN b
          WHERE b.BATCH_ID = p_batch_id
            AND b.DRUG_NDC = a.DRUG_NDC
      );

    -- Update batch control
    UPDATE ETL_BATCH_CONTROL
    SET ROWS_VALID = v_valid_count,
        ROWS_INVALID = v_invalid_count
    WHERE BATCH_ID = p_batch_id;

    p_rows_cleaned := v_valid_count;
    p_rows_rejected := v_invalid_count;
    COMMIT;

EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        UPDATE ETL_BATCH_CONTROL
        SET STATUS = 'FAILED',
            END_TIMESTAMP = SYSTIMESTAMP,
            ERROR_MESSAGE = SQLERRM
        WHERE BATCH_ID = p_batch_id;
        COMMIT;

        INSERT INTO ETL_ERROR_LOG (
            PROCEDURE_NAME, ERROR_CODE, ERROR_MESSAGE,
            SOURCE_TABLE, TARGET_TABLE, BATCH_ID
        ) VALUES (
            v_proc_name, SQLCODE, SQLERRM,
            'STG_DRUG_RAW', 'STG_DRUG_CLEAN', p_batch_id
        );
        COMMIT;
        RAISE;
END SP_CLEANSE_DRUG_DATA;
/

-- ============================================================================
-- SP_VALIDATE_DRUG_DATA: Final business rule validation before vault load
-- ============================================================================

CREATE OR REPLACE PROCEDURE SP_VALIDATE_DRUG_DATA (
    p_batch_id          IN NUMBER,
    p_is_valid          OUT BOOLEAN
)
AS
    v_proc_name     CONSTANT VARCHAR2(100) := 'SP_VALIDATE_DRUG_DATA';
    v_total_count   NUMBER;
    v_dup_ndc_count NUMBER;
    v_future_dates  NUMBER;
    v_threshold     CONSTANT NUMBER := 0.05;  -- 5% error threshold
    v_error_rate    NUMBER;
BEGIN
    -- Count total clean records
    SELECT COUNT(*)
    INTO v_total_count
    FROM STG_DRUG_CLEAN
    WHERE BATCH_ID = p_batch_id;

    IF v_total_count = 0 THEN
        p_is_valid := FALSE;
        INSERT INTO ETL_ERROR_LOG (
            PROCEDURE_NAME, ERROR_CODE, ERROR_MESSAGE, BATCH_ID
        ) VALUES (
            v_proc_name, -20010, 'No valid records in batch', p_batch_id
        );
        COMMIT;
        RETURN;
    END IF;

    -- Check for duplicate NDCs across batches (cross-batch dedup)
    SELECT COUNT(*)
    INTO v_dup_ndc_count
    FROM STG_DRUG_CLEAN sc
    WHERE sc.BATCH_ID = p_batch_id
      AND EXISTS (
          SELECT 1
          FROM STG_DRUG_CLEAN sc2
          WHERE sc2.DRUG_NDC = sc.DRUG_NDC
            AND sc2.BATCH_ID < sc.BATCH_ID
            AND sc2.BATCH_ID >= p_batch_id - 30  -- Look back 30 batches
      );

    -- Check for future approval dates
    SELECT COUNT(*)
    INTO v_future_dates
    FROM STG_DRUG_CLEAN
    WHERE BATCH_ID = p_batch_id
      AND APPROVAL_DATE > SYSDATE;

    -- Calculate error rate from raw staging
    SELECT NVL(
        (SELECT ROWS_INVALID FROM ETL_BATCH_CONTROL WHERE BATCH_ID = p_batch_id)
        / NULLIF(
            (SELECT ROWS_READ FROM ETL_BATCH_CONTROL WHERE BATCH_ID = p_batch_id),
            0
        ), 0
    )
    INTO v_error_rate
    FROM DUAL;

    -- Determine if batch passes validation
    IF v_error_rate > v_threshold THEN
        p_is_valid := FALSE;
        INSERT INTO ETL_ERROR_LOG (
            PROCEDURE_NAME, ERROR_CODE, ERROR_MESSAGE, BATCH_ID
        ) VALUES (
            v_proc_name, -20011,
            'Error rate ' || ROUND(v_error_rate * 100, 2) || '% exceeds threshold '
            || ROUND(v_threshold * 100, 2) || '%',
            p_batch_id
        );
    ELSIF v_future_dates > 0 THEN
        p_is_valid := FALSE;
        INSERT INTO ETL_ERROR_LOG (
            PROCEDURE_NAME, ERROR_CODE, ERROR_MESSAGE, BATCH_ID
        ) VALUES (
            v_proc_name, -20012,
            v_future_dates || ' records have future approval dates',
            p_batch_id
        );
    ELSE
        p_is_valid := TRUE;
    END IF;

    COMMIT;

EXCEPTION
    WHEN OTHERS THEN
        p_is_valid := FALSE;
        INSERT INTO ETL_ERROR_LOG (
            PROCEDURE_NAME, ERROR_CODE, ERROR_MESSAGE, BATCH_ID
        ) VALUES (
            v_proc_name, SQLCODE, SQLERRM, p_batch_id
        );
        COMMIT;
END SP_VALIDATE_DRUG_DATA;
/
