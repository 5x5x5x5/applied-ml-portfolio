-- ============================================================================
-- PL/SQL Data Quality Checks Package
--
-- Comprehensive data quality validation for pharmaceutical data warehouse:
--   1. Referential integrity validation
--   2. Null checks on required fields
--   3. Range validation (dates, numeric bounds)
--   4. Cross-table consistency checks
--   5. Quality score computation
--
-- Results are logged to dq_validation_result for audit and reporting.
-- ============================================================================

CREATE OR REPLACE PACKAGE pkg_data_quality AS

    -- Run all quality checks for a given batch
    PROCEDURE run_all_checks(
        p_batch_id      IN  NUMBER,
        p_target_table  IN  VARCHAR2     DEFAULT NULL,  -- NULL = all tables
        p_overall_score OUT NUMBER,
        p_total_rules   OUT NUMBER,
        p_passed_rules  OUT NUMBER,
        p_failed_rules  OUT NUMBER,
        p_return_code   OUT NUMBER,
        p_return_msg    OUT VARCHAR2
    );

    -- Individual check categories
    PROCEDURE check_referential_integrity(p_batch_id IN NUMBER);
    PROCEDURE check_required_fields(p_batch_id IN NUMBER);
    PROCEDURE check_value_ranges(p_batch_id IN NUMBER);
    PROCEDURE check_cross_table_consistency(p_batch_id IN NUMBER);
    PROCEDURE check_format_validations(p_batch_id IN NUMBER);

    -- Utility: Log a validation result
    PROCEDURE log_validation(
        p_batch_id    IN NUMBER,
        p_rule_name   IN VARCHAR2,
        p_rule_type   IN VARCHAR2,
        p_table_name  IN VARCHAR2,
        p_column_name IN VARCHAR2,
        p_total_rows  IN NUMBER,
        p_failed_rows IN NUMBER,
        p_sql_text    IN VARCHAR2     DEFAULT NULL
    );

    -- Compute overall quality score for a batch
    FUNCTION compute_quality_score(
        p_batch_id IN NUMBER
    ) RETURN NUMBER;

END pkg_data_quality;
/


CREATE OR REPLACE PACKAGE BODY pkg_data_quality AS

    -- Quality thresholds
    c_warning_threshold  CONSTANT NUMBER := 95.0;   -- Below 95% = WARNING
    c_failure_threshold  CONSTANT NUMBER := 90.0;   -- Below 90% = FAIL

    -- -----------------------------------------------------------------------
    -- log_validation: Centralized logging of validation results
    -- -----------------------------------------------------------------------
    PROCEDURE log_validation(
        p_batch_id    IN NUMBER,
        p_rule_name   IN VARCHAR2,
        p_rule_type   IN VARCHAR2,
        p_table_name  IN VARCHAR2,
        p_column_name IN VARCHAR2,
        p_total_rows  IN NUMBER,
        p_failed_rows IN NUMBER,
        p_sql_text    IN VARCHAR2     DEFAULT NULL
    ) IS
        v_passed_rows  NUMBER;
        v_pass_rate    NUMBER(5,2);
        v_status       VARCHAR2(20);
        v_quality_score NUMBER(5,2);
    BEGIN
        v_passed_rows := p_total_rows - p_failed_rows;

        IF p_total_rows > 0 THEN
            v_pass_rate := ROUND((v_passed_rows / p_total_rows) * 100, 2);
        ELSE
            v_pass_rate := 100.00;  -- No rows = vacuously true
        END IF;

        -- Determine status based on thresholds
        IF v_pass_rate >= c_warning_threshold THEN
            v_status := 'PASS';
            v_quality_score := v_pass_rate;
        ELSIF v_pass_rate >= c_failure_threshold THEN
            v_status := 'WARNING';
            v_quality_score := v_pass_rate;
        ELSE
            v_status := 'FAIL';
            v_quality_score := v_pass_rate;
        END IF;

        INSERT INTO dq_validation_result (
            batch_id, rule_name, rule_type, table_name, column_name,
            total_rows, passed_rows, failed_rows, pass_rate,
            quality_score, validation_sql, status
        ) VALUES (
            p_batch_id, p_rule_name, p_rule_type, p_table_name, p_column_name,
            p_total_rows, v_passed_rows, p_failed_rows, v_pass_rate,
            v_quality_score, SUBSTR(p_sql_text, 1, 4000), v_status
        );

        -- Log errors for FAIL results
        IF v_status = 'FAIL' THEN
            INSERT INTO etl_error_log (
                batch_id, mapping_name, error_code, error_message,
                error_severity, source_table
            ) VALUES (
                p_batch_id, 'DQ_CHECK', 'DQ_' || p_rule_type,
                p_rule_name || ': ' || p_failed_rows || ' of ' ||
                p_total_rows || ' rows failed (' || v_pass_rate || '%)',
                'WARNING', p_table_name
            );
        END IF;

    END log_validation;


    -- -----------------------------------------------------------------------
    -- check_referential_integrity: FK relationships between tables
    -- -----------------------------------------------------------------------
    PROCEDURE check_referential_integrity(p_batch_id IN NUMBER) IS
        v_total   NUMBER;
        v_orphans NUMBER;
    BEGIN
        -- Check 1: fact_adverse_event.drug_key -> dim_drug.drug_key
        SELECT COUNT(*) INTO v_total
          FROM fact_adverse_event
         WHERE load_batch_id = p_batch_id;

        SELECT COUNT(*) INTO v_orphans
          FROM fact_adverse_event f
         WHERE f.load_batch_id = p_batch_id
           AND f.drug_key IS NOT NULL
           AND NOT EXISTS (
               SELECT 1 FROM dim_drug d WHERE d.drug_key = f.drug_key
           );

        log_validation(
            p_batch_id    => p_batch_id,
            p_rule_name   => 'RI_FACT_AE_DIM_DRUG',
            p_rule_type   => 'REFERENTIAL',
            p_table_name  => 'fact_adverse_event',
            p_column_name => 'drug_key',
            p_total_rows  => v_total,
            p_failed_rows => v_orphans,
            p_sql_text    => 'Orphan drug_key in fact_adverse_event'
        );

        -- Check 2: fact_clinical_trial.trial_key -> dim_trial.trial_key
        SELECT COUNT(*) INTO v_total
          FROM fact_clinical_trial
         WHERE load_batch_id = p_batch_id;

        SELECT COUNT(*) INTO v_orphans
          FROM fact_clinical_trial f
         WHERE f.load_batch_id = p_batch_id
           AND f.trial_key IS NOT NULL
           AND NOT EXISTS (
               SELECT 1 FROM dim_trial d WHERE d.trial_key = f.trial_key
           );

        log_validation(
            p_batch_id    => p_batch_id,
            p_rule_name   => 'RI_FACT_TRIAL_DIM_TRIAL',
            p_rule_type   => 'REFERENTIAL',
            p_table_name  => 'fact_clinical_trial',
            p_column_name => 'trial_key',
            p_total_rows  => v_total,
            p_failed_rows => v_orphans,
            p_sql_text    => 'Orphan trial_key in fact_clinical_trial'
        );

        -- Check 3: fact_clinical_trial.drug_key -> dim_drug.drug_key
        SELECT COUNT(*) INTO v_orphans
          FROM fact_clinical_trial f
         WHERE f.load_batch_id = p_batch_id
           AND f.drug_key IS NOT NULL
           AND NOT EXISTS (
               SELECT 1 FROM dim_drug d WHERE d.drug_key = f.drug_key
           );

        log_validation(
            p_batch_id    => p_batch_id,
            p_rule_name   => 'RI_FACT_TRIAL_DIM_DRUG',
            p_rule_type   => 'REFERENTIAL',
            p_table_name  => 'fact_clinical_trial',
            p_column_name => 'drug_key',
            p_total_rows  => v_total,
            p_failed_rows => v_orphans,
            p_sql_text    => 'Orphan drug_key in fact_clinical_trial'
        );

        COMMIT;
    END check_referential_integrity;


    -- -----------------------------------------------------------------------
    -- check_required_fields: NOT NULL validation on business-critical columns
    -- -----------------------------------------------------------------------
    PROCEDURE check_required_fields(p_batch_id IN NUMBER) IS
        v_total  NUMBER;
        v_nulls  NUMBER;

        -- Table of (table_name, column_name) pairs to check
        TYPE t_col_check IS RECORD (
            tbl VARCHAR2(100),
            col VARCHAR2(100)
        );
        TYPE t_col_list IS TABLE OF t_col_check;
        v_checks t_col_list := t_col_list();

    BEGIN
        -- Drug dimension required fields
        SELECT COUNT(*) INTO v_total
          FROM dim_drug WHERE load_batch_id = p_batch_id;

        -- ndc_code
        SELECT COUNT(*) INTO v_nulls
          FROM dim_drug
         WHERE load_batch_id = p_batch_id
           AND (ndc_code IS NULL OR TRIM(ndc_code) = '');

        log_validation(p_batch_id, 'NULL_DIM_DRUG_NDC', 'NULL_CHECK',
                        'dim_drug', 'ndc_code', v_total, v_nulls);

        -- drug_name
        SELECT COUNT(*) INTO v_nulls
          FROM dim_drug
         WHERE load_batch_id = p_batch_id
           AND (drug_name IS NULL OR TRIM(drug_name) = '');

        log_validation(p_batch_id, 'NULL_DIM_DRUG_NAME', 'NULL_CHECK',
                        'dim_drug', 'drug_name', v_total, v_nulls);

        -- record_hash
        SELECT COUNT(*) INTO v_nulls
          FROM dim_drug
         WHERE load_batch_id = p_batch_id
           AND record_hash IS NULL;

        log_validation(p_batch_id, 'NULL_DIM_DRUG_HASH', 'NULL_CHECK',
                        'dim_drug', 'record_hash', v_total, v_nulls);

        -- Adverse event required fields
        SELECT COUNT(*) INTO v_total
          FROM ods_adverse_event
         WHERE load_batch_id = p_batch_id;

        SELECT COUNT(*) INTO v_nulls
          FROM ods_adverse_event
         WHERE load_batch_id = p_batch_id
           AND (primaryid IS NULL OR TRIM(primaryid) = '');

        log_validation(p_batch_id, 'NULL_AE_PRIMARYID', 'NULL_CHECK',
                        'ods_adverse_event', 'primaryid', v_total, v_nulls);

        SELECT COUNT(*) INTO v_nulls
          FROM ods_adverse_event
         WHERE load_batch_id = p_batch_id
           AND (pt_std IS NULL OR TRIM(pt_std) = '');

        log_validation(p_batch_id, 'NULL_AE_PT_STD', 'NULL_CHECK',
                        'ods_adverse_event', 'pt_std', v_total, v_nulls);

        SELECT COUNT(*) INTO v_nulls
          FROM ods_adverse_event
         WHERE load_batch_id = p_batch_id
           AND (drugname_std IS NULL OR TRIM(drugname_std) = '');

        log_validation(p_batch_id, 'NULL_AE_DRUGNAME', 'NULL_CHECK',
                        'ods_adverse_event', 'drugname_std', v_total, v_nulls);

        -- Clinical trial required fields
        SELECT COUNT(*) INTO v_total
          FROM ods_clinical_trial
         WHERE load_batch_id = p_batch_id;

        SELECT COUNT(*) INTO v_nulls
          FROM ods_clinical_trial
         WHERE load_batch_id = p_batch_id
           AND (nct_id IS NULL OR TRIM(nct_id) = '');

        log_validation(p_batch_id, 'NULL_TRIAL_NCT_ID', 'NULL_CHECK',
                        'ods_clinical_trial', 'nct_id', v_total, v_nulls);

        COMMIT;
    END check_required_fields;


    -- -----------------------------------------------------------------------
    -- check_value_ranges: Validate numeric/date ranges
    -- -----------------------------------------------------------------------
    PROCEDURE check_value_ranges(p_batch_id IN NUMBER) IS
        v_total    NUMBER;
        v_failures NUMBER;
    BEGIN
        -- SCD dates: effective_start_date must be <= effective_end_date
        SELECT COUNT(*) INTO v_total
          FROM dim_drug WHERE load_batch_id = p_batch_id;

        SELECT COUNT(*) INTO v_failures
          FROM dim_drug
         WHERE load_batch_id = p_batch_id
           AND effective_start_date > effective_end_date;

        log_validation(p_batch_id, 'RANGE_DRUG_SCD_DATES', 'RANGE',
                        'dim_drug', 'effective_start_date/end_date',
                        v_total, v_failures,
                        'start_date > end_date');

        -- Drug current_flag must be Y or N
        SELECT COUNT(*) INTO v_failures
          FROM dim_drug
         WHERE load_batch_id = p_batch_id
           AND current_flag NOT IN ('Y', 'N');

        log_validation(p_batch_id, 'RANGE_DRUG_CURRENT_FLAG', 'RANGE',
                        'dim_drug', 'current_flag', v_total, v_failures);

        -- DEA schedule must be valid (1-5, or empty)
        SELECT COUNT(*) INTO v_failures
          FROM dim_drug
         WHERE load_batch_id = p_batch_id
           AND dea_schedule IS NOT NULL
           AND dea_schedule NOT IN ('I', 'II', 'III', 'IV', 'V', '1', '2', '3', '4', '5', '');

        log_validation(p_batch_id, 'RANGE_DRUG_DEA_SCHEDULE', 'RANGE',
                        'dim_drug', 'dea_schedule', v_total, v_failures);

        -- AE: PRR should be >= 0
        SELECT COUNT(*) INTO v_total
          FROM ods_safety_signal WHERE load_batch_id = p_batch_id;

        SELECT COUNT(*) INTO v_failures
          FROM ods_safety_signal
         WHERE load_batch_id = p_batch_id
           AND prr < 0;

        log_validation(p_batch_id, 'RANGE_SIGNAL_PRR', 'RANGE',
                        'ods_safety_signal', 'prr', v_total, v_failures,
                        'PRR must be >= 0');

        -- Case count must be >= 1
        SELECT COUNT(*) INTO v_failures
          FROM ods_safety_signal
         WHERE load_batch_id = p_batch_id
           AND case_count < 1;

        log_validation(p_batch_id, 'RANGE_SIGNAL_CASE_COUNT', 'RANGE',
                        'ods_safety_signal', 'case_count', v_total, v_failures);

        -- Clinical trial enrollment must be >= 0
        SELECT COUNT(*) INTO v_total
          FROM ods_clinical_trial WHERE load_batch_id = p_batch_id;

        SELECT COUNT(*) INTO v_failures
          FROM ods_clinical_trial
         WHERE load_batch_id = p_batch_id
           AND enrollment < 0;

        log_validation(p_batch_id, 'RANGE_TRIAL_ENROLLMENT', 'RANGE',
                        'ods_clinical_trial', 'enrollment', v_total, v_failures);

        -- Trial duration must be >= 0
        SELECT COUNT(*) INTO v_failures
          FROM ods_clinical_trial
         WHERE load_batch_id = p_batch_id
           AND trial_duration_days < 0;

        log_validation(p_batch_id, 'RANGE_TRIAL_DURATION', 'RANGE',
                        'ods_clinical_trial', 'trial_duration_days',
                        v_total, v_failures);

        COMMIT;
    END check_value_ranges;


    -- -----------------------------------------------------------------------
    -- check_cross_table_consistency: Business rules spanning tables
    -- -----------------------------------------------------------------------
    PROCEDURE check_cross_table_consistency(p_batch_id IN NUMBER) IS
        v_total    NUMBER;
        v_failures NUMBER;
    BEGIN
        -- Rule 1: Every NDC in staging should appear in dim_drug after load
        SELECT COUNT(DISTINCT ndc_code) INTO v_total
          FROM stg_drug_master
         WHERE load_batch_id = p_batch_id
           AND record_status = 'PROCESSED';

        SELECT COUNT(DISTINCT s.ndc_code) INTO v_failures
          FROM stg_drug_master s
         WHERE s.load_batch_id = p_batch_id
           AND s.record_status = 'PROCESSED'
           AND NOT EXISTS (
               SELECT 1 FROM dim_drug d
                WHERE d.ndc_code = s.ndc_code
                  AND d.current_flag = 'Y'
           );

        log_validation(p_batch_id, 'CONSISTENCY_STG_DIM_DRUG', 'CONSISTENCY',
                        'stg_drug_master/dim_drug', 'ndc_code',
                        v_total, v_failures,
                        'Staged NDCs not found in dim_drug after load');

        -- Rule 2: No duplicate current records per business key in dim_drug
        SELECT COUNT(*) INTO v_total
          FROM dim_drug WHERE current_flag = 'Y';

        SELECT SUM(cnt - 1) INTO v_failures
          FROM (
              SELECT ndc_code, COUNT(*) AS cnt
                FROM dim_drug
               WHERE current_flag = 'Y'
               GROUP BY ndc_code
              HAVING COUNT(*) > 1
          );
        v_failures := NVL(v_failures, 0);

        log_validation(p_batch_id, 'CONSISTENCY_DIM_DRUG_UNIQUE_CURRENT',
                        'CONSISTENCY', 'dim_drug', 'ndc_code/current_flag',
                        v_total, v_failures,
                        'Multiple current records for same NDC');

        -- Rule 3: Row count reconciliation (source vs target)
        DECLARE
            v_source_count NUMBER;
            v_target_count NUMBER;
            v_diff         NUMBER;
        BEGIN
            SELECT COUNT(*) INTO v_source_count
              FROM stg_drug_master
             WHERE load_batch_id = p_batch_id
               AND record_status = 'PROCESSED'
               AND dq_error_code IS NULL;

            -- Target should have at least source_count new/updated records
            SELECT COUNT(*) INTO v_target_count
              FROM dim_drug
             WHERE load_batch_id = p_batch_id;

            v_diff := ABS(v_source_count - v_target_count);

            log_validation(p_batch_id, 'CONSISTENCY_ROW_RECONCILIATION',
                            'CONSISTENCY', 'stg_drug_master/dim_drug',
                            'row_count', v_source_count, v_diff,
                            'Source=' || v_source_count ||
                            ' Target=' || v_target_count);
        END;

        -- Rule 4: Safety signals must have matching AE records
        SELECT COUNT(*) INTO v_total
          FROM ods_safety_signal
         WHERE load_batch_id = p_batch_id;

        SELECT COUNT(*) INTO v_failures
          FROM ods_safety_signal s
         WHERE s.load_batch_id = p_batch_id
           AND NOT EXISTS (
               SELECT 1 FROM ods_adverse_event a
                WHERE a.drugname_std = s.drugname_std
                  AND a.pt_std = s.pt_std
           );

        log_validation(p_batch_id, 'CONSISTENCY_SIGNAL_AE', 'CONSISTENCY',
                        'ods_safety_signal', 'drugname_std/pt_std',
                        v_total, v_failures,
                        'Safety signals without matching AE records');

        COMMIT;
    END check_cross_table_consistency;


    -- -----------------------------------------------------------------------
    -- check_format_validations: Data format checks
    -- -----------------------------------------------------------------------
    PROCEDURE check_format_validations(p_batch_id IN NUMBER) IS
        v_total    NUMBER;
        v_failures NUMBER;
    BEGIN
        -- NDC format: should match pattern NNNNN-NNNN-NN or similar
        SELECT COUNT(*) INTO v_total
          FROM dim_drug WHERE load_batch_id = p_batch_id;

        SELECT COUNT(*) INTO v_failures
          FROM dim_drug
         WHERE load_batch_id = p_batch_id
           AND ndc_code IS NOT NULL
           AND NOT REGEXP_LIKE(ndc_code, '^\d{4,5}-\d{3,4}-\d{2}$');

        log_validation(p_batch_id, 'FORMAT_NDC_CODE', 'FORMAT',
                        'dim_drug', 'ndc_code', v_total, v_failures,
                        'NDC must match pattern NNNNN-NNNN-NN');

        -- NCT ID format: NCTxxxxxxxx
        SELECT COUNT(*) INTO v_total
          FROM ods_clinical_trial WHERE load_batch_id = p_batch_id;

        SELECT COUNT(*) INTO v_failures
          FROM ods_clinical_trial
         WHERE load_batch_id = p_batch_id
           AND nct_id IS NOT NULL
           AND NOT REGEXP_LIKE(nct_id, '^NCT\d{8}$');

        log_validation(p_batch_id, 'FORMAT_NCT_ID', 'FORMAT',
                        'ods_clinical_trial', 'nct_id', v_total, v_failures,
                        'NCT ID must match pattern NCTxxxxxxxx');

        -- FAERS primaryid format (numeric string)
        SELECT COUNT(*) INTO v_total
          FROM ods_adverse_event WHERE load_batch_id = p_batch_id;

        SELECT COUNT(*) INTO v_failures
          FROM ods_adverse_event
         WHERE load_batch_id = p_batch_id
           AND primaryid IS NOT NULL
           AND NOT REGEXP_LIKE(primaryid, '^\d+$');

        log_validation(p_batch_id, 'FORMAT_FAERS_PRIMARYID', 'FORMAT',
                        'ods_adverse_event', 'primaryid', v_total, v_failures);

        COMMIT;
    END check_format_validations;


    -- -----------------------------------------------------------------------
    -- compute_quality_score: Weighted average of all checks for a batch
    -- -----------------------------------------------------------------------
    FUNCTION compute_quality_score(
        p_batch_id IN NUMBER
    ) RETURN NUMBER IS
        v_score NUMBER;
    BEGIN
        SELECT NVL(ROUND(AVG(quality_score), 2), 100.0)
          INTO v_score
          FROM dq_validation_result
         WHERE batch_id = p_batch_id
           AND total_rows > 0;  -- Exclude vacuous checks

        RETURN v_score;
    END compute_quality_score;


    -- -----------------------------------------------------------------------
    -- run_all_checks: Master procedure to run all DQ validations
    -- -----------------------------------------------------------------------
    PROCEDURE run_all_checks(
        p_batch_id      IN  NUMBER,
        p_target_table  IN  VARCHAR2     DEFAULT NULL,
        p_overall_score OUT NUMBER,
        p_total_rules   OUT NUMBER,
        p_passed_rules  OUT NUMBER,
        p_failed_rules  OUT NUMBER,
        p_return_code   OUT NUMBER,
        p_return_msg    OUT VARCHAR2
    ) IS
    BEGIN
        p_return_code := 0;
        p_return_msg  := 'SUCCESS';

        -- Run all check categories
        check_referential_integrity(p_batch_id);
        check_required_fields(p_batch_id);
        check_value_ranges(p_batch_id);
        check_cross_table_consistency(p_batch_id);
        check_format_validations(p_batch_id);

        -- Compute summary metrics
        SELECT COUNT(*),
               SUM(CASE WHEN status = 'PASS' THEN 1 ELSE 0 END),
               SUM(CASE WHEN status = 'FAIL' THEN 1 ELSE 0 END)
          INTO p_total_rules, p_passed_rules, p_failed_rules
          FROM dq_validation_result
         WHERE batch_id = p_batch_id;

        p_overall_score := compute_quality_score(p_batch_id);

        -- Log summary to mapping run log
        INSERT INTO etl_mapping_run_log (
            batch_id, mapping_name, session_name, run_status,
            start_timestamp, end_timestamp,
            source_rows_read, target_rows_written
        ) VALUES (
            p_batch_id, 'DQ_VALIDATION', 's_data_quality_checks',
            CASE WHEN p_failed_rules > 0 THEN 'FAILED' ELSE 'SUCCEEDED' END,
            SYSTIMESTAMP, SYSTIMESTAMP,
            p_total_rules, p_passed_rules
        );

        COMMIT;

    EXCEPTION
        WHEN OTHERS THEN
            ROLLBACK;
            p_return_code := SQLCODE;
            p_return_msg  := SQLERRM;

            INSERT INTO etl_error_log (
                batch_id, mapping_name, error_code, error_message, error_severity
            ) VALUES (
                p_batch_id, 'DQ_VALIDATION',
                TO_CHAR(SQLCODE), SQLERRM, 'FATAL'
            );
            COMMIT;
    END run_all_checks;

END pkg_data_quality;
/
