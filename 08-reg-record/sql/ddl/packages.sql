--------------------------------------------------------------------------------
-- RegRecord - PL/SQL Packages
--
-- PKG_SUBMISSION_MGMT  : Submission lifecycle management
-- PKG_PSEUDO_RECORD    : Pseudo record keeping operations
-- PKG_COMPLIANCE       : Compliance tracking and reporting
-- PKG_AUDIT            : Audit trail management and reporting
--------------------------------------------------------------------------------

-- ============================================================================
-- PKG_SUBMISSION_MGMT - Submission Management Package
-- ============================================================================

CREATE OR REPLACE PACKAGE PKG_SUBMISSION_MGMT AS
    -- Types
    TYPE t_submission_rec IS RECORD (
        id                  REGULATORY_SUBMISSION.id%TYPE,
        drug_id             REGULATORY_SUBMISSION.drug_id%TYPE,
        submission_type     REGULATORY_SUBMISSION.submission_type%TYPE,
        status              REGULATORY_SUBMISSION.status%TYPE,
        agency              REGULATORY_SUBMISSION.agency%TYPE,
        tracking_number     REGULATORY_SUBMISSION.tracking_number%TYPE,
        submitted_date      REGULATORY_SUBMISSION.submitted_date%TYPE
    );

    TYPE t_submission_cur IS REF CURSOR RETURN t_submission_rec;
    TYPE t_submission_tab IS TABLE OF t_submission_rec INDEX BY PLS_INTEGER;

    -- Custom exceptions
    e_invalid_drug      EXCEPTION;
    e_invalid_agency    EXCEPTION;
    e_duplicate_tracking EXCEPTION;
    e_invalid_transition EXCEPTION;

    PRAGMA EXCEPTION_INIT(e_invalid_drug, -20010);
    PRAGMA EXCEPTION_INIT(e_invalid_agency, -20011);
    PRAGMA EXCEPTION_INIT(e_duplicate_tracking, -20012);
    PRAGMA EXCEPTION_INIT(e_invalid_transition, -20001);

    -- Procedures
    PROCEDURE create_submission(
        p_drug_id           IN  NUMBER,
        p_submission_type   IN  VARCHAR2,
        p_agency            IN  VARCHAR2,
        p_priority          IN  VARCHAR2 DEFAULT 'STANDARD',
        p_assigned_to       IN  VARCHAR2 DEFAULT NULL,
        p_description       IN  CLOB DEFAULT NULL,
        p_created_by        IN  VARCHAR2,
        p_submission_id     OUT NUMBER,
        p_tracking_number   OUT VARCHAR2
    );

    PROCEDURE update_status(
        p_submission_id     IN  NUMBER,
        p_new_status        IN  VARCHAR2,
        p_modified_by       IN  VARCHAR2,
        p_notes             IN  VARCHAR2 DEFAULT NULL
    );

    PROCEDURE add_document(
        p_submission_id     IN  NUMBER,
        p_doc_type          IN  VARCHAR2,
        p_doc_title         IN  VARCHAR2,
        p_file_path         IN  VARCHAR2,
        p_checksum          IN  VARCHAR2,
        p_uploaded_by       IN  VARCHAR2,
        p_confidentiality   IN  VARCHAR2 DEFAULT 'CONFIDENTIAL',
        p_document_id       OUT NUMBER
    );

    PROCEDURE get_submission_history(
        p_submission_id     IN  NUMBER,
        p_cursor            OUT SYS_REFCURSOR
    );

    FUNCTION generate_tracking_number(
        p_agency            IN  VARCHAR2,
        p_submission_type   IN  VARCHAR2,
        p_year              IN  NUMBER DEFAULT NULL
    ) RETURN VARCHAR2;

    FUNCTION get_submission_count_by_status(
        p_agency            IN  VARCHAR2 DEFAULT NULL,
        p_status            IN  VARCHAR2 DEFAULT NULL
    ) RETURN NUMBER;

END PKG_SUBMISSION_MGMT;
/

CREATE OR REPLACE PACKAGE BODY PKG_SUBMISSION_MGMT AS

    -- Private: Validate drug exists and is active
    PROCEDURE validate_drug(p_drug_id IN NUMBER) IS
        v_count NUMBER;
    BEGIN
        SELECT COUNT(*) INTO v_count
        FROM DRUG
        WHERE drug_id = p_drug_id AND active_flag = 'Y';

        IF v_count = 0 THEN
            RAISE_APPLICATION_ERROR(-20010,
                'Drug ID ' || p_drug_id || ' does not exist or is inactive');
        END IF;
    END validate_drug;

    -- Private: Validate agency exists
    PROCEDURE validate_agency(p_agency IN VARCHAR2) IS
        v_count NUMBER;
    BEGIN
        SELECT COUNT(*) INTO v_count
        FROM AGENCY
        WHERE agency_code = p_agency AND active_flag = 'Y';

        IF v_count = 0 THEN
            RAISE_APPLICATION_ERROR(-20011,
                'Agency code ' || p_agency || ' is not valid or inactive');
        END IF;
    END validate_agency;

    -- Generate a unique tracking number
    FUNCTION generate_tracking_number(
        p_agency            IN  VARCHAR2,
        p_submission_type   IN  VARCHAR2,
        p_year              IN  NUMBER DEFAULT NULL
    ) RETURN VARCHAR2 IS
        v_year      NUMBER := NVL(p_year, EXTRACT(YEAR FROM SYSDATE));
        v_seq       NUMBER;
        v_tracking  VARCHAR2(100);
    BEGIN
        SELECT seq_submission_id.NEXTVAL INTO v_seq FROM DUAL;
        v_tracking := p_agency || '-' ||
                      SUBSTR(p_submission_type, 1, 3) || '-' ||
                      TO_CHAR(v_year) || '-' ||
                      LPAD(TO_CHAR(v_seq), 6, '0');
        RETURN v_tracking;
    END generate_tracking_number;

    -- Create a new regulatory submission
    PROCEDURE create_submission(
        p_drug_id           IN  NUMBER,
        p_submission_type   IN  VARCHAR2,
        p_agency            IN  VARCHAR2,
        p_priority          IN  VARCHAR2 DEFAULT 'STANDARD',
        p_assigned_to       IN  VARCHAR2 DEFAULT NULL,
        p_description       IN  CLOB DEFAULT NULL,
        p_created_by        IN  VARCHAR2,
        p_submission_id     OUT NUMBER,
        p_tracking_number   OUT VARCHAR2
    ) IS
    BEGIN
        -- Validate inputs
        validate_drug(p_drug_id);
        validate_agency(p_agency);

        -- Generate tracking number
        p_tracking_number := generate_tracking_number(p_agency, p_submission_type);

        -- Insert the submission
        INSERT INTO REGULATORY_SUBMISSION (
            drug_id, submission_type, status, agency,
            tracking_number, priority, assigned_to,
            description, created_by, modified_by
        ) VALUES (
            p_drug_id, p_submission_type, 'DRAFT', p_agency,
            p_tracking_number, p_priority, p_assigned_to,
            p_description, p_created_by, p_created_by
        ) RETURNING id INTO p_submission_id;

        COMMIT;

        DBMS_OUTPUT.PUT_LINE('Submission created: ID=' || p_submission_id ||
                             ', Tracking=' || p_tracking_number);

    EXCEPTION
        WHEN DUP_VAL_ON_INDEX THEN
            ROLLBACK;
            RAISE_APPLICATION_ERROR(-20012,
                'Duplicate tracking number generated. Please retry.');
        WHEN OTHERS THEN
            ROLLBACK;
            RAISE;
    END create_submission;

    -- Update submission status with state machine enforcement
    PROCEDURE update_status(
        p_submission_id     IN  NUMBER,
        p_new_status        IN  VARCHAR2,
        p_modified_by       IN  VARCHAR2,
        p_notes             IN  VARCHAR2 DEFAULT NULL
    ) IS
        v_current_status    VARCHAR2(30);
    BEGIN
        -- Lock the row for update
        SELECT status INTO v_current_status
        FROM REGULATORY_SUBMISSION
        WHERE id = p_submission_id
        FOR UPDATE NOWAIT;

        -- The trigger trg_submission_status_validate will enforce the state machine
        UPDATE REGULATORY_SUBMISSION
        SET status = p_new_status,
            modified_by = p_modified_by,
            modified_date = SYSTIMESTAMP
        WHERE id = p_submission_id;

        COMMIT;

        DBMS_OUTPUT.PUT_LINE('Submission ' || p_submission_id ||
                             ' status updated: ' || v_current_status || ' -> ' || p_new_status);

    EXCEPTION
        WHEN NO_DATA_FOUND THEN
            RAISE_APPLICATION_ERROR(-20013,
                'Submission ID ' || p_submission_id || ' not found');
        WHEN OTHERS THEN
            ROLLBACK;
            RAISE;
    END update_status;

    -- Add a document to a submission
    PROCEDURE add_document(
        p_submission_id     IN  NUMBER,
        p_doc_type          IN  VARCHAR2,
        p_doc_title         IN  VARCHAR2,
        p_file_path         IN  VARCHAR2,
        p_checksum          IN  VARCHAR2,
        p_uploaded_by       IN  VARCHAR2,
        p_confidentiality   IN  VARCHAR2 DEFAULT 'CONFIDENTIAL',
        p_document_id       OUT NUMBER
    ) IS
        v_sub_exists    NUMBER;
    BEGIN
        -- Verify submission exists
        SELECT COUNT(*) INTO v_sub_exists
        FROM REGULATORY_SUBMISSION
        WHERE id = p_submission_id;

        IF v_sub_exists = 0 THEN
            RAISE_APPLICATION_ERROR(-20013,
                'Submission ID ' || p_submission_id || ' not found');
        END IF;

        -- Insert document (version auto-increment handled by trigger)
        INSERT INTO SUBMISSION_DOCUMENT (
            submission_id, doc_type, doc_title, file_path,
            checksum, uploaded_by, confidentiality
        ) VALUES (
            p_submission_id, p_doc_type, p_doc_title, p_file_path,
            p_checksum, p_uploaded_by, p_confidentiality
        ) RETURNING id INTO p_document_id;

        COMMIT;

        DBMS_OUTPUT.PUT_LINE('Document added: ID=' || p_document_id ||
                             ' to Submission=' || p_submission_id);

    EXCEPTION
        WHEN OTHERS THEN
            ROLLBACK;
            RAISE;
    END add_document;

    -- Get complete submission history from audit trail
    PROCEDURE get_submission_history(
        p_submission_id     IN  NUMBER,
        p_cursor            OUT SYS_REFCURSOR
    ) IS
    BEGIN
        OPEN p_cursor FOR
            SELECT
                at.id           AS audit_id,
                at.action,
                at.old_values,
                at.new_values,
                at.changed_by,
                at.changed_at,
                at.ip_address
            FROM AUDIT_TRAIL at
            WHERE at.table_name = 'REGULATORY_SUBMISSION'
              AND at.record_id = p_submission_id
            ORDER BY at.changed_at DESC;
    END get_submission_history;

    -- Get count of submissions by status/agency
    FUNCTION get_submission_count_by_status(
        p_agency            IN  VARCHAR2 DEFAULT NULL,
        p_status            IN  VARCHAR2 DEFAULT NULL
    ) RETURN NUMBER IS
        v_count NUMBER;
    BEGIN
        SELECT COUNT(*)
        INTO v_count
        FROM REGULATORY_SUBMISSION
        WHERE (p_agency IS NULL OR agency = p_agency)
          AND (p_status IS NULL OR status = p_status);
        RETURN v_count;
    END get_submission_count_by_status;

END PKG_SUBMISSION_MGMT;
/

-- ============================================================================
-- PKG_PSEUDO_RECORD - Pseudo Record Management Package
-- ============================================================================

CREATE OR REPLACE PACKAGE PKG_PSEUDO_RECORD AS
    -- Types
    TYPE t_pseudo_mapping IS RECORD (
        pseudo_id       PSEUDO_RECORD.id%TYPE,
        pseudo_value    PSEUDO_RECORD.pseudo_value%TYPE,
        original_id     PSEUDO_RECORD.original_record_id%TYPE,
        source_table    PSEUDO_RECORD.source_table%TYPE,
        is_active       PSEUDO_RECORD.is_active%TYPE
    );

    TYPE t_pseudo_tab IS TABLE OF t_pseudo_mapping INDEX BY PLS_INTEGER;

    -- Custom exceptions
    e_mapping_exists    EXCEPTION;
    e_mapping_expired   EXCEPTION;
    e_invalid_key       EXCEPTION;
    e_unauthorized      EXCEPTION;

    PRAGMA EXCEPTION_INIT(e_mapping_exists, -20020);
    PRAGMA EXCEPTION_INIT(e_mapping_expired, -20021);
    PRAGMA EXCEPTION_INIT(e_invalid_key, -20022);
    PRAGMA EXCEPTION_INIT(e_unauthorized, -20023);

    -- Generate a pseudo ID for a given record
    FUNCTION generate_pseudo_id(
        p_original_id       IN  NUMBER,
        p_source_table      IN  VARCHAR2,
        p_pseudo_type       IN  VARCHAR2,
        p_mapping_key       IN  VARCHAR2,
        p_created_by        IN  VARCHAR2,
        p_purpose           IN  VARCHAR2 DEFAULT NULL,
        p_expiry_days       IN  NUMBER DEFAULT 365
    ) RETURN VARCHAR2;

    -- Map pseudo value back to real record (requires authorization)
    FUNCTION map_pseudo_to_real(
        p_pseudo_value      IN  VARCHAR2,
        p_requested_by      IN  VARCHAR2,
        p_authorized_by     IN  VARCHAR2,
        p_reason            IN  VARCHAR2
    ) RETURN NUMBER;

    -- Validate that a pseudo mapping is still active and valid
    FUNCTION validate_pseudo_mapping(
        p_pseudo_value      IN  VARCHAR2
    ) RETURN BOOLEAN;

    -- Purge expired mappings
    PROCEDURE purge_expired_mappings(
        p_purge_before      IN  TIMESTAMP DEFAULT SYSTIMESTAMP,
        p_purged_count      OUT NUMBER
    );

    -- Batch generate pseudo IDs
    PROCEDURE batch_generate_pseudo_ids(
        p_source_table      IN  VARCHAR2,
        p_pseudo_type       IN  VARCHAR2,
        p_mapping_key       IN  VARCHAR2,
        p_created_by        IN  VARCHAR2,
        p_purpose           IN  VARCHAR2 DEFAULT NULL,
        p_expiry_days       IN  NUMBER DEFAULT 365,
        p_generated_count   OUT NUMBER
    );

    -- Rotate mapping keys (re-generate pseudo IDs with new key)
    PROCEDURE rotate_mapping_key(
        p_old_key           IN  VARCHAR2,
        p_new_key           IN  VARCHAR2,
        p_source_table      IN  VARCHAR2,
        p_rotated_by        IN  VARCHAR2,
        p_rotated_count     OUT NUMBER
    );

END PKG_PSEUDO_RECORD;
/

CREATE OR REPLACE PACKAGE BODY PKG_PSEUDO_RECORD AS

    -- Private: Compute a deterministic hash for pseudo ID generation
    FUNCTION compute_hash(
        p_input     IN  VARCHAR2,
        p_salt      IN  VARCHAR2,
        p_key       IN  VARCHAR2
    ) RETURN VARCHAR2 IS
        v_raw_input RAW(2000);
        v_hash      RAW(32);
    BEGIN
        -- Combine input + salt + key for deterministic but secure hashing
        v_raw_input := UTL_RAW.CAST_TO_RAW(p_input || '|' || p_salt || '|' || p_key);
        v_hash := DBMS_CRYPTO.HASH(v_raw_input, DBMS_CRYPTO.HASH_SH256);
        RETURN RAWTOHEX(v_hash);
    END compute_hash;

    -- Private: Generate a random salt
    FUNCTION generate_salt RETURN VARCHAR2 IS
        v_raw RAW(32);
    BEGIN
        v_raw := DBMS_CRYPTO.RANDOMBYTES(32);
        RETURN RAWTOHEX(v_raw);
    END generate_salt;

    -- Private: Format pseudo value with prefix for readability
    FUNCTION format_pseudo_value(
        p_hash          IN  VARCHAR2,
        p_pseudo_type   IN  VARCHAR2
    ) RETURN VARCHAR2 IS
        v_prefix VARCHAR2(10);
    BEGIN
        CASE p_pseudo_type
            WHEN 'SUBMISSION_ID'    THEN v_prefix := 'PSB';
            WHEN 'DOCUMENT_ID'      THEN v_prefix := 'PDC';
            WHEN 'PATIENT_ID'       THEN v_prefix := 'PPT';
            WHEN 'INVESTIGATOR_ID'  THEN v_prefix := 'PIV';
            WHEN 'SITE_ID'          THEN v_prefix := 'PST';
            WHEN 'BATCH_NUMBER'     THEN v_prefix := 'PBN';
            WHEN 'COMPOUND_CODE'    THEN v_prefix := 'PCC';
            WHEN 'PROTOCOL_NUMBER'  THEN v_prefix := 'PPN';
            ELSE v_prefix := 'PXX';
        END CASE;

        -- Return prefix + first 32 chars of hash (128 bits, sufficient for uniqueness)
        RETURN v_prefix || '-' || SUBSTR(p_hash, 1, 32);
    END format_pseudo_value;

    -- Generate a pseudo ID for a given record
    FUNCTION generate_pseudo_id(
        p_original_id       IN  NUMBER,
        p_source_table      IN  VARCHAR2,
        p_pseudo_type       IN  VARCHAR2,
        p_mapping_key       IN  VARCHAR2,
        p_created_by        IN  VARCHAR2,
        p_purpose           IN  VARCHAR2 DEFAULT NULL,
        p_expiry_days       IN  NUMBER DEFAULT 365
    ) RETURN VARCHAR2 IS
        v_salt          VARCHAR2(64);
        v_hash          VARCHAR2(256);
        v_pseudo_value  VARCHAR2(256);
        v_existing      VARCHAR2(256);
        v_expiry        TIMESTAMP;
    BEGIN
        -- Check if mapping already exists
        BEGIN
            SELECT pseudo_value INTO v_existing
            FROM PSEUDO_RECORD
            WHERE original_record_id = p_original_id
              AND source_table = p_source_table
              AND pseudo_type = p_pseudo_type
              AND mapping_key = p_mapping_key
              AND is_active = 'Y';

            -- Mapping already exists, return existing pseudo value
            RETURN v_existing;
        EXCEPTION
            WHEN NO_DATA_FOUND THEN
                NULL;  -- Continue to create new mapping
        END;

        -- Generate salt and hash
        v_salt := generate_salt();
        v_hash := compute_hash(
            TO_CHAR(p_original_id) || '|' || p_source_table,
            v_salt,
            p_mapping_key
        );
        v_pseudo_value := format_pseudo_value(v_hash, p_pseudo_type);

        -- Calculate expiry
        IF p_expiry_days IS NOT NULL AND p_expiry_days > 0 THEN
            v_expiry := SYSTIMESTAMP + NUMTODSINTERVAL(p_expiry_days, 'DAY');
        END IF;

        -- Insert the mapping
        INSERT INTO PSEUDO_RECORD (
            original_record_id, source_table, pseudo_type,
            pseudo_value, mapping_key, hash_algorithm, salt,
            is_active, expiry_date, created_by, purpose
        ) VALUES (
            p_original_id, p_source_table, p_pseudo_type,
            v_pseudo_value, p_mapping_key, 'SHA-256', v_salt,
            'Y', v_expiry, p_created_by, p_purpose
        );

        COMMIT;
        RETURN v_pseudo_value;

    EXCEPTION
        WHEN DUP_VAL_ON_INDEX THEN
            ROLLBACK;
            RAISE_APPLICATION_ERROR(-20020,
                'Pseudo mapping already exists for this record/type/key combination');
        WHEN OTHERS THEN
            ROLLBACK;
            RAISE;
    END generate_pseudo_id;

    -- Map pseudo value back to real record (requires authorization)
    FUNCTION map_pseudo_to_real(
        p_pseudo_value      IN  VARCHAR2,
        p_requested_by      IN  VARCHAR2,
        p_authorized_by     IN  VARCHAR2,
        p_reason            IN  VARCHAR2
    ) RETURN NUMBER IS
        v_original_id       NUMBER;
        v_pseudo_record_id  NUMBER;
        v_is_active         CHAR(1);
        v_expiry            TIMESTAMP;
    BEGIN
        -- Authorization check: requested_by and authorized_by must differ
        IF p_requested_by = p_authorized_by THEN
            RAISE_APPLICATION_ERROR(-20023,
                'Re-identification requires authorization from a different user than the requester');
        END IF;

        -- Lookup the pseudo record
        BEGIN
            SELECT id, original_record_id, is_active, expiry_date
            INTO v_pseudo_record_id, v_original_id, v_is_active, v_expiry
            FROM PSEUDO_RECORD
            WHERE pseudo_value = p_pseudo_value;
        EXCEPTION
            WHEN NO_DATA_FOUND THEN
                RAISE_APPLICATION_ERROR(-20022,
                    'Pseudo value not found: ' || p_pseudo_value);
        END;

        -- Check if mapping is still active
        IF v_is_active = 'N' THEN
            RAISE_APPLICATION_ERROR(-20021,
                'Pseudo mapping has been deactivated');
        END IF;

        -- Check expiry
        IF v_expiry IS NOT NULL AND v_expiry < SYSTIMESTAMP THEN
            -- Mark as inactive
            UPDATE PSEUDO_RECORD SET is_active = 'N' WHERE id = v_pseudo_record_id;
            RAISE_APPLICATION_ERROR(-20021,
                'Pseudo mapping has expired as of ' || TO_CHAR(v_expiry, 'YYYY-MM-DD HH24:MI:SS'));
        END IF;

        -- Log the re-identification request
        INSERT INTO REIDENTIFICATION_LOG (
            pseudo_record_id, requested_by, authorized_by,
            request_reason, request_date, approval_date, status,
            ip_address
        ) VALUES (
            v_pseudo_record_id, p_requested_by, p_authorized_by,
            p_reason, SYSTIMESTAMP, SYSTIMESTAMP, 'APPROVED',
            SYS_CONTEXT('USERENV', 'IP_ADDRESS')
        );

        -- Update access tracking on pseudo record
        UPDATE PSEUDO_RECORD
        SET last_accessed = SYSTIMESTAMP,
            access_count = access_count + 1
        WHERE id = v_pseudo_record_id;

        COMMIT;
        RETURN v_original_id;

    EXCEPTION
        WHEN OTHERS THEN
            -- Log the failed attempt too
            BEGIN
                INSERT INTO REIDENTIFICATION_LOG (
                    pseudo_record_id, requested_by, authorized_by,
                    request_reason, request_date, status
                ) VALUES (
                    NVL(v_pseudo_record_id, -1), p_requested_by, p_authorized_by,
                    p_reason || ' [FAILED: ' || SQLERRM || ']', SYSTIMESTAMP, 'DENIED'
                );
                COMMIT;
            EXCEPTION
                WHEN OTHERS THEN NULL;  -- Don't let logging failure mask original error
            END;
            RAISE;
    END map_pseudo_to_real;

    -- Validate that a pseudo mapping is still active and valid
    FUNCTION validate_pseudo_mapping(
        p_pseudo_value      IN  VARCHAR2
    ) RETURN BOOLEAN IS
        v_is_active     CHAR(1);
        v_expiry        TIMESTAMP;
    BEGIN
        SELECT is_active, expiry_date
        INTO v_is_active, v_expiry
        FROM PSEUDO_RECORD
        WHERE pseudo_value = p_pseudo_value;

        IF v_is_active = 'N' THEN
            RETURN FALSE;
        END IF;

        IF v_expiry IS NOT NULL AND v_expiry < SYSTIMESTAMP THEN
            RETURN FALSE;
        END IF;

        RETURN TRUE;

    EXCEPTION
        WHEN NO_DATA_FOUND THEN
            RETURN FALSE;
    END validate_pseudo_mapping;

    -- Purge expired mappings
    PROCEDURE purge_expired_mappings(
        p_purge_before      IN  TIMESTAMP DEFAULT SYSTIMESTAMP,
        p_purged_count      OUT NUMBER
    ) IS
    BEGIN
        -- First deactivate expired records
        UPDATE PSEUDO_RECORD
        SET is_active = 'N'
        WHERE expiry_date < p_purge_before
          AND is_active = 'Y';

        -- Delete inactive records older than retention period (keep audit trail)
        DELETE FROM PSEUDO_RECORD
        WHERE is_active = 'N'
          AND expiry_date < p_purge_before - INTERVAL '90' DAY
          AND id NOT IN (
              SELECT pseudo_record_id FROM REIDENTIFICATION_LOG
          );

        p_purged_count := SQL%ROWCOUNT;
        COMMIT;

        DBMS_OUTPUT.PUT_LINE('Purged ' || p_purged_count || ' expired pseudo mappings');

    EXCEPTION
        WHEN OTHERS THEN
            ROLLBACK;
            RAISE;
    END purge_expired_mappings;

    -- Batch generate pseudo IDs for all records in a source table
    PROCEDURE batch_generate_pseudo_ids(
        p_source_table      IN  VARCHAR2,
        p_pseudo_type       IN  VARCHAR2,
        p_mapping_key       IN  VARCHAR2,
        p_created_by        IN  VARCHAR2,
        p_purpose           IN  VARCHAR2 DEFAULT NULL,
        p_expiry_days       IN  NUMBER DEFAULT 365,
        p_generated_count   OUT NUMBER
    ) IS
        v_sql           VARCHAR2(4000);
        v_cursor        SYS_REFCURSOR;
        v_record_id     NUMBER;
        v_pseudo_value  VARCHAR2(256);
        v_count         NUMBER := 0;
    BEGIN
        -- Build dynamic SQL to get all IDs from source table
        v_sql := 'SELECT id FROM ' || DBMS_ASSERT.SQL_OBJECT_NAME(p_source_table) ||
                 ' WHERE id NOT IN (' ||
                 '  SELECT original_record_id FROM PSEUDO_RECORD' ||
                 '  WHERE source_table = :1 AND pseudo_type = :2' ||
                 '    AND mapping_key = :3 AND is_active = ''Y'')';

        OPEN v_cursor FOR v_sql USING p_source_table, p_pseudo_type, p_mapping_key;

        LOOP
            FETCH v_cursor INTO v_record_id;
            EXIT WHEN v_cursor%NOTFOUND;

            v_pseudo_value := generate_pseudo_id(
                p_original_id   => v_record_id,
                p_source_table  => p_source_table,
                p_pseudo_type   => p_pseudo_type,
                p_mapping_key   => p_mapping_key,
                p_created_by    => p_created_by,
                p_purpose       => p_purpose,
                p_expiry_days   => p_expiry_days
            );

            v_count := v_count + 1;

            -- Commit in batches of 1000
            IF MOD(v_count, 1000) = 0 THEN
                COMMIT;
                DBMS_OUTPUT.PUT_LINE('Processed ' || v_count || ' records...');
            END IF;
        END LOOP;

        CLOSE v_cursor;
        COMMIT;

        p_generated_count := v_count;
        DBMS_OUTPUT.PUT_LINE('Batch generation complete: ' || v_count || ' pseudo IDs generated');

    EXCEPTION
        WHEN OTHERS THEN
            IF v_cursor%ISOPEN THEN CLOSE v_cursor; END IF;
            ROLLBACK;
            RAISE;
    END batch_generate_pseudo_ids;

    -- Rotate mapping keys
    PROCEDURE rotate_mapping_key(
        p_old_key           IN  VARCHAR2,
        p_new_key           IN  VARCHAR2,
        p_source_table      IN  VARCHAR2,
        p_rotated_by        IN  VARCHAR2,
        p_rotated_count     OUT NUMBER
    ) IS
        CURSOR c_old_mappings IS
            SELECT id, original_record_id, pseudo_type, purpose
            FROM PSEUDO_RECORD
            WHERE mapping_key = p_old_key
              AND source_table = p_source_table
              AND is_active = 'Y'
            FOR UPDATE;

        v_new_pseudo    VARCHAR2(256);
        v_count         NUMBER := 0;
    BEGIN
        FOR rec IN c_old_mappings LOOP
            -- Deactivate old mapping
            UPDATE PSEUDO_RECORD
            SET is_active = 'N',
                expiry_date = SYSTIMESTAMP
            WHERE id = rec.id;

            -- Generate new mapping with new key
            v_new_pseudo := generate_pseudo_id(
                p_original_id   => rec.original_record_id,
                p_source_table  => p_source_table,
                p_pseudo_type   => rec.pseudo_type,
                p_mapping_key   => p_new_key,
                p_created_by    => p_rotated_by,
                p_purpose       => 'Key rotation from ' || SUBSTR(p_old_key, 1, 8) || '...'
            );

            v_count := v_count + 1;
        END LOOP;

        COMMIT;
        p_rotated_count := v_count;

        DBMS_OUTPUT.PUT_LINE('Key rotation complete: ' || v_count ||
                             ' mappings rotated for table ' || p_source_table);

    EXCEPTION
        WHEN OTHERS THEN
            ROLLBACK;
            RAISE;
    END rotate_mapping_key;

END PKG_PSEUDO_RECORD;
/

-- ============================================================================
-- PKG_COMPLIANCE - Compliance Management Package
-- ============================================================================

CREATE OR REPLACE PACKAGE PKG_COMPLIANCE AS

    TYPE t_compliance_summary IS RECORD (
        total_records       NUMBER,
        pending_count       NUMBER,
        in_progress_count   NUMBER,
        overdue_count       NUMBER,
        escalated_count     NUMBER,
        completed_count     NUMBER,
        avg_risk_score      NUMBER
    );

    TYPE t_gap_record IS RECORD (
        drug_id             NUMBER,
        drug_name           VARCHAR2(200),
        requirement_type    VARCHAR2(50),
        gap_description     VARCHAR2(500),
        severity            VARCHAR2(20),
        days_overdue        NUMBER
    );

    TYPE t_gap_tab IS TABLE OF t_gap_record INDEX BY PLS_INTEGER;

    -- Check compliance status for all or specific drug
    PROCEDURE check_compliance_status(
        p_drug_id           IN  NUMBER DEFAULT NULL,
        p_summary           OUT t_compliance_summary
    );

    -- Generate compliance report as cursor
    PROCEDURE generate_compliance_report(
        p_drug_id           IN  NUMBER DEFAULT NULL,
        p_agency            IN  VARCHAR2 DEFAULT NULL,
        p_include_completed IN  CHAR DEFAULT 'N',
        p_cursor            OUT SYS_REFCURSOR
    );

    -- Escalate overdue compliance records
    PROCEDURE escalate_overdue(
        p_escalation_user   IN  VARCHAR2,
        p_escalated_count   OUT NUMBER
    );

    -- Calculate risk score for a compliance record
    FUNCTION calculate_risk_score(
        p_compliance_id     IN  NUMBER
    ) RETURN NUMBER;

    -- Get SLA status for a requirement
    FUNCTION get_sla_status(
        p_compliance_id     IN  NUMBER
    ) RETURN VARCHAR2;

END PKG_COMPLIANCE;
/

CREATE OR REPLACE PACKAGE BODY PKG_COMPLIANCE AS

    -- Check compliance status
    PROCEDURE check_compliance_status(
        p_drug_id           IN  NUMBER DEFAULT NULL,
        p_summary           OUT t_compliance_summary
    ) IS
    BEGIN
        SELECT
            COUNT(*),
            SUM(CASE WHEN status = 'PENDING' THEN 1 ELSE 0 END),
            SUM(CASE WHEN status = 'IN_PROGRESS' THEN 1 ELSE 0 END),
            SUM(CASE WHEN status = 'OVERDUE' THEN 1 ELSE 0 END),
            SUM(CASE WHEN status = 'ESCALATED' THEN 1 ELSE 0 END),
            SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END),
            ROUND(AVG(risk_score), 2)
        INTO
            p_summary.total_records,
            p_summary.pending_count,
            p_summary.in_progress_count,
            p_summary.overdue_count,
            p_summary.escalated_count,
            p_summary.completed_count,
            p_summary.avg_risk_score
        FROM COMPLIANCE_RECORD
        WHERE (p_drug_id IS NULL OR drug_id = p_drug_id)
          AND status NOT IN ('WAIVED', 'CANCELLED');

        DBMS_OUTPUT.PUT_LINE('Compliance Status Summary:');
        DBMS_OUTPUT.PUT_LINE('  Total: ' || p_summary.total_records);
        DBMS_OUTPUT.PUT_LINE('  Pending: ' || p_summary.pending_count);
        DBMS_OUTPUT.PUT_LINE('  In Progress: ' || p_summary.in_progress_count);
        DBMS_OUTPUT.PUT_LINE('  Overdue: ' || p_summary.overdue_count);
        DBMS_OUTPUT.PUT_LINE('  Escalated: ' || p_summary.escalated_count);
        DBMS_OUTPUT.PUT_LINE('  Completed: ' || p_summary.completed_count);
        DBMS_OUTPUT.PUT_LINE('  Avg Risk Score: ' || p_summary.avg_risk_score);
    END check_compliance_status;

    -- Generate compliance report
    PROCEDURE generate_compliance_report(
        p_drug_id           IN  NUMBER DEFAULT NULL,
        p_agency            IN  VARCHAR2 DEFAULT NULL,
        p_include_completed IN  CHAR DEFAULT 'N',
        p_cursor            OUT SYS_REFCURSOR
    ) IS
    BEGIN
        OPEN p_cursor FOR
            SELECT
                cr.id                       AS compliance_id,
                d.drug_name,
                d.ndc_code,
                cr.requirement_type,
                cr.requirement_desc,
                cr.due_date,
                cr.completion_date,
                cr.status,
                cr.responsible_party,
                cr.escalation_level,
                cr.risk_score,
                TRUNC(cr.due_date) - TRUNC(SYSDATE) AS days_until_due,
                CASE
                    WHEN cr.status = 'COMPLETED' THEN 'GREEN'
                    WHEN TRUNC(cr.due_date) - TRUNC(SYSDATE) < 0 THEN 'RED'
                    WHEN TRUNC(cr.due_date) - TRUNC(SYSDATE) <= 7 THEN 'AMBER'
                    WHEN TRUNC(cr.due_date) - TRUNC(SYSDATE) <= 30 THEN 'YELLOW'
                    ELSE 'GREEN'
                END AS rag_status,
                rs.agency,
                rs.tracking_number
            FROM COMPLIANCE_RECORD cr
            JOIN DRUG d ON d.drug_id = cr.drug_id
            LEFT JOIN REGULATORY_SUBMISSION rs ON rs.id = cr.linked_submission
            WHERE (p_drug_id IS NULL OR cr.drug_id = p_drug_id)
              AND (p_agency IS NULL OR rs.agency = p_agency)
              AND (p_include_completed = 'Y' OR cr.status NOT IN ('COMPLETED', 'WAIVED', 'CANCELLED'))
            ORDER BY
                CASE cr.status
                    WHEN 'ESCALATED' THEN 1
                    WHEN 'OVERDUE' THEN 2
                    WHEN 'IN_PROGRESS' THEN 3
                    WHEN 'PENDING' THEN 4
                    ELSE 5
                END,
                cr.due_date ASC;
    END generate_compliance_report;

    -- Escalate overdue compliance records
    PROCEDURE escalate_overdue(
        p_escalation_user   IN  VARCHAR2,
        p_escalated_count   OUT NUMBER
    ) IS
        CURSOR c_overdue IS
            SELECT id, status, escalation_level, due_date,
                   TRUNC(SYSDATE) - TRUNC(due_date) AS days_overdue
            FROM COMPLIANCE_RECORD
            WHERE status IN ('PENDING', 'IN_PROGRESS', 'OVERDUE')
              AND due_date < SYSDATE
              AND completion_date IS NULL
            FOR UPDATE;

        v_new_level     NUMBER;
        v_new_status    VARCHAR2(20);
        v_count         NUMBER := 0;
    BEGIN
        FOR rec IN c_overdue LOOP
            -- Determine new escalation level based on days overdue
            IF rec.days_overdue > 90 THEN
                v_new_level := 5;
                v_new_status := 'ESCALATED';
            ELSIF rec.days_overdue > 60 THEN
                v_new_level := 4;
                v_new_status := 'ESCALATED';
            ELSIF rec.days_overdue > 30 THEN
                v_new_level := 3;
                v_new_status := 'ESCALATED';
            ELSIF rec.days_overdue > 14 THEN
                v_new_level := 2;
                v_new_status := 'OVERDUE';
            ELSE
                v_new_level := 1;
                v_new_status := 'OVERDUE';
            END IF;

            -- Only escalate if level increases
            IF v_new_level > rec.escalation_level THEN
                UPDATE COMPLIANCE_RECORD
                SET status = v_new_status,
                    escalation_level = v_new_level,
                    risk_score = LEAST(100, 50 + (rec.days_overdue * 0.5)),
                    modified_by = p_escalation_user,
                    modified_date = SYSTIMESTAMP
                WHERE id = rec.id;

                v_count := v_count + 1;
            END IF;
        END LOOP;

        COMMIT;
        p_escalated_count := v_count;

        DBMS_OUTPUT.PUT_LINE('Escalation complete: ' || v_count || ' records escalated');

    EXCEPTION
        WHEN OTHERS THEN
            ROLLBACK;
            RAISE;
    END escalate_overdue;

    -- Calculate risk score
    FUNCTION calculate_risk_score(
        p_compliance_id     IN  NUMBER
    ) RETURN NUMBER IS
        v_days_until_due    NUMBER;
        v_escalation        NUMBER;
        v_req_type          VARCHAR2(50);
        v_base_score        NUMBER := 0;
        v_time_factor       NUMBER := 0;
        v_type_weight       NUMBER := 1;
    BEGIN
        SELECT
            TRUNC(due_date) - TRUNC(SYSDATE),
            NVL(escalation_level, 0),
            requirement_type
        INTO v_days_until_due, v_escalation, v_req_type
        FROM COMPLIANCE_RECORD
        WHERE id = p_compliance_id;

        -- Time-based factor
        IF v_days_until_due < 0 THEN
            v_time_factor := LEAST(50, ABS(v_days_until_due));
        ELSIF v_days_until_due <= 7 THEN
            v_time_factor := 30;
        ELSIF v_days_until_due <= 30 THEN
            v_time_factor := 15;
        ELSE
            v_time_factor := 5;
        END IF;

        -- Requirement type weight
        CASE v_req_type
            WHEN 'REMS'                THEN v_type_weight := 2.0;
            WHEN 'POST_MARKETING_STUDY' THEN v_type_weight := 1.8;
            WHEN 'PHARMACOVIGILANCE'   THEN v_type_weight := 1.7;
            WHEN 'PMR'                 THEN v_type_weight := 1.5;
            WHEN 'PMC'                 THEN v_type_weight := 1.3;
            WHEN 'LABELING_UPDATE'     THEN v_type_weight := 1.4;
            WHEN 'GMP_INSPECTION'      THEN v_type_weight := 1.6;
            ELSE v_type_weight := 1.0;
        END CASE;

        -- Base score from escalation
        v_base_score := v_escalation * 10;

        -- Final score (capped at 100)
        RETURN LEAST(100, ROUND((v_base_score + v_time_factor) * v_type_weight, 2));

    EXCEPTION
        WHEN NO_DATA_FOUND THEN
            RETURN -1;
    END calculate_risk_score;

    -- Get SLA status
    FUNCTION get_sla_status(
        p_compliance_id     IN  NUMBER
    ) RETURN VARCHAR2 IS
        v_due_date      DATE;
        v_comp_date     DATE;
        v_status        VARCHAR2(20);
        v_days          NUMBER;
    BEGIN
        SELECT due_date, completion_date, status
        INTO v_due_date, v_comp_date, v_status
        FROM COMPLIANCE_RECORD
        WHERE id = p_compliance_id;

        IF v_status IN ('COMPLETED', 'WAIVED') THEN
            IF v_comp_date IS NOT NULL AND v_comp_date <= v_due_date THEN
                RETURN 'MET';
            ELSIF v_comp_date IS NOT NULL THEN
                v_days := TRUNC(v_comp_date) - TRUNC(v_due_date);
                RETURN 'BREACHED_BY_' || v_days || '_DAYS';
            ELSE
                RETURN 'MET';
            END IF;
        END IF;

        v_days := TRUNC(v_due_date) - TRUNC(SYSDATE);

        IF v_days < 0 THEN
            RETURN 'BREACHED_BY_' || ABS(v_days) || '_DAYS';
        ELSIF v_days <= 7 THEN
            RETURN 'AT_RISK';
        ELSIF v_days <= 30 THEN
            RETURN 'ON_TRACK';
        ELSE
            RETURN 'ON_TRACK';
        END IF;

    EXCEPTION
        WHEN NO_DATA_FOUND THEN
            RETURN 'UNKNOWN';
    END get_sla_status;

END PKG_COMPLIANCE;
/

-- ============================================================================
-- PKG_AUDIT - Audit Trail Management Package
-- ============================================================================

CREATE OR REPLACE PACKAGE PKG_AUDIT AS

    TYPE t_audit_summary IS RECORD (
        table_name      VARCHAR2(50),
        insert_count    NUMBER,
        update_count    NUMBER,
        delete_count    NUMBER,
        unique_users    NUMBER,
        date_range_from TIMESTAMP,
        date_range_to   TIMESTAMP
    );

    TYPE t_audit_summary_tab IS TABLE OF t_audit_summary INDEX BY PLS_INTEGER;

    -- Query audit trail with filters
    PROCEDURE query_audit_trail(
        p_table_name    IN  VARCHAR2 DEFAULT NULL,
        p_record_id     IN  NUMBER DEFAULT NULL,
        p_changed_by    IN  VARCHAR2 DEFAULT NULL,
        p_date_from     IN  TIMESTAMP DEFAULT NULL,
        p_date_to       IN  TIMESTAMP DEFAULT NULL,
        p_action        IN  VARCHAR2 DEFAULT NULL,
        p_cursor        OUT SYS_REFCURSOR
    );

    -- Generate audit report summary
    PROCEDURE generate_audit_report(
        p_date_from     IN  TIMESTAMP,
        p_date_to       IN  TIMESTAMP,
        p_cursor        OUT SYS_REFCURSOR
    );

    -- Archive old audit records to a history table
    PROCEDURE archive_old_audit_records(
        p_archive_before    IN  TIMESTAMP,
        p_archived_count    OUT NUMBER,
        p_deleted_count     OUT NUMBER
    );

    -- Get record change count in period
    FUNCTION get_change_count(
        p_table_name    IN  VARCHAR2,
        p_record_id     IN  NUMBER,
        p_date_from     IN  TIMESTAMP DEFAULT NULL,
        p_date_to       IN  TIMESTAMP DEFAULT NULL
    ) RETURN NUMBER;

END PKG_AUDIT;
/

CREATE OR REPLACE PACKAGE BODY PKG_AUDIT AS

    -- Query audit trail with flexible filters
    PROCEDURE query_audit_trail(
        p_table_name    IN  VARCHAR2 DEFAULT NULL,
        p_record_id     IN  NUMBER DEFAULT NULL,
        p_changed_by    IN  VARCHAR2 DEFAULT NULL,
        p_date_from     IN  TIMESTAMP DEFAULT NULL,
        p_date_to       IN  TIMESTAMP DEFAULT NULL,
        p_action        IN  VARCHAR2 DEFAULT NULL,
        p_cursor        OUT SYS_REFCURSOR
    ) IS
    BEGIN
        OPEN p_cursor FOR
            SELECT
                id,
                table_name,
                record_id,
                action,
                old_values,
                new_values,
                changed_by,
                changed_at,
                ip_address,
                session_id,
                application_name,
                transaction_id
            FROM AUDIT_TRAIL
            WHERE (p_table_name IS NULL OR table_name = p_table_name)
              AND (p_record_id IS NULL OR record_id = p_record_id)
              AND (p_changed_by IS NULL OR changed_by = p_changed_by)
              AND (p_date_from IS NULL OR changed_at >= p_date_from)
              AND (p_date_to IS NULL OR changed_at <= p_date_to)
              AND (p_action IS NULL OR action = p_action)
            ORDER BY changed_at DESC;
    END query_audit_trail;

    -- Generate audit summary report
    PROCEDURE generate_audit_report(
        p_date_from     IN  TIMESTAMP,
        p_date_to       IN  TIMESTAMP,
        p_cursor        OUT SYS_REFCURSOR
    ) IS
    BEGIN
        OPEN p_cursor FOR
            SELECT
                table_name,
                SUM(CASE WHEN action = 'INSERT' THEN 1 ELSE 0 END) AS insert_count,
                SUM(CASE WHEN action = 'UPDATE' THEN 1 ELSE 0 END) AS update_count,
                SUM(CASE WHEN action = 'DELETE' THEN 1 ELSE 0 END) AS delete_count,
                COUNT(DISTINCT changed_by)                          AS unique_users,
                MIN(changed_at)                                     AS first_change,
                MAX(changed_at)                                     AS last_change,
                COUNT(*)                                            AS total_changes
            FROM AUDIT_TRAIL
            WHERE changed_at BETWEEN p_date_from AND p_date_to
            GROUP BY table_name
            ORDER BY total_changes DESC;
    END generate_audit_report;

    -- Archive old audit records
    PROCEDURE archive_old_audit_records(
        p_archive_before    IN  TIMESTAMP,
        p_archived_count    OUT NUMBER,
        p_deleted_count     OUT NUMBER
    ) IS
    BEGIN
        -- Create archive table if not exists (first run)
        BEGIN
            EXECUTE IMMEDIATE '
                CREATE TABLE AUDIT_TRAIL_ARCHIVE AS
                SELECT * FROM AUDIT_TRAIL WHERE 1=0';
        EXCEPTION
            WHEN OTHERS THEN
                IF SQLCODE != -955 THEN  -- ORA-00955: name already in use
                    RAISE;
                END IF;
        END;

        -- Copy old records to archive
        INSERT INTO AUDIT_TRAIL_ARCHIVE
        SELECT * FROM AUDIT_TRAIL
        WHERE changed_at < p_archive_before;

        p_archived_count := SQL%ROWCOUNT;

        -- Delete archived records from main table
        DELETE FROM AUDIT_TRAIL
        WHERE changed_at < p_archive_before;

        p_deleted_count := SQL%ROWCOUNT;

        COMMIT;

        DBMS_OUTPUT.PUT_LINE('Audit archive complete:');
        DBMS_OUTPUT.PUT_LINE('  Archived: ' || p_archived_count || ' records');
        DBMS_OUTPUT.PUT_LINE('  Deleted:  ' || p_deleted_count || ' records');

    EXCEPTION
        WHEN OTHERS THEN
            ROLLBACK;
            RAISE;
    END archive_old_audit_records;

    -- Get change count for a specific record
    FUNCTION get_change_count(
        p_table_name    IN  VARCHAR2,
        p_record_id     IN  NUMBER,
        p_date_from     IN  TIMESTAMP DEFAULT NULL,
        p_date_to       IN  TIMESTAMP DEFAULT NULL
    ) RETURN NUMBER IS
        v_count NUMBER;
    BEGIN
        SELECT COUNT(*)
        INTO v_count
        FROM AUDIT_TRAIL
        WHERE table_name = p_table_name
          AND record_id = p_record_id
          AND (p_date_from IS NULL OR changed_at >= p_date_from)
          AND (p_date_to IS NULL OR changed_at <= p_date_to);
        RETURN v_count;
    END get_change_count;

END PKG_AUDIT;
/
