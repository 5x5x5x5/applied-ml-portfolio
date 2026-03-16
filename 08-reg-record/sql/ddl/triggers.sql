--------------------------------------------------------------------------------
-- RegRecord - PL/SQL Triggers
--
-- 1. Audit trail triggers on all main tables
-- 2. Submission status validation trigger (state machine enforcement)
-- 3. Document version auto-increment trigger
-- 4. Compliance due date alert trigger
--------------------------------------------------------------------------------

-- ============================================================================
-- 1. AUDIT TRAIL TRIGGERS
-- Captures old and new values as JSON for complete change tracking
-- ============================================================================

-- Helper function to build JSON from column values (used by all audit triggers)
CREATE OR REPLACE FUNCTION fn_build_json_pair(
    p_key   IN VARCHAR2,
    p_value IN VARCHAR2
) RETURN VARCHAR2 IS
BEGIN
    IF p_value IS NULL THEN
        RETURN '"' || p_key || '": null';
    ELSE
        RETURN '"' || p_key || '": "' || REPLACE(REPLACE(p_value, '\', '\\'), '"', '\"') || '"';
    END IF;
END fn_build_json_pair;
/

-- ---------------------------------------------------------------------------
-- REGULATORY_SUBMISSION Audit Trigger
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TRIGGER trg_audit_submission
AFTER INSERT OR UPDATE OR DELETE ON REGULATORY_SUBMISSION
FOR EACH ROW
DECLARE
    v_action        VARCHAR2(10);
    v_old_values    CLOB;
    v_new_values    CLOB;
    v_record_id     NUMBER;
    v_changed_by    VARCHAR2(100);
BEGIN
    IF INSERTING THEN
        v_action := 'INSERT';
        v_record_id := :NEW.id;
        v_changed_by := :NEW.created_by;
        v_old_values := NULL;
        v_new_values := '{' ||
            fn_build_json_pair('drug_id', TO_CHAR(:NEW.drug_id)) || ', ' ||
            fn_build_json_pair('submission_type', :NEW.submission_type) || ', ' ||
            fn_build_json_pair('status', :NEW.status) || ', ' ||
            fn_build_json_pair('agency', :NEW.agency) || ', ' ||
            fn_build_json_pair('tracking_number', :NEW.tracking_number) || ', ' ||
            fn_build_json_pair('priority', :NEW.priority) || ', ' ||
            fn_build_json_pair('assigned_to', :NEW.assigned_to) ||
        '}';
    ELSIF UPDATING THEN
        v_action := 'UPDATE';
        v_record_id := :NEW.id;
        v_changed_by := NVL(:NEW.modified_by, :OLD.created_by);
        v_old_values := '{' ||
            fn_build_json_pair('status', :OLD.status) || ', ' ||
            fn_build_json_pair('priority', :OLD.priority) || ', ' ||
            fn_build_json_pair('assigned_to', :OLD.assigned_to) || ', ' ||
            fn_build_json_pair('tracking_number', :OLD.tracking_number) ||
        '}';
        v_new_values := '{' ||
            fn_build_json_pair('status', :NEW.status) || ', ' ||
            fn_build_json_pair('priority', :NEW.priority) || ', ' ||
            fn_build_json_pair('assigned_to', :NEW.assigned_to) || ', ' ||
            fn_build_json_pair('tracking_number', :NEW.tracking_number) ||
        '}';
    ELSIF DELETING THEN
        v_action := 'DELETE';
        v_record_id := :OLD.id;
        v_changed_by := :OLD.created_by;
        v_new_values := NULL;
        v_old_values := '{' ||
            fn_build_json_pair('drug_id', TO_CHAR(:OLD.drug_id)) || ', ' ||
            fn_build_json_pair('submission_type', :OLD.submission_type) || ', ' ||
            fn_build_json_pair('status', :OLD.status) || ', ' ||
            fn_build_json_pair('agency', :OLD.agency) || ', ' ||
            fn_build_json_pair('tracking_number', :OLD.tracking_number) ||
        '}';
    END IF;

    INSERT INTO AUDIT_TRAIL (
        table_name, record_id, action, old_values, new_values,
        changed_by, changed_at, ip_address, session_id, application_name
    ) VALUES (
        'REGULATORY_SUBMISSION', v_record_id, v_action, v_old_values, v_new_values,
        v_changed_by, SYSTIMESTAMP,
        SYS_CONTEXT('USERENV', 'IP_ADDRESS'),
        SYS_CONTEXT('USERENV', 'SESSIONID'),
        SYS_CONTEXT('USERENV', 'MODULE')
    );
END trg_audit_submission;
/

-- ---------------------------------------------------------------------------
-- SUBMISSION_DOCUMENT Audit Trigger
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TRIGGER trg_audit_document
AFTER INSERT OR UPDATE OR DELETE ON SUBMISSION_DOCUMENT
FOR EACH ROW
DECLARE
    v_action        VARCHAR2(10);
    v_old_values    CLOB;
    v_new_values    CLOB;
    v_record_id     NUMBER;
    v_changed_by    VARCHAR2(100);
BEGIN
    IF INSERTING THEN
        v_action := 'INSERT';
        v_record_id := :NEW.id;
        v_changed_by := :NEW.uploaded_by;
        v_old_values := NULL;
        v_new_values := '{' ||
            fn_build_json_pair('submission_id', TO_CHAR(:NEW.submission_id)) || ', ' ||
            fn_build_json_pair('doc_type', :NEW.doc_type) || ', ' ||
            fn_build_json_pair('version', TO_CHAR(:NEW.version)) || ', ' ||
            fn_build_json_pair('file_path', :NEW.file_path) || ', ' ||
            fn_build_json_pair('checksum', :NEW.checksum) ||
        '}';
    ELSIF UPDATING THEN
        v_action := 'UPDATE';
        v_record_id := :NEW.id;
        v_changed_by := :NEW.uploaded_by;
        v_old_values := '{' ||
            fn_build_json_pair('version', TO_CHAR(:OLD.version)) || ', ' ||
            fn_build_json_pair('status', :OLD.status) || ', ' ||
            fn_build_json_pair('file_path', :OLD.file_path) ||
        '}';
        v_new_values := '{' ||
            fn_build_json_pair('version', TO_CHAR(:NEW.version)) || ', ' ||
            fn_build_json_pair('status', :NEW.status) || ', ' ||
            fn_build_json_pair('file_path', :NEW.file_path) ||
        '}';
    ELSIF DELETING THEN
        v_action := 'DELETE';
        v_record_id := :OLD.id;
        v_changed_by := :OLD.uploaded_by;
        v_new_values := NULL;
        v_old_values := '{' ||
            fn_build_json_pair('submission_id', TO_CHAR(:OLD.submission_id)) || ', ' ||
            fn_build_json_pair('doc_type', :OLD.doc_type) || ', ' ||
            fn_build_json_pair('version', TO_CHAR(:OLD.version)) ||
        '}';
    END IF;

    INSERT INTO AUDIT_TRAIL (
        table_name, record_id, action, old_values, new_values,
        changed_by, changed_at, ip_address, session_id, application_name
    ) VALUES (
        'SUBMISSION_DOCUMENT', v_record_id, v_action, v_old_values, v_new_values,
        v_changed_by, SYSTIMESTAMP,
        SYS_CONTEXT('USERENV', 'IP_ADDRESS'),
        SYS_CONTEXT('USERENV', 'SESSIONID'),
        SYS_CONTEXT('USERENV', 'MODULE')
    );
END trg_audit_document;
/

-- ---------------------------------------------------------------------------
-- APPROVAL_RECORD Audit Trigger
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TRIGGER trg_audit_approval
AFTER INSERT OR UPDATE OR DELETE ON APPROVAL_RECORD
FOR EACH ROW
DECLARE
    v_action        VARCHAR2(10);
    v_old_values    CLOB;
    v_new_values    CLOB;
    v_record_id     NUMBER;
    v_changed_by    VARCHAR2(100);
BEGIN
    IF INSERTING THEN
        v_action := 'INSERT';
        v_record_id := :NEW.id;
        v_changed_by := :NEW.created_by;
        v_old_values := NULL;
        v_new_values := '{' ||
            fn_build_json_pair('submission_id', TO_CHAR(:NEW.submission_id)) || ', ' ||
            fn_build_json_pair('approval_date', TO_CHAR(:NEW.approval_date, 'YYYY-MM-DD')) || ', ' ||
            fn_build_json_pair('approval_type', :NEW.approval_type) || ', ' ||
            fn_build_json_pair('approval_number', :NEW.approval_number) ||
        '}';
    ELSIF UPDATING THEN
        v_action := 'UPDATE';
        v_record_id := :NEW.id;
        v_changed_by := NVL(:NEW.created_by, :OLD.created_by);
        v_old_values := '{' ||
            fn_build_json_pair('approval_type', :OLD.approval_type) || ', ' ||
            fn_build_json_pair('conditions', DBMS_LOB.SUBSTR(:OLD.conditions, 200, 1)) || ', ' ||
            fn_build_json_pair('expiry_date', TO_CHAR(:OLD.expiry_date, 'YYYY-MM-DD')) ||
        '}';
        v_new_values := '{' ||
            fn_build_json_pair('approval_type', :NEW.approval_type) || ', ' ||
            fn_build_json_pair('conditions', DBMS_LOB.SUBSTR(:NEW.conditions, 200, 1)) || ', ' ||
            fn_build_json_pair('expiry_date', TO_CHAR(:NEW.expiry_date, 'YYYY-MM-DD')) ||
        '}';
    ELSIF DELETING THEN
        v_action := 'DELETE';
        v_record_id := :OLD.id;
        v_changed_by := :OLD.created_by;
        v_new_values := NULL;
        v_old_values := '{' ||
            fn_build_json_pair('submission_id', TO_CHAR(:OLD.submission_id)) || ', ' ||
            fn_build_json_pair('approval_type', :OLD.approval_type) ||
        '}';
    END IF;

    INSERT INTO AUDIT_TRAIL (
        table_name, record_id, action, old_values, new_values,
        changed_by, changed_at, ip_address, session_id, application_name
    ) VALUES (
        'APPROVAL_RECORD', v_record_id, v_action, v_old_values, v_new_values,
        v_changed_by, SYSTIMESTAMP,
        SYS_CONTEXT('USERENV', 'IP_ADDRESS'),
        SYS_CONTEXT('USERENV', 'SESSIONID'),
        SYS_CONTEXT('USERENV', 'MODULE')
    );
END trg_audit_approval;
/

-- ---------------------------------------------------------------------------
-- LABELING_CHANGE Audit Trigger
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TRIGGER trg_audit_labeling
AFTER INSERT OR UPDATE OR DELETE ON LABELING_CHANGE
FOR EACH ROW
DECLARE
    v_action        VARCHAR2(10);
    v_old_values    CLOB;
    v_new_values    CLOB;
    v_record_id     NUMBER;
    v_changed_by    VARCHAR2(100);
BEGIN
    IF INSERTING THEN
        v_action := 'INSERT';
        v_record_id := :NEW.id;
        v_changed_by := :NEW.created_by;
        v_old_values := NULL;
        v_new_values := '{' ||
            fn_build_json_pair('drug_id', TO_CHAR(:NEW.drug_id)) || ', ' ||
            fn_build_json_pair('change_type', :NEW.change_type) || ', ' ||
            fn_build_json_pair('effective_date', TO_CHAR(:NEW.effective_date, 'YYYY-MM-DD')) || ', ' ||
            fn_build_json_pair('status', :NEW.status) || ', ' ||
            fn_build_json_pair('approved_by', :NEW.approved_by) ||
        '}';
    ELSIF UPDATING THEN
        v_action := 'UPDATE';
        v_record_id := :NEW.id;
        v_changed_by := NVL(:NEW.approved_by, :OLD.created_by);
        v_old_values := '{' ||
            fn_build_json_pair('status', :OLD.status) || ', ' ||
            fn_build_json_pair('change_type', :OLD.change_type) || ', ' ||
            fn_build_json_pair('approved_by', :OLD.approved_by) ||
        '}';
        v_new_values := '{' ||
            fn_build_json_pair('status', :NEW.status) || ', ' ||
            fn_build_json_pair('change_type', :NEW.change_type) || ', ' ||
            fn_build_json_pair('approved_by', :NEW.approved_by) ||
        '}';
    ELSIF DELETING THEN
        v_action := 'DELETE';
        v_record_id := :OLD.id;
        v_changed_by := :OLD.created_by;
        v_new_values := NULL;
        v_old_values := '{' ||
            fn_build_json_pair('drug_id', TO_CHAR(:OLD.drug_id)) || ', ' ||
            fn_build_json_pair('change_type', :OLD.change_type) || ', ' ||
            fn_build_json_pair('status', :OLD.status) ||
        '}';
    END IF;

    INSERT INTO AUDIT_TRAIL (
        table_name, record_id, action, old_values, new_values,
        changed_by, changed_at, ip_address, session_id, application_name
    ) VALUES (
        'LABELING_CHANGE', v_record_id, v_action, v_old_values, v_new_values,
        v_changed_by, SYSTIMESTAMP,
        SYS_CONTEXT('USERENV', 'IP_ADDRESS'),
        SYS_CONTEXT('USERENV', 'SESSIONID'),
        SYS_CONTEXT('USERENV', 'MODULE')
    );
END trg_audit_labeling;
/

-- ---------------------------------------------------------------------------
-- COMPLIANCE_RECORD Audit Trigger
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TRIGGER trg_audit_compliance
AFTER INSERT OR UPDATE OR DELETE ON COMPLIANCE_RECORD
FOR EACH ROW
DECLARE
    v_action        VARCHAR2(10);
    v_old_values    CLOB;
    v_new_values    CLOB;
    v_record_id     NUMBER;
    v_changed_by    VARCHAR2(100);
BEGIN
    IF INSERTING THEN
        v_action := 'INSERT';
        v_record_id := :NEW.id;
        v_changed_by := :NEW.created_by;
        v_old_values := NULL;
        v_new_values := '{' ||
            fn_build_json_pair('drug_id', TO_CHAR(:NEW.drug_id)) || ', ' ||
            fn_build_json_pair('requirement_type', :NEW.requirement_type) || ', ' ||
            fn_build_json_pair('due_date', TO_CHAR(:NEW.due_date, 'YYYY-MM-DD')) || ', ' ||
            fn_build_json_pair('status', :NEW.status) || ', ' ||
            fn_build_json_pair('responsible_party', :NEW.responsible_party) ||
        '}';
    ELSIF UPDATING THEN
        v_action := 'UPDATE';
        v_record_id := :NEW.id;
        v_changed_by := NVL(:NEW.modified_by, :OLD.created_by);
        v_old_values := '{' ||
            fn_build_json_pair('status', :OLD.status) || ', ' ||
            fn_build_json_pair('escalation_level', TO_CHAR(:OLD.escalation_level)) || ', ' ||
            fn_build_json_pair('risk_score', TO_CHAR(:OLD.risk_score)) || ', ' ||
            fn_build_json_pair('responsible_party', :OLD.responsible_party) ||
        '}';
        v_new_values := '{' ||
            fn_build_json_pair('status', :NEW.status) || ', ' ||
            fn_build_json_pair('escalation_level', TO_CHAR(:NEW.escalation_level)) || ', ' ||
            fn_build_json_pair('risk_score', TO_CHAR(:NEW.risk_score)) || ', ' ||
            fn_build_json_pair('responsible_party', :NEW.responsible_party) ||
        '}';
    ELSIF DELETING THEN
        v_action := 'DELETE';
        v_record_id := :OLD.id;
        v_changed_by := :OLD.created_by;
        v_new_values := NULL;
        v_old_values := '{' ||
            fn_build_json_pair('drug_id', TO_CHAR(:OLD.drug_id)) || ', ' ||
            fn_build_json_pair('requirement_type', :OLD.requirement_type) || ', ' ||
            fn_build_json_pair('status', :OLD.status) ||
        '}';
    END IF;

    INSERT INTO AUDIT_TRAIL (
        table_name, record_id, action, old_values, new_values,
        changed_by, changed_at, ip_address, session_id, application_name
    ) VALUES (
        'COMPLIANCE_RECORD', v_record_id, v_action, v_old_values, v_new_values,
        v_changed_by, SYSTIMESTAMP,
        SYS_CONTEXT('USERENV', 'IP_ADDRESS'),
        SYS_CONTEXT('USERENV', 'SESSIONID'),
        SYS_CONTEXT('USERENV', 'MODULE')
    );
END trg_audit_compliance;
/

-- ============================================================================
-- 2. SUBMISSION STATUS VALIDATION TRIGGER
-- Enforces the state machine: DRAFT -> SUBMITTED -> UNDER_REVIEW ->
--   APPROVED / REJECTED / COMPLETE_RESPONSE
-- Also allows: any -> WITHDRAWN, UNDER_REVIEW -> ON_HOLD -> UNDER_REVIEW
-- ============================================================================

CREATE OR REPLACE TRIGGER trg_submission_status_validate
BEFORE UPDATE OF status ON REGULATORY_SUBMISSION
FOR EACH ROW
DECLARE
    v_valid_transition  BOOLEAN := FALSE;

    -- Define valid state transitions as a nested procedure
    PROCEDURE check_transition(
        p_from IN VARCHAR2,
        p_to   IN VARCHAR2
    ) IS
    BEGIN
        -- Any state can transition to WITHDRAWN
        IF p_to = 'WITHDRAWN' THEN
            v_valid_transition := TRUE;
            RETURN;
        END IF;

        CASE p_from
            WHEN 'DRAFT' THEN
                v_valid_transition := p_to IN ('SUBMITTED');
            WHEN 'SUBMITTED' THEN
                v_valid_transition := p_to IN ('UNDER_REVIEW');
            WHEN 'UNDER_REVIEW' THEN
                v_valid_transition := p_to IN ('APPROVED', 'REJECTED', 'ON_HOLD', 'COMPLETE_RESPONSE');
            WHEN 'ON_HOLD' THEN
                v_valid_transition := p_to IN ('UNDER_REVIEW');
            WHEN 'COMPLETE_RESPONSE' THEN
                v_valid_transition := p_to IN ('SUBMITTED');
            WHEN 'REJECTED' THEN
                v_valid_transition := p_to IN ('DRAFT');  -- Allow resubmission from rejected
            WHEN 'APPROVED' THEN
                v_valid_transition := FALSE;  -- Approved is a terminal state
            ELSE
                v_valid_transition := FALSE;
        END CASE;
    END check_transition;

BEGIN
    -- Skip validation if status hasn't actually changed
    IF :OLD.status = :NEW.status THEN
        RETURN;
    END IF;

    check_transition(:OLD.status, :NEW.status);

    IF NOT v_valid_transition THEN
        RAISE_APPLICATION_ERROR(
            -20001,
            'Invalid status transition from ' || :OLD.status || ' to ' || :NEW.status ||
            ' for submission ID ' || :OLD.id ||
            '. Valid transitions: DRAFT->SUBMITTED->UNDER_REVIEW->APPROVED/REJECTED/ON_HOLD/COMPLETE_RESPONSE'
        );
    END IF;

    -- Auto-set submitted_date when moving to SUBMITTED
    IF :NEW.status = 'SUBMITTED' AND :OLD.status = 'DRAFT' THEN
        :NEW.submitted_date := SYSTIMESTAMP;
    END IF;

    -- Auto-set modified_date
    :NEW.modified_date := SYSTIMESTAMP;
END trg_submission_status_validate;
/

-- ============================================================================
-- 3. DOCUMENT VERSION AUTO-INCREMENT TRIGGER
-- When a new document is inserted for the same submission_id and doc_type,
-- automatically set version to max(version) + 1 and mark previous as SUPERSEDED
-- ============================================================================

CREATE OR REPLACE TRIGGER trg_doc_version_auto
BEFORE INSERT ON SUBMISSION_DOCUMENT
FOR EACH ROW
DECLARE
    v_max_version   NUMBER;
    v_prev_doc_id   NUMBER;
BEGIN
    -- Find the current maximum version for this submission + doc_type
    SELECT NVL(MAX(version), 0), MAX(id) KEEP (DENSE_RANK LAST ORDER BY version)
    INTO v_max_version, v_prev_doc_id
    FROM SUBMISSION_DOCUMENT
    WHERE submission_id = :NEW.submission_id
      AND doc_type = :NEW.doc_type
      AND status = 'ACTIVE';

    -- If there is an existing version and the new version is not explicitly set higher
    IF v_max_version > 0 AND :NEW.version <= v_max_version THEN
        :NEW.version := v_max_version + 1;
    END IF;

    -- Mark previous active version as SUPERSEDED
    IF v_prev_doc_id IS NOT NULL THEN
        UPDATE SUBMISSION_DOCUMENT
        SET status = 'SUPERSEDED',
            modified_date = SYSTIMESTAMP
        WHERE id = v_prev_doc_id
          AND status = 'ACTIVE';
    END IF;
END trg_doc_version_auto;
/

-- ============================================================================
-- 4. COMPLIANCE DUE DATE ALERT TRIGGER
-- When a compliance record approaches its due date or becomes overdue,
-- automatically update status and escalation level
-- ============================================================================

CREATE OR REPLACE TRIGGER trg_compliance_due_date_alert
BEFORE INSERT OR UPDATE ON COMPLIANCE_RECORD
FOR EACH ROW
DECLARE
    v_days_until_due    NUMBER;
    v_new_status        VARCHAR2(20);
    v_new_escalation    NUMBER(1);
BEGIN
    -- Only process records that are PENDING or IN_PROGRESS
    IF :NEW.status NOT IN ('PENDING', 'IN_PROGRESS', 'OVERDUE', 'ESCALATED') THEN
        RETURN;
    END IF;

    -- Skip if already completed, waived, or cancelled
    IF :NEW.completion_date IS NOT NULL THEN
        RETURN;
    END IF;

    v_days_until_due := TRUNC(:NEW.due_date) - TRUNC(SYSDATE);
    v_new_status := :NEW.status;
    v_new_escalation := NVL(:NEW.escalation_level, 0);

    -- Determine status and escalation based on days until due
    IF v_days_until_due < -30 THEN
        -- More than 30 days overdue: maximum escalation
        v_new_status := 'ESCALATED';
        v_new_escalation := LEAST(v_new_escalation + 1, 5);
    ELSIF v_days_until_due < 0 THEN
        -- Overdue
        v_new_status := 'OVERDUE';
        v_new_escalation := LEAST(v_new_escalation + 1, 3);
    ELSIF v_days_until_due <= 7 THEN
        -- Due within 7 days: keep current status but increase risk
        IF :NEW.risk_score IS NULL OR :NEW.risk_score < 75 THEN
            :NEW.risk_score := 75;
        END IF;
    ELSIF v_days_until_due <= 30 THEN
        -- Due within 30 days: moderate risk
        IF :NEW.risk_score IS NULL OR :NEW.risk_score < 50 THEN
            :NEW.risk_score := 50;
        END IF;
    END IF;

    :NEW.status := v_new_status;
    :NEW.escalation_level := v_new_escalation;
    :NEW.modified_date := SYSTIMESTAMP;
END trg_compliance_due_date_alert;
/

-- ============================================================================
-- 5. PSEUDO RECORD EXPIRY CHECK TRIGGER
-- Validates pseudo record on access and marks expired records
-- ============================================================================

CREATE OR REPLACE TRIGGER trg_pseudo_access_check
BEFORE UPDATE ON PSEUDO_RECORD
FOR EACH ROW
BEGIN
    -- Check if the pseudo record has expired
    IF :NEW.expiry_date IS NOT NULL AND :NEW.expiry_date < SYSTIMESTAMP THEN
        :NEW.is_active := 'N';
    END IF;

    -- Update last accessed timestamp and increment counter
    :NEW.last_accessed := SYSTIMESTAMP;
    :NEW.access_count := NVL(:OLD.access_count, 0) + 1;
END trg_pseudo_access_check;
/
