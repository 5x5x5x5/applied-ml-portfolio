--------------------------------------------------------------------------------
-- RegRecord - Regulatory Record Keeping System
-- Database Schema (Oracle PL/SQL Compatible)
--
-- Tables:
--   REGULATORY_SUBMISSION, SUBMISSION_DOCUMENT, APPROVAL_RECORD,
--   LABELING_CHANGE, COMPLIANCE_RECORD, AUDIT_TRAIL, PSEUDO_RECORD
--
-- Includes: sequences, constraints, indexes, partitioning, comments
--------------------------------------------------------------------------------

-- ============================================================================
-- SEQUENCES
-- ============================================================================

CREATE SEQUENCE seq_submission_id START WITH 1000 INCREMENT BY 1 NOCACHE;
CREATE SEQUENCE seq_document_id START WITH 5000 INCREMENT BY 1 NOCACHE;
CREATE SEQUENCE seq_approval_id START WITH 2000 INCREMENT BY 1 NOCACHE;
CREATE SEQUENCE seq_labeling_id START WITH 3000 INCREMENT BY 1 NOCACHE;
CREATE SEQUENCE seq_compliance_id START WITH 4000 INCREMENT BY 1 NOCACHE;
CREATE SEQUENCE seq_audit_id START WITH 1 INCREMENT BY 1 NOCACHE;
CREATE SEQUENCE seq_pseudo_id START WITH 10000 INCREMENT BY 1 NOCACHE;

-- ============================================================================
-- LOOKUP / REFERENCE TABLES
-- ============================================================================

CREATE TABLE DRUG (
    drug_id             NUMBER          GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    drug_name           VARCHAR2(200)   NOT NULL,
    generic_name        VARCHAR2(200),
    ndc_code            VARCHAR2(50)    UNIQUE,
    therapeutic_area    VARCHAR2(100),
    manufacturer        VARCHAR2(200)   NOT NULL,
    active_flag         CHAR(1)         DEFAULT 'Y' CHECK (active_flag IN ('Y', 'N')),
    created_date        TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL,
    modified_date       TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL
);

CREATE TABLE AGENCY (
    agency_code         VARCHAR2(20)    PRIMARY KEY,
    agency_name         VARCHAR2(200)   NOT NULL,
    country             VARCHAR2(100)   NOT NULL,
    region              VARCHAR2(50),
    active_flag         CHAR(1)         DEFAULT 'Y' CHECK (active_flag IN ('Y', 'N'))
);

COMMENT ON TABLE AGENCY IS 'Regulatory agencies worldwide (FDA, EMA, PMDA, etc.)';

-- Seed reference data for agencies
INSERT INTO AGENCY (agency_code, agency_name, country, region) VALUES ('FDA', 'Food and Drug Administration', 'United States', 'North America');
INSERT INTO AGENCY (agency_code, agency_name, country, region) VALUES ('EMA', 'European Medicines Agency', 'European Union', 'Europe');
INSERT INTO AGENCY (agency_code, agency_name, country, region) VALUES ('PMDA', 'Pharmaceuticals and Medical Devices Agency', 'Japan', 'Asia-Pacific');
INSERT INTO AGENCY (agency_code, agency_name, country, region) VALUES ('NMPA', 'National Medical Products Administration', 'China', 'Asia-Pacific');
INSERT INTO AGENCY (agency_code, agency_name, country, region) VALUES ('MHRA', 'Medicines and Healthcare products Regulatory Agency', 'United Kingdom', 'Europe');
INSERT INTO AGENCY (agency_code, agency_name, country, region) VALUES ('TGA', 'Therapeutic Goods Administration', 'Australia', 'Asia-Pacific');
INSERT INTO AGENCY (agency_code, agency_name, country, region) VALUES ('HC', 'Health Canada', 'Canada', 'North America');

COMMIT;

-- ============================================================================
-- REGULATORY_SUBMISSION
-- ============================================================================

CREATE TABLE REGULATORY_SUBMISSION (
    id                  NUMBER          DEFAULT seq_submission_id.NEXTVAL PRIMARY KEY,
    drug_id             NUMBER          NOT NULL,
    submission_type     VARCHAR2(50)    NOT NULL,
    status              VARCHAR2(30)    DEFAULT 'DRAFT' NOT NULL,
    submitted_date      TIMESTAMP,
    target_date         TIMESTAMP,
    agency              VARCHAR2(20)    NOT NULL,
    tracking_number     VARCHAR2(100)   UNIQUE,
    priority            VARCHAR2(20)    DEFAULT 'STANDARD',
    assigned_to         VARCHAR2(100),
    description         CLOB,
    created_by          VARCHAR2(100)   NOT NULL,
    created_date        TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL,
    modified_by         VARCHAR2(100),
    modified_date       TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL,
    CONSTRAINT fk_sub_drug FOREIGN KEY (drug_id) REFERENCES DRUG(drug_id),
    CONSTRAINT fk_sub_agency FOREIGN KEY (agency) REFERENCES AGENCY(agency_code),
    CONSTRAINT chk_sub_type CHECK (
        submission_type IN (
            'NDA', 'ANDA', 'BLA', 'IND', 'sNDA', 'sBLA',
            'MAA', 'TYPE_II_VARIATION', 'TYPE_IB_VARIATION',
            'RENEWAL', 'ANNUAL_REPORT', 'PSUR', 'DSUR'
        )
    ),
    CONSTRAINT chk_sub_status CHECK (
        status IN ('DRAFT', 'SUBMITTED', 'UNDER_REVIEW', 'APPROVED',
                   'REJECTED', 'WITHDRAWN', 'ON_HOLD', 'COMPLETE_RESPONSE')
    ),
    CONSTRAINT chk_sub_priority CHECK (
        priority IN ('STANDARD', 'PRIORITY', 'ACCELERATED', 'BREAKTHROUGH', 'FAST_TRACK')
    )
);

COMMENT ON TABLE REGULATORY_SUBMISSION IS 'Tracks all regulatory submissions to agencies worldwide';
COMMENT ON COLUMN REGULATORY_SUBMISSION.submission_type IS 'Type: NDA, ANDA, BLA, IND, sNDA, MAA, etc.';
COMMENT ON COLUMN REGULATORY_SUBMISSION.status IS 'Workflow state: DRAFT->SUBMITTED->UNDER_REVIEW->APPROVED/REJECTED';

CREATE INDEX idx_sub_drug_id ON REGULATORY_SUBMISSION(drug_id);
CREATE INDEX idx_sub_status ON REGULATORY_SUBMISSION(status);
CREATE INDEX idx_sub_agency ON REGULATORY_SUBMISSION(agency);
CREATE INDEX idx_sub_submitted_date ON REGULATORY_SUBMISSION(submitted_date);
CREATE INDEX idx_sub_tracking ON REGULATORY_SUBMISSION(tracking_number);
CREATE INDEX idx_sub_type_status ON REGULATORY_SUBMISSION(submission_type, status);

-- ============================================================================
-- SUBMISSION_DOCUMENT
-- ============================================================================

CREATE TABLE SUBMISSION_DOCUMENT (
    id                  NUMBER          DEFAULT seq_document_id.NEXTVAL PRIMARY KEY,
    submission_id       NUMBER          NOT NULL,
    doc_type            VARCHAR2(50)    NOT NULL,
    doc_title           VARCHAR2(500)   NOT NULL,
    version             NUMBER(5,1)     DEFAULT 1.0 NOT NULL,
    file_path           VARCHAR2(1000)  NOT NULL,
    file_size_bytes     NUMBER,
    checksum            VARCHAR2(128)   NOT NULL,
    checksum_algorithm  VARCHAR2(20)    DEFAULT 'SHA-256' NOT NULL,
    uploaded_by         VARCHAR2(100)   NOT NULL,
    upload_date         TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL,
    effective_date      DATE,
    expiry_date         DATE,
    status              VARCHAR2(20)    DEFAULT 'ACTIVE',
    confidentiality     VARCHAR2(20)    DEFAULT 'CONFIDENTIAL',
    created_date        TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL,
    modified_date       TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL,
    CONSTRAINT fk_doc_submission FOREIGN KEY (submission_id)
        REFERENCES REGULATORY_SUBMISSION(id) ON DELETE CASCADE,
    CONSTRAINT chk_doc_type CHECK (
        doc_type IN (
            'COVER_LETTER', 'MODULE_1', 'MODULE_2', 'MODULE_3',
            'MODULE_4', 'MODULE_5', 'LABEL', 'PI', 'PPI',
            'CLINICAL_STUDY_REPORT', 'SAFETY_UPDATE', 'CMC_DATA',
            'CORRESPONDENCE', 'MEETING_MINUTES', 'OTHER'
        )
    ),
    CONSTRAINT chk_doc_status CHECK (status IN ('ACTIVE', 'SUPERSEDED', 'ARCHIVED', 'DELETED')),
    CONSTRAINT chk_doc_conf CHECK (
        confidentiality IN ('PUBLIC', 'INTERNAL', 'CONFIDENTIAL', 'HIGHLY_CONFIDENTIAL')
    ),
    CONSTRAINT uk_doc_version UNIQUE (submission_id, doc_type, version)
);

COMMENT ON TABLE SUBMISSION_DOCUMENT IS 'Documents attached to regulatory submissions (eCTD modules, labels, etc.)';

CREATE INDEX idx_doc_submission ON SUBMISSION_DOCUMENT(submission_id);
CREATE INDEX idx_doc_type ON SUBMISSION_DOCUMENT(doc_type);
CREATE INDEX idx_doc_uploaded_by ON SUBMISSION_DOCUMENT(uploaded_by);
CREATE INDEX idx_doc_status ON SUBMISSION_DOCUMENT(status);

-- ============================================================================
-- APPROVAL_RECORD
-- ============================================================================

CREATE TABLE APPROVAL_RECORD (
    id                  NUMBER          DEFAULT seq_approval_id.NEXTVAL PRIMARY KEY,
    submission_id       NUMBER          NOT NULL,
    approval_date       DATE            NOT NULL,
    approval_type       VARCHAR2(50)    NOT NULL,
    conditions          CLOB,
    expiry_date         DATE,
    approval_number     VARCHAR2(100),
    approved_indication CLOB,
    market_exclusivity  VARCHAR2(50),
    exclusivity_expiry  DATE,
    pediatric_exclusivity CHAR(1)       DEFAULT 'N' CHECK (pediatric_exclusivity IN ('Y', 'N')),
    created_by          VARCHAR2(100)   NOT NULL,
    created_date        TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL,
    modified_date       TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL,
    CONSTRAINT fk_appr_submission FOREIGN KEY (submission_id)
        REFERENCES REGULATORY_SUBMISSION(id),
    CONSTRAINT chk_appr_type CHECK (
        approval_type IN (
            'FULL_APPROVAL', 'ACCELERATED_APPROVAL', 'CONDITIONAL_APPROVAL',
            'TENTATIVE_APPROVAL', 'EMERGENCY_USE', 'ORPHAN_DESIGNATION',
            'FAST_TRACK', 'BREAKTHROUGH_THERAPY', 'PRIORITY_REVIEW'
        )
    ),
    CONSTRAINT chk_appr_excl CHECK (
        market_exclusivity IN ('NCE', 'ORPHAN', 'PEDIATRIC', 'NEW_CLINICAL', 'NONE') OR market_exclusivity IS NULL
    )
);

COMMENT ON TABLE APPROVAL_RECORD IS 'Records of regulatory approvals, conditions, and exclusivity periods';

CREATE INDEX idx_appr_submission ON APPROVAL_RECORD(submission_id);
CREATE INDEX idx_appr_date ON APPROVAL_RECORD(approval_date);
CREATE INDEX idx_appr_type ON APPROVAL_RECORD(approval_type);
CREATE INDEX idx_appr_expiry ON APPROVAL_RECORD(expiry_date);

-- ============================================================================
-- LABELING_CHANGE
-- ============================================================================

CREATE TABLE LABELING_CHANGE (
    id                  NUMBER          DEFAULT seq_labeling_id.NEXTVAL PRIMARY KEY,
    drug_id             NUMBER          NOT NULL,
    change_type         VARCHAR2(50)    NOT NULL,
    change_category     VARCHAR2(50)    DEFAULT 'MINOR',
    effective_date      DATE            NOT NULL,
    description         CLOB            NOT NULL,
    rationale           CLOB,
    affected_sections   VARCHAR2(500),
    approved_by         VARCHAR2(100)   NOT NULL,
    reviewed_by         VARCHAR2(100),
    review_date         DATE,
    status              VARCHAR2(20)    DEFAULT 'PENDING',
    previous_label_ref  NUMBER,
    created_by          VARCHAR2(100)   NOT NULL,
    created_date        TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL,
    modified_date       TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL,
    CONSTRAINT fk_lbl_drug FOREIGN KEY (drug_id) REFERENCES DRUG(drug_id),
    CONSTRAINT fk_lbl_prev FOREIGN KEY (previous_label_ref) REFERENCES LABELING_CHANGE(id),
    CONSTRAINT chk_lbl_type CHECK (
        change_type IN (
            'SAFETY_UPDATE', 'EFFICACY_UPDATE', 'INDICATION_ADDITION',
            'INDICATION_REMOVAL', 'DOSAGE_CHANGE', 'WARNING_UPDATE',
            'CONTRAINDICATION', 'BOXED_WARNING', 'REMS_UPDATE',
            'PACKAGING_CHANGE', 'FORMULATION_CHANGE', 'GENERIC_LABEL'
        )
    ),
    CONSTRAINT chk_lbl_category CHECK (change_category IN ('MINOR', 'MODERATE', 'MAJOR', 'CBE', 'CBE_30', 'PAS')),
    CONSTRAINT chk_lbl_status CHECK (status IN ('PENDING', 'APPROVED', 'REJECTED', 'IMPLEMENTED', 'SUPERSEDED'))
);

COMMENT ON TABLE LABELING_CHANGE IS 'Tracks all labeling changes including safety updates, indication changes, etc.';

CREATE INDEX idx_lbl_drug ON LABELING_CHANGE(drug_id);
CREATE INDEX idx_lbl_type ON LABELING_CHANGE(change_type);
CREATE INDEX idx_lbl_effective ON LABELING_CHANGE(effective_date);
CREATE INDEX idx_lbl_status ON LABELING_CHANGE(status);

-- ============================================================================
-- COMPLIANCE_RECORD
-- ============================================================================

CREATE TABLE COMPLIANCE_RECORD (
    id                  NUMBER          DEFAULT seq_compliance_id.NEXTVAL PRIMARY KEY,
    drug_id             NUMBER          NOT NULL,
    requirement_type    VARCHAR2(50)    NOT NULL,
    requirement_desc    CLOB            NOT NULL,
    due_date            DATE            NOT NULL,
    completion_date     DATE,
    status              VARCHAR2(20)    DEFAULT 'PENDING' NOT NULL,
    responsible_party   VARCHAR2(100)   NOT NULL,
    escalation_level    NUMBER(1)       DEFAULT 0,
    risk_score          NUMBER(5,2),
    linked_submission   NUMBER,
    regulatory_reference VARCHAR2(200),
    notes               CLOB,
    created_by          VARCHAR2(100)   NOT NULL,
    created_date        TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL,
    modified_by         VARCHAR2(100),
    modified_date       TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL,
    CONSTRAINT fk_comp_drug FOREIGN KEY (drug_id) REFERENCES DRUG(drug_id),
    CONSTRAINT fk_comp_submission FOREIGN KEY (linked_submission)
        REFERENCES REGULATORY_SUBMISSION(id),
    CONSTRAINT chk_comp_type CHECK (
        requirement_type IN (
            'POST_MARKETING_STUDY', 'REMS', 'PSUR', 'DSUR',
            'ANNUAL_REPORT', 'PEDIATRIC_STUDY', 'PMC', 'PMR',
            'GMP_INSPECTION', 'LABELING_UPDATE', 'PHARMACOVIGILANCE',
            'RISK_EVALUATION', 'PERIODIC_REPORT', 'AD_HOC'
        )
    ),
    CONSTRAINT chk_comp_status CHECK (
        status IN ('PENDING', 'IN_PROGRESS', 'COMPLETED', 'OVERDUE',
                   'ESCALATED', 'WAIVED', 'CANCELLED')
    ),
    CONSTRAINT chk_comp_escl CHECK (escalation_level BETWEEN 0 AND 5)
);

COMMENT ON TABLE COMPLIANCE_RECORD IS 'Tracks regulatory compliance requirements, deadlines, and status';

CREATE INDEX idx_comp_drug ON COMPLIANCE_RECORD(drug_id);
CREATE INDEX idx_comp_due ON COMPLIANCE_RECORD(due_date);
CREATE INDEX idx_comp_status ON COMPLIANCE_RECORD(status);
CREATE INDEX idx_comp_responsible ON COMPLIANCE_RECORD(responsible_party);
CREATE INDEX idx_comp_type_status ON COMPLIANCE_RECORD(requirement_type, status);

-- ============================================================================
-- AUDIT_TRAIL (Partitioned by changed_at month for performance)
-- ============================================================================

CREATE TABLE AUDIT_TRAIL (
    id                  NUMBER          DEFAULT seq_audit_id.NEXTVAL,
    table_name          VARCHAR2(50)    NOT NULL,
    record_id           NUMBER          NOT NULL,
    action              VARCHAR2(10)    NOT NULL,
    old_values          CLOB,
    new_values          CLOB,
    changed_by          VARCHAR2(100)   NOT NULL,
    changed_at          TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL,
    ip_address          VARCHAR2(45),
    session_id          VARCHAR2(100),
    application_name    VARCHAR2(100),
    transaction_id      VARCHAR2(100),
    CONSTRAINT pk_audit PRIMARY KEY (id, changed_at),
    CONSTRAINT chk_audit_action CHECK (action IN ('INSERT', 'UPDATE', 'DELETE'))
)
PARTITION BY RANGE (changed_at)
INTERVAL (NUMTOYMINTERVAL(1, 'MONTH'))
(
    PARTITION p_audit_initial VALUES LESS THAN (TIMESTAMP '2025-01-01 00:00:00')
);

COMMENT ON TABLE AUDIT_TRAIL IS 'Complete audit trail for all regulatory record changes, partitioned by month';

CREATE INDEX idx_audit_table_record ON AUDIT_TRAIL(table_name, record_id) LOCAL;
CREATE INDEX idx_audit_changed_by ON AUDIT_TRAIL(changed_by) LOCAL;
CREATE INDEX idx_audit_changed_at ON AUDIT_TRAIL(changed_at) LOCAL;
CREATE INDEX idx_audit_action ON AUDIT_TRAIL(action) LOCAL;

-- ============================================================================
-- PSEUDO_RECORD (Core pseudo record keeping system)
-- ============================================================================

CREATE TABLE PSEUDO_RECORD (
    id                  NUMBER          DEFAULT seq_pseudo_id.NEXTVAL PRIMARY KEY,
    original_record_id  NUMBER          NOT NULL,
    source_table        VARCHAR2(50)    NOT NULL,
    pseudo_type         VARCHAR2(30)    NOT NULL,
    pseudo_value        VARCHAR2(256)   NOT NULL,
    mapping_key         VARCHAR2(128)   NOT NULL,
    hash_algorithm      VARCHAR2(20)    DEFAULT 'SHA-256' NOT NULL,
    salt                VARCHAR2(64),
    is_active           CHAR(1)         DEFAULT 'Y' CHECK (is_active IN ('Y', 'N')),
    created_date        TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL,
    expiry_date         TIMESTAMP,
    last_accessed       TIMESTAMP,
    access_count        NUMBER          DEFAULT 0,
    created_by          VARCHAR2(100)   NOT NULL,
    purpose             VARCHAR2(200),
    CONSTRAINT chk_pseudo_type CHECK (
        pseudo_type IN (
            'SUBMISSION_ID', 'DOCUMENT_ID', 'PATIENT_ID',
            'INVESTIGATOR_ID', 'SITE_ID', 'BATCH_NUMBER',
            'COMPOUND_CODE', 'PROTOCOL_NUMBER'
        )
    ),
    CONSTRAINT chk_pseudo_hash CHECK (
        hash_algorithm IN ('SHA-256', 'SHA-512', 'HMAC-SHA256', 'BLAKE2B')
    ),
    CONSTRAINT uk_pseudo_mapping UNIQUE (source_table, original_record_id, pseudo_type, mapping_key)
);

COMMENT ON TABLE PSEUDO_RECORD IS 'Pseudo record keeping system - maps real IDs to deterministic pseudo IDs for privacy and compliance';
COMMENT ON COLUMN PSEUDO_RECORD.mapping_key IS 'Cryptographic key used for deterministic mapping; required for re-identification';
COMMENT ON COLUMN PSEUDO_RECORD.salt IS 'Per-record salt for additional security in hash generation';

CREATE INDEX idx_pseudo_original ON PSEUDO_RECORD(original_record_id, source_table);
CREATE INDEX idx_pseudo_value ON PSEUDO_RECORD(pseudo_value);
CREATE INDEX idx_pseudo_mapping ON PSEUDO_RECORD(mapping_key);
CREATE INDEX idx_pseudo_type ON PSEUDO_RECORD(pseudo_type);
CREATE INDEX idx_pseudo_active ON PSEUDO_RECORD(is_active, expiry_date);
CREATE INDEX idx_pseudo_expiry ON PSEUDO_RECORD(expiry_date) WHERE is_active = 'Y';

-- ============================================================================
-- RE-IDENTIFICATION ACCESS LOG (controls who accessed real mappings)
-- ============================================================================

CREATE TABLE REIDENTIFICATION_LOG (
    id                  NUMBER          GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pseudo_record_id    NUMBER          NOT NULL,
    requested_by        VARCHAR2(100)   NOT NULL,
    authorized_by       VARCHAR2(100)   NOT NULL,
    request_reason      CLOB            NOT NULL,
    request_date        TIMESTAMP       DEFAULT SYSTIMESTAMP NOT NULL,
    approval_date       TIMESTAMP,
    status              VARCHAR2(20)    DEFAULT 'PENDING',
    ip_address          VARCHAR2(45),
    CONSTRAINT fk_reid_pseudo FOREIGN KEY (pseudo_record_id) REFERENCES PSEUDO_RECORD(id),
    CONSTRAINT chk_reid_status CHECK (status IN ('PENDING', 'APPROVED', 'DENIED', 'EXPIRED'))
);

COMMENT ON TABLE REIDENTIFICATION_LOG IS 'Audit log for re-identification requests; ensures controlled access to real record mappings';

CREATE INDEX idx_reid_pseudo ON REIDENTIFICATION_LOG(pseudo_record_id);
CREATE INDEX idx_reid_requested ON REIDENTIFICATION_LOG(requested_by);
CREATE INDEX idx_reid_status ON REIDENTIFICATION_LOG(status);

-- ============================================================================
-- GRANTS (example role-based access)
-- ============================================================================

-- CREATE ROLE reg_read_role;
-- CREATE ROLE reg_write_role;
-- CREATE ROLE reg_admin_role;
-- CREATE ROLE pseudo_admin_role;
--
-- GRANT SELECT ON REGULATORY_SUBMISSION TO reg_read_role;
-- GRANT SELECT ON SUBMISSION_DOCUMENT TO reg_read_role;
-- GRANT SELECT ON APPROVAL_RECORD TO reg_read_role;
-- GRANT SELECT ON COMPLIANCE_RECORD TO reg_read_role;
-- GRANT SELECT ON AUDIT_TRAIL TO reg_read_role;
--
-- GRANT INSERT, UPDATE ON REGULATORY_SUBMISSION TO reg_write_role;
-- GRANT INSERT, UPDATE ON SUBMISSION_DOCUMENT TO reg_write_role;
-- GRANT INSERT, UPDATE ON APPROVAL_RECORD TO reg_write_role;
-- GRANT INSERT, UPDATE ON COMPLIANCE_RECORD TO reg_write_role;
--
-- GRANT ALL ON PSEUDO_RECORD TO pseudo_admin_role;
-- GRANT ALL ON REIDENTIFICATION_LOG TO pseudo_admin_role;
