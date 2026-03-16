--------------------------------------------------------------------------------
-- RegRecord - Complex Reporting Queries
--
-- 1. Submission pipeline report (status, aging, SLA compliance)
-- 2. Drug approval timeline analysis
-- 3. Compliance gap analysis
-- 4. Audit trail investigation queries
-- 5. Pseudo record reconciliation
--------------------------------------------------------------------------------

-- ============================================================================
-- 1. SUBMISSION PIPELINE REPORT
-- Shows current submission pipeline with aging, SLA metrics, and bottlenecks
-- ============================================================================

-- 1a. Pipeline overview by status and agency
SELECT
    rs.agency,
    ag.agency_name,
    rs.status,
    COUNT(*)                                                    AS submission_count,
    ROUND(AVG(
        EXTRACT(DAY FROM (SYSTIMESTAMP - rs.created_date))
    ), 1)                                                       AS avg_age_days,
    MIN(rs.created_date)                                        AS oldest_submission,
    SUM(CASE WHEN rs.priority IN ('PRIORITY', 'BREAKTHROUGH', 'FAST_TRACK')
        THEN 1 ELSE 0 END)                                     AS priority_count,
    SUM(CASE
        WHEN rs.status = 'UNDER_REVIEW'
         AND EXTRACT(DAY FROM (SYSTIMESTAMP - rs.submitted_date)) > 180
        THEN 1 ELSE 0 END)                                     AS sla_breach_count
FROM REGULATORY_SUBMISSION rs
JOIN AGENCY ag ON ag.agency_code = rs.agency
GROUP BY rs.agency, ag.agency_name, rs.status
ORDER BY rs.agency, rs.status;

-- 1b. Detailed aging analysis with SLA tiers
SELECT
    rs.id                                                       AS submission_id,
    rs.tracking_number,
    d.drug_name,
    rs.submission_type,
    rs.agency,
    rs.status,
    rs.priority,
    rs.assigned_to,
    rs.submitted_date,
    EXTRACT(DAY FROM (SYSTIMESTAMP - rs.submitted_date))         AS days_since_submitted,
    CASE
        WHEN rs.status IN ('APPROVED', 'REJECTED', 'WITHDRAWN') THEN 'CLOSED'
        WHEN EXTRACT(DAY FROM (SYSTIMESTAMP - rs.submitted_date)) > 365 THEN 'CRITICAL'
        WHEN EXTRACT(DAY FROM (SYSTIMESTAMP - rs.submitted_date)) > 180 THEN 'SLA_BREACH'
        WHEN EXTRACT(DAY FROM (SYSTIMESTAMP - rs.submitted_date)) > 120 THEN 'AT_RISK'
        WHEN EXTRACT(DAY FROM (SYSTIMESTAMP - rs.submitted_date)) > 60  THEN 'ON_TRACK'
        ELSE 'EARLY'
    END                                                         AS sla_tier,
    rs.target_date,
    CASE
        WHEN rs.target_date IS NOT NULL
        THEN TRUNC(rs.target_date) - TRUNC(SYSDATE)
        ELSE NULL
    END                                                         AS days_to_target,
    (SELECT COUNT(*) FROM SUBMISSION_DOCUMENT sd
     WHERE sd.submission_id = rs.id AND sd.status = 'ACTIVE')   AS active_doc_count
FROM REGULATORY_SUBMISSION rs
JOIN DRUG d ON d.drug_id = rs.drug_id
WHERE rs.status NOT IN ('APPROVED', 'REJECTED', 'WITHDRAWN')
ORDER BY
    CASE rs.priority
        WHEN 'BREAKTHROUGH' THEN 1
        WHEN 'FAST_TRACK' THEN 2
        WHEN 'ACCELERATED' THEN 3
        WHEN 'PRIORITY' THEN 4
        ELSE 5
    END,
    rs.submitted_date ASC NULLS LAST;

-- 1c. Submission throughput analysis (rolling 12-month trend)
SELECT
    TO_CHAR(submitted_date, 'YYYY-MM')                          AS month,
    agency,
    COUNT(*)                                                    AS total_submitted,
    SUM(CASE WHEN status = 'APPROVED' THEN 1 ELSE 0 END)       AS approved,
    SUM(CASE WHEN status = 'REJECTED' THEN 1 ELSE 0 END)       AS rejected,
    ROUND(
        SUM(CASE WHEN status = 'APPROVED' THEN 1 ELSE 0 END) * 100.0 /
        NULLIF(SUM(CASE WHEN status IN ('APPROVED', 'REJECTED') THEN 1 ELSE 0 END), 0),
        1
    )                                                           AS approval_rate_pct,
    ROUND(AVG(
        CASE WHEN status = 'APPROVED' THEN
            EXTRACT(DAY FROM (
                (SELECT MIN(ar.approval_date) FROM APPROVAL_RECORD ar WHERE ar.submission_id = rs.id)
                - rs.submitted_date
            ))
        END
    ), 1)                                                       AS avg_approval_days
FROM REGULATORY_SUBMISSION rs
WHERE submitted_date >= ADD_MONTHS(SYSDATE, -12)
  AND submitted_date IS NOT NULL
GROUP BY TO_CHAR(submitted_date, 'YYYY-MM'), agency
ORDER BY month DESC, agency;

-- ============================================================================
-- 2. DRUG APPROVAL TIMELINE ANALYSIS
-- Full timeline of a drug's regulatory journey across agencies
-- ============================================================================

-- 2a. Drug regulatory timeline (cross-agency view)
SELECT
    d.drug_name,
    d.generic_name,
    d.therapeutic_area,
    rs.agency,
    rs.submission_type,
    rs.submitted_date,
    ar.approval_date,
    ar.approval_type,
    ar.market_exclusivity,
    ar.exclusivity_expiry,
    EXTRACT(DAY FROM (ar.approval_date - rs.submitted_date))    AS review_duration_days,
    ar.conditions                                               AS approval_conditions,
    RANK() OVER (
        PARTITION BY d.drug_id, rs.agency
        ORDER BY rs.submitted_date
    )                                                           AS submission_sequence,
    LEAD(rs.submitted_date) OVER (
        PARTITION BY d.drug_id, rs.agency
        ORDER BY rs.submitted_date
    )                                                           AS next_submission_date
FROM DRUG d
JOIN REGULATORY_SUBMISSION rs ON rs.drug_id = d.drug_id
LEFT JOIN APPROVAL_RECORD ar ON ar.submission_id = rs.id
WHERE rs.status IN ('APPROVED', 'UNDER_REVIEW', 'SUBMITTED')
ORDER BY d.drug_name, rs.agency, rs.submitted_date;

-- 2b. Comparative approval timelines across agencies
WITH approval_timeline AS (
    SELECT
        d.drug_id,
        d.drug_name,
        rs.agency,
        MIN(rs.submitted_date) AS first_submitted,
        MIN(ar.approval_date) AS first_approved,
        EXTRACT(DAY FROM (MIN(ar.approval_date) - MIN(rs.submitted_date))) AS days_to_approval
    FROM DRUG d
    JOIN REGULATORY_SUBMISSION rs ON rs.drug_id = d.drug_id
    LEFT JOIN APPROVAL_RECORD ar ON ar.submission_id = rs.id
    WHERE rs.submission_type IN ('NDA', 'BLA', 'MAA')
    GROUP BY d.drug_id, d.drug_name, rs.agency
)
SELECT
    at1.drug_name,
    at1.agency                                                  AS agency,
    at1.first_submitted,
    at1.first_approved,
    at1.days_to_approval,
    RANK() OVER (
        PARTITION BY at1.drug_id ORDER BY at1.first_approved NULLS LAST
    )                                                           AS approval_order,
    FIRST_VALUE(at1.agency) OVER (
        PARTITION BY at1.drug_id ORDER BY at1.first_approved NULLS LAST
    )                                                           AS first_approving_agency,
    at1.days_to_approval - AVG(at1.days_to_approval) OVER (
        PARTITION BY at1.drug_id
    )                                                           AS days_vs_avg
FROM approval_timeline at1
ORDER BY at1.drug_name, at1.first_approved NULLS LAST;

-- 2c. Labeling change history for a drug
SELECT
    d.drug_name,
    lc.change_type,
    lc.change_category,
    lc.effective_date,
    lc.status,
    lc.approved_by,
    lc.affected_sections,
    DBMS_LOB.SUBSTR(lc.description, 200, 1)                     AS change_summary,
    lc.rationale,
    LAG(lc.effective_date) OVER (
        PARTITION BY lc.drug_id ORDER BY lc.effective_date
    )                                                           AS previous_change_date,
    lc.effective_date - LAG(lc.effective_date) OVER (
        PARTITION BY lc.drug_id ORDER BY lc.effective_date
    )                                                           AS days_since_last_change
FROM LABELING_CHANGE lc
JOIN DRUG d ON d.drug_id = lc.drug_id
ORDER BY d.drug_name, lc.effective_date DESC;

-- ============================================================================
-- 3. COMPLIANCE GAP ANALYSIS
-- Identifies gaps, risks, and overdue requirements
-- ============================================================================

-- 3a. Compliance gap heatmap (drug x requirement type)
SELECT
    d.drug_name,
    cr.requirement_type,
    COUNT(*)                                                    AS total_requirements,
    SUM(CASE WHEN cr.status = 'COMPLETED' THEN 1 ELSE 0 END)   AS completed,
    SUM(CASE WHEN cr.status = 'OVERDUE' THEN 1 ELSE 0 END)     AS overdue,
    SUM(CASE WHEN cr.status = 'ESCALATED' THEN 1 ELSE 0 END)   AS escalated,
    SUM(CASE WHEN cr.status IN ('PENDING', 'IN_PROGRESS') THEN 1 ELSE 0 END) AS in_progress,
    ROUND(
        SUM(CASE WHEN cr.status = 'COMPLETED' THEN 1 ELSE 0 END) * 100.0 /
        NULLIF(COUNT(*), 0), 1
    )                                                           AS completion_pct,
    ROUND(AVG(cr.risk_score), 1)                                AS avg_risk_score,
    MAX(cr.escalation_level)                                    AS max_escalation,
    CASE
        WHEN SUM(CASE WHEN cr.status = 'ESCALATED' THEN 1 ELSE 0 END) > 0 THEN 'RED'
        WHEN SUM(CASE WHEN cr.status = 'OVERDUE' THEN 1 ELSE 0 END) > 0 THEN 'AMBER'
        WHEN SUM(CASE WHEN cr.risk_score > 70 THEN 1 ELSE 0 END) > 0 THEN 'YELLOW'
        ELSE 'GREEN'
    END                                                         AS rag_status
FROM COMPLIANCE_RECORD cr
JOIN DRUG d ON d.drug_id = cr.drug_id
WHERE cr.status NOT IN ('WAIVED', 'CANCELLED')
GROUP BY d.drug_name, cr.requirement_type
ORDER BY
    CASE
        WHEN SUM(CASE WHEN cr.status = 'ESCALATED' THEN 1 ELSE 0 END) > 0 THEN 1
        WHEN SUM(CASE WHEN cr.status = 'OVERDUE' THEN 1 ELSE 0 END) > 0 THEN 2
        ELSE 3
    END,
    d.drug_name;

-- 3b. Upcoming deadlines (next 90 days)
SELECT
    cr.id                                                       AS compliance_id,
    d.drug_name,
    cr.requirement_type,
    cr.due_date,
    TRUNC(cr.due_date) - TRUNC(SYSDATE)                         AS days_remaining,
    cr.status,
    cr.responsible_party,
    cr.risk_score,
    cr.regulatory_reference,
    CASE
        WHEN TRUNC(cr.due_date) - TRUNC(SYSDATE) <= 0  THEN 'OVERDUE'
        WHEN TRUNC(cr.due_date) - TRUNC(SYSDATE) <= 7  THEN 'CRITICAL'
        WHEN TRUNC(cr.due_date) - TRUNC(SYSDATE) <= 14 THEN 'URGENT'
        WHEN TRUNC(cr.due_date) - TRUNC(SYSDATE) <= 30 THEN 'HIGH'
        WHEN TRUNC(cr.due_date) - TRUNC(SYSDATE) <= 60 THEN 'MEDIUM'
        ELSE 'NORMAL'
    END                                                         AS urgency,
    (SELECT rs.tracking_number FROM REGULATORY_SUBMISSION rs
     WHERE rs.id = cr.linked_submission)                        AS linked_tracking
FROM COMPLIANCE_RECORD cr
JOIN DRUG d ON d.drug_id = cr.drug_id
WHERE cr.status IN ('PENDING', 'IN_PROGRESS')
  AND cr.due_date <= SYSDATE + 90
ORDER BY cr.due_date ASC;

-- 3c. Responsible party workload analysis
SELECT
    cr.responsible_party,
    COUNT(*)                                                    AS total_assignments,
    SUM(CASE WHEN cr.status = 'OVERDUE' THEN 1 ELSE 0 END)     AS overdue_count,
    SUM(CASE WHEN cr.due_date BETWEEN SYSDATE AND SYSDATE + 30 THEN 1 ELSE 0 END) AS due_within_30d,
    ROUND(AVG(cr.risk_score), 1)                                AS avg_risk_score,
    MAX(cr.escalation_level)                                    AS max_escalation,
    LISTAGG(DISTINCT d.drug_name, ', ')
        WITHIN GROUP (ORDER BY d.drug_name)                     AS assigned_drugs
FROM COMPLIANCE_RECORD cr
JOIN DRUG d ON d.drug_id = cr.drug_id
WHERE cr.status NOT IN ('COMPLETED', 'WAIVED', 'CANCELLED')
GROUP BY cr.responsible_party
ORDER BY overdue_count DESC, total_assignments DESC;

-- ============================================================================
-- 4. AUDIT TRAIL INVESTIGATION QUERIES
-- For regulatory inquiries, internal investigations, and compliance audits
-- ============================================================================

-- 4a. Full change history for a specific submission (parameterized)
-- Usage: Replace :submission_id with actual ID
SELECT
    at.changed_at,
    at.action,
    at.changed_by,
    at.ip_address,
    at.session_id,
    at.old_values,
    at.new_values,
    LAG(at.changed_at) OVER (ORDER BY at.changed_at)           AS previous_change,
    EXTRACT(MINUTE FROM (
        at.changed_at - LAG(at.changed_at) OVER (ORDER BY at.changed_at)
    ))                                                          AS minutes_since_last_change
FROM AUDIT_TRAIL at
WHERE at.table_name = 'REGULATORY_SUBMISSION'
  AND at.record_id = :submission_id
ORDER BY at.changed_at;

-- 4b. User activity investigation (who did what and when)
SELECT
    at.changed_by,
    at.table_name,
    at.action,
    COUNT(*)                                                    AS change_count,
    MIN(at.changed_at)                                          AS first_activity,
    MAX(at.changed_at)                                          AS last_activity,
    COUNT(DISTINCT at.ip_address)                               AS distinct_ips,
    COUNT(DISTINCT TRUNC(at.changed_at))                        AS active_days,
    LISTAGG(DISTINCT at.ip_address, ', ')
        WITHIN GROUP (ORDER BY at.ip_address)                   AS ip_addresses
FROM AUDIT_TRAIL at
WHERE at.changed_at >= SYSTIMESTAMP - INTERVAL '30' DAY
GROUP BY at.changed_by, at.table_name, at.action
ORDER BY at.changed_by, change_count DESC;

-- 4c. Suspicious activity detection
-- Changes outside business hours, rapid successive changes, bulk deletions
SELECT
    at.changed_by,
    at.table_name,
    at.action,
    at.changed_at,
    at.ip_address,
    at.record_id,
    CASE
        WHEN EXTRACT(HOUR FROM at.changed_at) < 6
          OR EXTRACT(HOUR FROM at.changed_at) > 22
        THEN 'OFF_HOURS'
        WHEN at.action = 'DELETE' THEN 'DELETION'
        ELSE 'NORMAL'
    END                                                         AS flag_type,
    COUNT(*) OVER (
        PARTITION BY at.changed_by
        ORDER BY at.changed_at
        RANGE BETWEEN INTERVAL '5' MINUTE PRECEDING AND CURRENT ROW
    )                                                           AS changes_in_5min_window
FROM AUDIT_TRAIL at
WHERE at.changed_at >= SYSTIMESTAMP - INTERVAL '7' DAY
  AND (
    -- Off-hours activity
    EXTRACT(HOUR FROM at.changed_at) < 6 OR
    EXTRACT(HOUR FROM at.changed_at) > 22 OR
    -- Deletions
    at.action = 'DELETE'
  )
ORDER BY at.changed_at DESC;

-- 4d. Data integrity check: compare audit trail with current state
SELECT
    rs.id                                                       AS submission_id,
    rs.tracking_number,
    rs.status                                                   AS current_status,
    (SELECT COUNT(*) FROM AUDIT_TRAIL at
     WHERE at.table_name = 'REGULATORY_SUBMISSION'
       AND at.record_id = rs.id)                                AS total_changes,
    (SELECT MAX(at.changed_at) FROM AUDIT_TRAIL at
     WHERE at.table_name = 'REGULATORY_SUBMISSION'
       AND at.record_id = rs.id)                                AS last_audit_entry,
    rs.modified_date                                            AS last_modified,
    CASE
        WHEN rs.modified_date > (
            SELECT MAX(at.changed_at) FROM AUDIT_TRAIL at
            WHERE at.table_name = 'REGULATORY_SUBMISSION'
              AND at.record_id = rs.id
        ) THEN 'AUDIT_GAP'
        ELSE 'OK'
    END                                                         AS integrity_status
FROM REGULATORY_SUBMISSION rs
ORDER BY integrity_status DESC, rs.id;

-- ============================================================================
-- 5. PSEUDO RECORD RECONCILIATION
-- Ensures pseudo record mappings are consistent and complete
-- ============================================================================

-- 5a. Pseudo record coverage report
SELECT
    pr.source_table,
    pr.pseudo_type,
    COUNT(*)                                                    AS total_mappings,
    SUM(CASE WHEN pr.is_active = 'Y' THEN 1 ELSE 0 END)       AS active_mappings,
    SUM(CASE WHEN pr.is_active = 'N' THEN 1 ELSE 0 END)       AS inactive_mappings,
    SUM(CASE
        WHEN pr.expiry_date IS NOT NULL AND pr.expiry_date < SYSTIMESTAMP
        THEN 1 ELSE 0 END)                                     AS expired_not_deactivated,
    COUNT(DISTINCT pr.mapping_key)                              AS unique_keys,
    MIN(pr.created_date)                                        AS oldest_mapping,
    MAX(pr.created_date)                                        AS newest_mapping,
    ROUND(AVG(pr.access_count), 1)                              AS avg_access_count,
    MAX(pr.access_count)                                        AS max_access_count
FROM PSEUDO_RECORD pr
GROUP BY pr.source_table, pr.pseudo_type
ORDER BY pr.source_table, pr.pseudo_type;

-- 5b. Orphaned pseudo records (no matching source record)
SELECT
    pr.id                                                       AS pseudo_id,
    pr.pseudo_value,
    pr.original_record_id,
    pr.source_table,
    pr.pseudo_type,
    pr.created_date,
    pr.is_active,
    'ORPHANED'                                                  AS reconciliation_status
FROM PSEUDO_RECORD pr
WHERE pr.source_table = 'REGULATORY_SUBMISSION'
  AND NOT EXISTS (
      SELECT 1 FROM REGULATORY_SUBMISSION rs WHERE rs.id = pr.original_record_id
  )
UNION ALL
SELECT
    pr.id, pr.pseudo_value, pr.original_record_id,
    pr.source_table, pr.pseudo_type, pr.created_date,
    pr.is_active, 'ORPHANED'
FROM PSEUDO_RECORD pr
WHERE pr.source_table = 'SUBMISSION_DOCUMENT'
  AND NOT EXISTS (
      SELECT 1 FROM SUBMISSION_DOCUMENT sd WHERE sd.id = pr.original_record_id
  )
ORDER BY source_table, pseudo_id;

-- 5c. Records without pseudo IDs (coverage gaps)
SELECT
    'REGULATORY_SUBMISSION'                                     AS table_name,
    rs.id                                                       AS record_id,
    rs.tracking_number,
    rs.created_date,
    'MISSING_PSEUDO'                                            AS gap_type
FROM REGULATORY_SUBMISSION rs
WHERE NOT EXISTS (
    SELECT 1 FROM PSEUDO_RECORD pr
    WHERE pr.original_record_id = rs.id
      AND pr.source_table = 'REGULATORY_SUBMISSION'
      AND pr.is_active = 'Y'
)
ORDER BY rs.created_date DESC;

-- 5d. Re-identification access audit
SELECT
    rl.request_date,
    rl.requested_by,
    rl.authorized_by,
    rl.status                                                   AS request_status,
    rl.request_reason,
    pr.pseudo_value,
    pr.source_table,
    pr.pseudo_type,
    pr.original_record_id,
    pr.access_count                                             AS total_accesses,
    rl.ip_address
FROM REIDENTIFICATION_LOG rl
JOIN PSEUDO_RECORD pr ON pr.id = rl.pseudo_record_id
ORDER BY rl.request_date DESC;

-- 5e. Pseudo record key rotation status
SELECT
    pr.mapping_key,
    pr.source_table,
    COUNT(*)                                                    AS mapping_count,
    SUM(CASE WHEN pr.is_active = 'Y' THEN 1 ELSE 0 END)       AS active_count,
    MIN(pr.created_date)                                        AS key_first_used,
    MAX(pr.created_date)                                        AS key_last_used,
    EXTRACT(DAY FROM (SYSTIMESTAMP - MIN(pr.created_date)))     AS key_age_days,
    CASE
        WHEN EXTRACT(DAY FROM (SYSTIMESTAMP - MIN(pr.created_date))) > 365
        THEN 'ROTATION_REQUIRED'
        WHEN EXTRACT(DAY FROM (SYSTIMESTAMP - MIN(pr.created_date))) > 270
        THEN 'ROTATION_RECOMMENDED'
        ELSE 'CURRENT'
    END                                                         AS rotation_status
FROM PSEUDO_RECORD pr
WHERE pr.is_active = 'Y'
GROUP BY pr.mapping_key, pr.source_table
ORDER BY key_age_days DESC;
