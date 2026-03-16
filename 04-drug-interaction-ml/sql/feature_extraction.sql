-- =============================================================================
-- DrugInteractionML: Complex Snowflake SQL for Drug Interaction Features
-- =============================================================================
-- This script extracts a comprehensive feature set for drug-drug interaction
-- prediction. It combines prescription data, patient demographics, adverse
-- events, and molecular properties into a single feature matrix.
-- =============================================================================

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. Co-prescription features with temporal overlap analysis
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE TABLE PHARMA_DB.DRUG_INTERACTION.co_prescription_features AS
WITH prescription_windows AS (
    SELECT
        p.patient_id,
        p.drug_ndc,
        p.prescribe_date,
        p.dosage_mg,
        p.duration_days,
        DATEADD('day', p.duration_days, p.prescribe_date) AS end_date,
        -- Running count of active prescriptions per patient at each point
        COUNT(*) OVER (
            PARTITION BY p.patient_id
            ORDER BY p.prescribe_date
            RANGE BETWEEN INTERVAL '30 DAYS' PRECEDING AND CURRENT ROW
        ) AS concurrent_rx_count,
        -- Previous prescription gap for same drug
        DATEDIFF('day',
            LAG(p.prescribe_date) OVER (
                PARTITION BY p.patient_id, p.drug_ndc
                ORDER BY p.prescribe_date
            ),
            p.prescribe_date
        ) AS days_since_last_rx
    FROM PHARMA_DB.DRUG_INTERACTION.prescriptions p
    WHERE p.prescribe_date >= :start_date
      AND p.prescribe_date < :end_date
      AND p.status = 'ACTIVE'
),
drug_pairs AS (
    SELECT
        pw1.patient_id,
        pw1.drug_ndc AS drug_a_ndc,
        pw2.drug_ndc AS drug_b_ndc,
        pw1.prescribe_date AS rx_date_a,
        pw2.prescribe_date AS rx_date_b,
        pw1.dosage_mg AS dosage_a,
        pw2.dosage_mg AS dosage_b,
        pw1.duration_days AS duration_a,
        pw2.duration_days AS duration_b,
        -- Overlap calculation
        GREATEST(0,
            DATEDIFF('day',
                GREATEST(pw1.prescribe_date, pw2.prescribe_date),
                LEAST(pw1.end_date, pw2.end_date)
            )
        ) AS overlap_days,
        ABS(DATEDIFF('day', pw1.prescribe_date, pw2.prescribe_date)) AS prescription_gap,
        pw1.concurrent_rx_count AS polypharmacy_a,
        pw2.concurrent_rx_count AS polypharmacy_b
    FROM prescription_windows pw1
    JOIN prescription_windows pw2
        ON pw1.patient_id = pw2.patient_id
        AND pw1.drug_ndc < pw2.drug_ndc
        AND ABS(DATEDIFF('day', pw1.prescribe_date, pw2.prescribe_date)) <= 30
),
pair_aggregates AS (
    SELECT
        drug_a_ndc,
        drug_b_ndc,
        COUNT(DISTINCT patient_id) AS n_patients,
        COUNT(*) AS n_coprescriptions,
        AVG(overlap_days) AS avg_overlap_days,
        STDDEV(overlap_days) AS std_overlap_days,
        MAX(overlap_days) AS max_overlap_days,
        AVG(prescription_gap) AS avg_gap_days,
        AVG(dosage_a) AS avg_dosage_a,
        AVG(dosage_b) AS avg_dosage_b,
        STDDEV(dosage_a) AS std_dosage_a,
        STDDEV(dosage_b) AS std_dosage_b,
        AVG(duration_a) AS avg_duration_a,
        AVG(duration_b) AS avg_duration_b,
        AVG(polypharmacy_a + polypharmacy_b) / 2.0 AS avg_polypharmacy,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY overlap_days) AS overlap_p25,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY overlap_days) AS overlap_p50,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY overlap_days) AS overlap_p75,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY overlap_days) AS overlap_p95,
        -- Time trend: is the co-prescription increasing?
        REGR_SLOPE(n_coprescriptions_weekly, week_num) AS coprescription_trend
    FROM (
        SELECT
            dp.*,
            DATE_TRUNC('week', rx_date_a) AS week_start,
            DATEDIFF('week', :start_date, rx_date_a) AS week_num,
            COUNT(*) OVER (
                PARTITION BY dp.drug_a_ndc, dp.drug_b_ndc, DATE_TRUNC('week', rx_date_a)
            ) AS n_coprescriptions_weekly
        FROM drug_pairs dp
    )
    GROUP BY drug_a_ndc, drug_b_ndc
    HAVING n_patients >= 5  -- minimum support threshold
)
SELECT * FROM pair_aggregates;


-- ─────────────────────────────────────────────────────────────────────────────
-- 2. Patient demographics and clinical features per drug pair
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE TABLE PHARMA_DB.DRUG_INTERACTION.patient_features AS
WITH drug_pair_patients AS (
    SELECT DISTINCT
        p1.drug_ndc AS drug_a_ndc,
        p2.drug_ndc AS drug_b_ndc,
        pt.patient_id,
        pt.age,
        pt.sex,
        pt.weight_kg,
        pt.height_cm,
        pt.bmi,
        pt.ethnicity,
        pt.renal_function_egfr,
        pt.hepatic_function_class,
        -- Semi-structured comorbidity parsing
        ARRAY_SIZE(pt.comorbidities:conditions) AS n_comorbidities,
        pt.comorbidities:polypharmacy_count::INT AS polypharmacy_count,
        -- Check for specific high-risk comorbidities
        ARRAY_CONTAINS('E11'::VARIANT, pt.comorbidities:icd_codes) AS has_diabetes,
        ARRAY_CONTAINS('I10'::VARIANT, pt.comorbidities:icd_codes) AS has_hypertension,
        ARRAY_CONTAINS('N18'::VARIANT, pt.comorbidities:icd_codes) AS has_ckd,
        ARRAY_CONTAINS('K70'::VARIANT, pt.comorbidities:icd_codes) AS has_liver_disease,
        -- Lab values from semi-structured data
        pt.lab_results:creatinine::FLOAT AS creatinine,
        pt.lab_results:alt::FLOAT AS alt_liver,
        pt.lab_results:ast::FLOAT AS ast_liver,
        pt.lab_results:platelets::FLOAT AS platelets,
        pt.lab_results:inr::FLOAT AS inr,
        pt.lab_results:albumin::FLOAT AS albumin
    FROM PHARMA_DB.DRUG_INTERACTION.prescriptions p1
    JOIN PHARMA_DB.DRUG_INTERACTION.prescriptions p2
        ON p1.patient_id = p2.patient_id
        AND p1.drug_ndc < p2.drug_ndc
        AND ABS(DATEDIFF('day', p1.prescribe_date, p2.prescribe_date)) <= 30
    JOIN PHARMA_DB.DRUG_INTERACTION.patients pt
        ON p1.patient_id = pt.patient_id
    WHERE p1.prescribe_date >= :start_date
      AND p1.prescribe_date < :end_date
)
SELECT
    drug_a_ndc,
    drug_b_ndc,
    COUNT(DISTINCT patient_id) AS n_patients,
    -- Age statistics
    AVG(age) AS avg_age,
    STDDEV(age) AS std_age,
    MIN(age) AS min_age,
    MAX(age) AS max_age,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY age) AS median_age,
    SUM(CASE WHEN age >= 65 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS pct_elderly,
    -- Sex distribution
    SUM(CASE WHEN sex = 'F' THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS pct_female,
    -- Body metrics
    AVG(bmi) AS avg_bmi,
    STDDEV(bmi) AS std_bmi,
    SUM(CASE WHEN bmi >= 30 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS pct_obese,
    -- Renal function
    AVG(renal_function_egfr) AS avg_egfr,
    SUM(CASE WHEN renal_function_egfr < 60 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS pct_renal_impaired,
    -- Hepatic function
    SUM(CASE WHEN hepatic_function_class IN ('B', 'C') THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS pct_hepatic_impaired,
    -- Comorbidity burden
    AVG(n_comorbidities) AS avg_comorbidities,
    AVG(polypharmacy_count) AS avg_polypharmacy,
    -- High-risk comorbidity prevalence
    SUM(CASE WHEN has_diabetes THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS pct_diabetes,
    SUM(CASE WHEN has_hypertension THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS pct_hypertension,
    SUM(CASE WHEN has_ckd THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS pct_ckd,
    SUM(CASE WHEN has_liver_disease THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS pct_liver_disease,
    -- Lab value averages
    AVG(creatinine) AS avg_creatinine,
    AVG(alt_liver) AS avg_alt,
    AVG(ast_liver) AS avg_ast,
    AVG(platelets) AS avg_platelets,
    AVG(inr) AS avg_inr,
    AVG(albumin) AS avg_albumin,
    -- Ethnicity diversity (Shannon entropy proxy)
    COUNT(DISTINCT ethnicity) AS n_ethnicities
FROM drug_pair_patients
GROUP BY drug_a_ndc, drug_b_ndc;


-- ─────────────────────────────────────────────────────────────────────────────
-- 3. Adverse event signal detection with disproportionality analysis
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE TABLE PHARMA_DB.DRUG_INTERACTION.adverse_event_features AS
WITH ae_pair_data AS (
    SELECT
        ae.drug_a_ndc,
        ae.drug_b_ndc,
        ae.event_type,
        ae.severity,
        ae.outcome,
        ae.reported_date,
        ae.patient_id,
        -- Semi-structured event details
        ae.details:mechanism::STRING AS mechanism,
        ae.details:affected_organ::STRING AS affected_organ,
        ae.details:onset_days::INT AS onset_days,
        ae.details:dechallenge_positive::BOOLEAN AS dechallenge_positive,
        ae.details:rechallenge_positive::BOOLEAN AS rechallenge_positive,
        -- Cumulative event tracking
        ROW_NUMBER() OVER (
            PARTITION BY ae.drug_a_ndc, ae.drug_b_ndc
            ORDER BY ae.reported_date
        ) AS event_sequence,
        DATEDIFF('day',
            LAG(ae.reported_date) OVER (
                PARTITION BY ae.drug_a_ndc, ae.drug_b_ndc
                ORDER BY ae.reported_date
            ),
            ae.reported_date
        ) AS days_between_events
    FROM PHARMA_DB.DRUG_INTERACTION.adverse_events ae
    WHERE ae.reported_date >= :start_date
      AND ae.reported_date < :end_date
),
-- Background rates for PRR calculation
background_rates AS (
    SELECT
        event_type,
        COUNT(*) AS total_events,
        COUNT(DISTINCT drug_a_ndc || '-' || drug_b_ndc) AS total_pairs,
        COUNT(DISTINCT patient_id) AS total_patients
    FROM ae_pair_data
    GROUP BY event_type
),
pair_signals AS (
    SELECT
        ae.drug_a_ndc,
        ae.drug_b_ndc,
        COUNT(*) AS total_ae_count,
        COUNT(DISTINCT ae.patient_id) AS ae_patients,
        -- Severity breakdown
        SUM(CASE WHEN ae.severity = 'SEVERE' THEN 1 ELSE 0 END) AS severe_count,
        SUM(CASE WHEN ae.severity = 'MODERATE' THEN 1 ELSE 0 END) AS moderate_count,
        SUM(CASE WHEN ae.severity = 'MILD' THEN 1 ELSE 0 END) AS mild_count,
        -- Outcome breakdown
        SUM(CASE WHEN ae.outcome = 'HOSPITALIZATION' THEN 1 ELSE 0 END) AS hospitalizations,
        SUM(CASE WHEN ae.outcome = 'DEATH' THEN 1 ELSE 0 END) AS deaths,
        SUM(CASE WHEN ae.outcome = 'DISABILITY' THEN 1 ELSE 0 END) AS disabilities,
        -- Timing features
        AVG(ae.onset_days) AS avg_onset_days,
        STDDEV(ae.onset_days) AS std_onset_days,
        MIN(ae.onset_days) AS min_onset_days,
        AVG(ae.days_between_events) AS avg_inter_event_days,
        -- Causality indicators
        SUM(CASE WHEN ae.dechallenge_positive THEN 1 ELSE 0 END)::FLOAT
            / NULLIF(COUNT(*), 0) AS dechallenge_rate,
        SUM(CASE WHEN ae.rechallenge_positive THEN 1 ELSE 0 END)::FLOAT
            / NULLIF(COUNT(*), 0) AS rechallenge_rate,
        -- Mechanism and organ
        MODE(ae.mechanism) AS primary_mechanism,
        MODE(ae.affected_organ) AS primary_affected_organ,
        COUNT(DISTINCT ae.mechanism) AS n_mechanisms,
        COUNT(DISTINCT ae.affected_organ) AS n_affected_organs,
        -- Event acceleration (are events getting more frequent?)
        REGR_SLOPE(ae.event_sequence, DATEDIFF('day', :start_date, ae.reported_date)) AS event_acceleration
    FROM ae_pair_data ae
    GROUP BY ae.drug_a_ndc, ae.drug_b_ndc
),
-- Compute disproportionality metrics
prr_calculation AS (
    SELECT
        ps.*,
        -- Proportional Reporting Ratio per event type
        bg.total_events AS bg_total_events,
        bg.total_pairs AS bg_total_pairs,
        (ps.total_ae_count::FLOAT / NULLIF(ps.ae_patients, 0))
            / NULLIF(bg.total_events::FLOAT / NULLIF(bg.total_patients, 0), 0) AS prr,
        -- Reporting Odds Ratio approximation
        LN(
            (ps.total_ae_count::FLOAT * (bg.total_patients - ps.ae_patients))
            / NULLIF((bg.total_events - ps.total_ae_count) * ps.ae_patients::FLOAT, 0)
        ) AS log_ror
    FROM pair_signals ps
    CROSS JOIN (
        SELECT SUM(total_events) AS total_events,
               SUM(total_pairs) AS total_pairs,
               SUM(total_patients) AS total_patients
        FROM background_rates
    ) bg
)
SELECT * FROM prr_calculation;


-- ─────────────────────────────────────────────────────────────────────────────
-- 4. Final feature matrix: join all feature sets
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE TABLE PHARMA_DB.DRUG_INTERACTION.feature_matrix AS
SELECT
    cp.drug_a_ndc,
    cp.drug_b_ndc,
    -- Co-prescription features
    cp.n_patients,
    cp.n_coprescriptions,
    cp.avg_overlap_days,
    cp.std_overlap_days,
    cp.max_overlap_days,
    cp.avg_gap_days,
    cp.avg_dosage_a,
    cp.avg_dosage_b,
    cp.avg_duration_a,
    cp.avg_duration_b,
    cp.avg_polypharmacy,
    cp.overlap_p50 AS median_overlap,
    cp.coprescription_trend,
    -- Patient features
    pf.avg_age,
    pf.std_age,
    pf.pct_elderly,
    pf.pct_female,
    pf.avg_bmi,
    pf.pct_obese,
    pf.avg_egfr,
    pf.pct_renal_impaired,
    pf.pct_hepatic_impaired,
    pf.avg_comorbidities,
    pf.avg_polypharmacy AS patient_polypharmacy,
    pf.pct_diabetes,
    pf.pct_hypertension,
    pf.pct_ckd,
    pf.pct_liver_disease,
    pf.avg_creatinine,
    pf.avg_alt,
    pf.avg_inr,
    -- Adverse event features
    COALESCE(ae.total_ae_count, 0) AS ae_count,
    COALESCE(ae.ae_patients, 0) AS ae_patients,
    COALESCE(ae.severe_count, 0) AS severe_ae_count,
    COALESCE(ae.hospitalizations, 0) AS hospitalizations,
    COALESCE(ae.deaths, 0) AS deaths,
    COALESCE(ae.avg_onset_days, -1) AS avg_onset_days,
    COALESCE(ae.dechallenge_rate, 0) AS dechallenge_rate,
    COALESCE(ae.rechallenge_rate, 0) AS rechallenge_rate,
    COALESCE(ae.prr, 0) AS prr,
    COALESCE(ae.log_ror, 0) AS log_ror,
    COALESCE(ae.n_mechanisms, 0) AS n_mechanisms,
    COALESCE(ae.event_acceleration, 0) AS event_acceleration,
    -- Derived risk scores
    (COALESCE(ae.severe_count, 0) + COALESCE(ae.hospitalizations, 0) * 2 + COALESCE(ae.deaths, 0) * 5)
        / NULLIF(cp.n_patients, 0)::FLOAT AS severity_risk_score,
    CASE
        WHEN ae.prr > 2.0 AND ae.total_ae_count >= 3 AND ae.dechallenge_rate > 0.5 THEN 'HIGH_SIGNAL'
        WHEN ae.prr > 1.5 AND ae.total_ae_count >= 2 THEN 'MODERATE_SIGNAL'
        WHEN ae.total_ae_count >= 1 THEN 'LOW_SIGNAL'
        ELSE 'NO_SIGNAL'
    END AS safety_signal_class
FROM PHARMA_DB.DRUG_INTERACTION.co_prescription_features cp
LEFT JOIN PHARMA_DB.DRUG_INTERACTION.patient_features pf
    ON cp.drug_a_ndc = pf.drug_a_ndc AND cp.drug_b_ndc = pf.drug_b_ndc
LEFT JOIN PHARMA_DB.DRUG_INTERACTION.adverse_event_features ae
    ON cp.drug_a_ndc = ae.drug_a_ndc AND cp.drug_b_ndc = ae.drug_b_ndc
ORDER BY COALESCE(ae.prr, 0) DESC;
