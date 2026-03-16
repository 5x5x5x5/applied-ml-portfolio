-- =============================================================================
-- Patient demographic features
-- Extracts age, gender, insurance, geographic, and chronic condition features
-- using window functions, aggregations, and multi-table joins.
-- =============================================================================

WITH patient_base AS (
    SELECT
        p.patient_id,
        p.date_of_birth,
        p.gender,
        p.race,
        p.ethnicity,
        p.zip_code,
        p.state,
        p.insurance_type,
        p.primary_care_provider_id,
        p.enrollment_date,
        DATEDIFF('year', p.date_of_birth, %(as_of_date)s::DATE) AS age,
        DATEDIFF('day', p.enrollment_date, %(as_of_date)s::DATE) AS days_enrolled
    FROM patients p
    WHERE p.is_active = TRUE
      -- PATIENT_ID_FILTER
),

-- Chronic condition counts from the condition table
chronic_conditions AS (
    SELECT
        c.patient_id,
        COUNT(DISTINCT c.condition_code) AS chronic_condition_count,
        COUNT(DISTINCT c.condition_category) AS condition_category_count,
        -- Specific condition flags
        MAX(CASE WHEN c.condition_category = 'DIABETES' THEN 1 ELSE 0 END)
            AS has_diabetes,
        MAX(CASE WHEN c.condition_category = 'HYPERTENSION' THEN 1 ELSE 0 END)
            AS has_hypertension,
        MAX(CASE WHEN c.condition_category = 'HEART_FAILURE' THEN 1 ELSE 0 END)
            AS has_heart_failure,
        MAX(CASE WHEN c.condition_category = 'COPD' THEN 1 ELSE 0 END)
            AS has_copd,
        MAX(CASE WHEN c.condition_category = 'CKD' THEN 1 ELSE 0 END)
            AS has_ckd,
        -- Charlson comorbidity approximation
        SUM(
            CASE
                WHEN c.condition_category IN ('DIABETES') THEN 1
                WHEN c.condition_category IN ('HEART_FAILURE', 'MI') THEN 2
                WHEN c.condition_category IN ('CKD', 'LIVER_DISEASE') THEN 3
                WHEN c.condition_category IN ('CANCER', 'METASTATIC') THEN 6
                ELSE 0
            END
        ) AS charlson_score_approx
    FROM conditions c
    WHERE c.onset_date <= %(as_of_date)s::DATE
      AND (c.resolved_date IS NULL OR c.resolved_date > %(as_of_date)s::DATE)
    GROUP BY c.patient_id
),

-- Encounter history aggregates
encounter_history AS (
    SELECT
        e.patient_id,
        COUNT(*) AS total_encounters_12m,
        COUNT(CASE WHEN e.encounter_type = 'INPATIENT' THEN 1 END)
            AS inpatient_count_12m,
        COUNT(CASE WHEN e.encounter_type = 'EMERGENCY' THEN 1 END)
            AS ed_visit_count_12m,
        COUNT(CASE WHEN e.encounter_type = 'OUTPATIENT' THEN 1 END)
            AS outpatient_count_12m,
        SUM(COALESCE(e.length_of_stay, 0)) AS total_los_days_12m,
        AVG(COALESCE(e.length_of_stay, 0)) AS avg_los_days_12m,
        MAX(e.encounter_date) AS last_encounter_date,
        DATEDIFF('day', MAX(e.encounter_date), %(as_of_date)s::DATE)
            AS days_since_last_encounter,
        -- 30-day readmission flag using window function
        MAX(
            CASE
                WHEN e.encounter_type = 'INPATIENT'
                     AND LAG(e.discharge_date) OVER (
                         PARTITION BY e.patient_id
                         ORDER BY e.encounter_date
                     ) IS NOT NULL
                     AND DATEDIFF(
                         'day',
                         LAG(e.discharge_date) OVER (
                             PARTITION BY e.patient_id
                             ORDER BY e.encounter_date
                         ),
                         e.encounter_date
                     ) <= 30
                THEN 1
                ELSE 0
            END
        ) AS had_30day_readmission
    FROM encounters e
    WHERE e.encounter_date >= DATEADD('month', -12, %(as_of_date)s::DATE)
      AND e.encounter_date <= %(as_of_date)s::DATE
    GROUP BY e.patient_id
),

-- Geographic risk score (based on zip-level socioeconomic data)
geo_risk AS (
    SELECT
        z.zip_code,
        z.area_deprivation_index AS adi_score,
        z.median_household_income,
        z.uninsured_rate,
        z.rural_urban_code,
        NTILE(10) OVER (ORDER BY z.area_deprivation_index) AS adi_decile
    FROM zip_demographics z
),

-- Provider panel size (for primary care attribution)
provider_info AS (
    SELECT
        pr.provider_id,
        COUNT(DISTINCT pr.patient_id) AS panel_size,
        AVG(pr.quality_score) AS avg_quality_score
    FROM provider_patients pr
    WHERE pr.is_active = TRUE
    GROUP BY pr.provider_id
)

SELECT
    pb.patient_id,
    pb.age,
    pb.gender,
    pb.race,
    pb.ethnicity,
    pb.insurance_type,
    pb.days_enrolled,

    -- Chronic condition features
    COALESCE(cc.chronic_condition_count, 0) AS chronic_condition_count,
    COALESCE(cc.condition_category_count, 0) AS condition_category_count,
    COALESCE(cc.has_diabetes, 0) AS has_diabetes,
    COALESCE(cc.has_hypertension, 0) AS has_hypertension,
    COALESCE(cc.has_heart_failure, 0) AS has_heart_failure,
    COALESCE(cc.has_copd, 0) AS has_copd,
    COALESCE(cc.has_ckd, 0) AS has_ckd,
    COALESCE(cc.charlson_score_approx, 0) AS charlson_score_approx,

    -- Encounter features
    COALESCE(eh.total_encounters_12m, 0) AS total_encounters_12m,
    COALESCE(eh.inpatient_count_12m, 0) AS inpatient_count_12m,
    COALESCE(eh.ed_visit_count_12m, 0) AS ed_visit_count_12m,
    COALESCE(eh.outpatient_count_12m, 0) AS outpatient_count_12m,
    COALESCE(eh.total_los_days_12m, 0) AS total_los_days_12m,
    COALESCE(eh.avg_los_days_12m, 0) AS avg_los_days_12m,
    COALESCE(eh.days_since_last_encounter, 9999) AS days_since_last_encounter,
    COALESCE(eh.had_30day_readmission, 0) AS had_30day_readmission,

    -- Geographic features
    COALESCE(gr.adi_score, 50) AS adi_score,
    COALESCE(gr.adi_decile, 5) AS adi_decile,
    COALESCE(gr.median_household_income, 0) AS median_household_income,
    COALESCE(gr.rural_urban_code, 0) AS rural_urban_code,

    -- Provider features
    COALESCE(pi.panel_size, 0) AS pcp_panel_size,
    COALESCE(pi.avg_quality_score, 0) AS pcp_quality_score,

    CURRENT_TIMESTAMP() AS feature_ts

FROM patient_base pb
LEFT JOIN chronic_conditions cc ON pb.patient_id = cc.patient_id
LEFT JOIN encounter_history eh ON pb.patient_id = eh.patient_id
LEFT JOIN geo_risk gr ON pb.zip_code = gr.zip_code
LEFT JOIN provider_info pi ON pb.primary_care_provider_id = pi.provider_id
ORDER BY pb.patient_id;
