-- =============================================================================
-- Semi-structured feature extraction from JSON VARIANT columns
-- Uses LATERAL FLATTEN, PARSE_JSON, and path-based extraction to build
-- features from clinical notes, medical records, and nested lab panels.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. Clinical note features from VARIANT column
-- ---------------------------------------------------------------------------
WITH note_flattened AS (
    SELECT
        cn.patient_id,
        cn.encounter_id,
        cn.note_date,
        -- Top-level VARIANT field extraction
        cn.clinical_note:noteType::STRING                       AS note_type,
        cn.clinical_note:provider.specialty::STRING              AS provider_specialty,
        cn.clinical_note:provider.npi::STRING                   AS provider_npi,
        cn.clinical_note:content.text::STRING                   AS note_text,
        LENGTH(cn.clinical_note:content.text::STRING)           AS note_length,
        cn.clinical_note:metadata.priority::STRING              AS priority,
        cn.clinical_note:metadata.is_addendum::BOOLEAN          AS is_addendum,
        cn.clinical_note:metadata.created_by::STRING            AS created_by,
        cn.clinical_note:content.sections                       AS sections_variant
    FROM clinical_notes cn
    WHERE cn.note_date >= DATEADD('day', -%(lookback_days)s, %(as_of_date)s::DATE)
      AND cn.note_date <= %(as_of_date)s::DATE
),

-- Flatten nested diagnosis arrays within sections
diagnoses_unnested AS (
    SELECT
        nf.patient_id,
        nf.encounter_id,
        nf.note_date,
        dx.value:code::STRING               AS dx_code,
        dx.value:description::STRING         AS dx_description,
        dx.value:type::STRING                AS dx_type,
        dx.value:is_primary::BOOLEAN         AS is_primary,
        dx.value:onset_date::DATE            AS dx_onset_date
    FROM note_flattened nf,
        LATERAL FLATTEN(
            input => PARSE_JSON(nf.sections_variant):diagnoses,
            outer => TRUE
        ) dx
    WHERE dx.value IS NOT NULL
),

-- Flatten nested medication references
medication_mentions AS (
    SELECT
        nf.patient_id,
        nf.encounter_id,
        med.value:drug_name::STRING          AS drug_name,
        med.value:ndc_code::STRING           AS ndc_code,
        med.value:action::STRING             AS med_action,  -- STARTED, CONTINUED, DISCONTINUED
        med.value:dosage::STRING             AS dosage
    FROM note_flattened nf,
        LATERAL FLATTEN(
            input => PARSE_JSON(nf.sections_variant):medications,
            outer => TRUE
        ) med
    WHERE med.value IS NOT NULL
),

-- Flatten nested vital signs recorded in notes
vitals_from_notes AS (
    SELECT
        nf.patient_id,
        nf.encounter_id,
        nf.note_date,
        v.value:name::STRING                 AS vital_name,
        v.value:value::FLOAT                 AS vital_value,
        v.value:unit::STRING                 AS vital_unit,
        v.value:abnormal::BOOLEAN            AS vital_abnormal
    FROM note_flattened nf,
        LATERAL FLATTEN(
            input => PARSE_JSON(nf.sections_variant):vitals,
            outer => TRUE
        ) v
    WHERE v.value IS NOT NULL
),

-- ---------------------------------------------------------------------------
-- 2. Medical record entity extraction from VARIANT
-- ---------------------------------------------------------------------------
record_latest AS (
    SELECT
        mr.patient_id,
        mr.record_id,
        mr.medical_record,
        mr.record_date
    FROM medical_records mr
    WHERE mr.record_date <= %(as_of_date)s::DATE
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY mr.patient_id ORDER BY mr.record_date DESC
    ) = 1
),

-- Extract nested allergy array
allergies_flat AS (
    SELECT
        rl.patient_id,
        a.value:substance::STRING            AS substance,
        a.value:reaction::STRING             AS reaction,
        a.value:severity::STRING             AS severity,
        a.value:category::STRING             AS category,
        a.value:onset_date::DATE             AS onset_date,
        a.value:verified::BOOLEAN            AS verified
    FROM record_latest rl,
        LATERAL FLATTEN(input => rl.medical_record:allergies, outer => TRUE) a
),

-- Extract nested immunization array
immunizations_flat AS (
    SELECT
        rl.patient_id,
        imm.value:vaccine_code::STRING       AS vaccine_code,
        imm.value:vaccine_name::STRING       AS vaccine_name,
        imm.value:administration_date::DATE  AS admin_date,
        imm.value:dose_number::INTEGER       AS dose_number,
        imm.value:manufacturer::STRING       AS manufacturer,
        imm.value:lot_number::STRING         AS lot_number
    FROM record_latest rl,
        LATERAL FLATTEN(input => rl.medical_record:immunizations, outer => TRUE) imm
),

-- Extract social determinants of health from nested JSON
social_determinants AS (
    SELECT
        rl.patient_id,
        rl.medical_record:social_history.smoking_status::STRING       AS smoking_status,
        rl.medical_record:social_history.alcohol_use::STRING          AS alcohol_use,
        rl.medical_record:social_history.substance_use::STRING        AS substance_use,
        rl.medical_record:social_history.exercise_frequency::STRING   AS exercise_freq,
        rl.medical_record:social_history.housing_status::STRING       AS housing_status,
        rl.medical_record:social_history.employment_status::STRING    AS employment_status,
        rl.medical_record:social_history.food_insecurity::BOOLEAN     AS food_insecurity,
        rl.medical_record:social_history.transportation_barrier::BOOLEAN AS transportation_barrier
    FROM record_latest rl
),

-- Extract deeply nested family history with recursive conditions
family_history_flat AS (
    SELECT
        rl.patient_id,
        fh.value:relationship::STRING        AS relationship,
        fh.value:condition::STRING           AS fh_condition,
        fh.value:age_at_onset::INTEGER       AS age_at_onset,
        fh.value:is_deceased::BOOLEAN        AS is_deceased,
        fh.value:cause_of_death::STRING      AS cause_of_death,
        -- Nested conditions within family member
        fc.value:icd_code::STRING            AS fh_icd_code
    FROM record_latest rl,
        LATERAL FLATTEN(input => rl.medical_record:family_history, outer => TRUE) fh,
        LATERAL FLATTEN(input => fh.value:conditions, outer => TRUE) fc
),

-- ---------------------------------------------------------------------------
-- 3. Nested lab panel extraction
-- ---------------------------------------------------------------------------
lab_panel_components AS (
    SELECT
        lp.patient_id,
        lp.order_id,
        lp.collection_date,
        lp.lab_panel_data:panel_name::STRING                    AS panel_name,
        lp.lab_panel_data:panel_code::STRING                    AS panel_code,
        lp.lab_panel_data:ordering_provider::STRING             AS ordering_provider,
        lp.lab_panel_data:specimen_type::STRING                 AS specimen_type,
        comp.value:component_name::STRING                       AS component_name,
        comp.value:component_code::STRING                       AS component_code,
        comp.value:result_value::FLOAT                          AS result_value,
        comp.value:unit::STRING                                 AS unit,
        comp.value:reference_range.low::FLOAT                   AS ref_low,
        comp.value:reference_range.high::FLOAT                  AS ref_high,
        comp.value:abnormal_flag::STRING                        AS abnormal_flag,
        -- Nested qualitative results
        comp.value:qualitative_result::STRING                   AS qual_result,
        comp.value:interpretation.narrative::STRING              AS interpretation,
        CASE
            WHEN comp.value:result_value::FLOAT < comp.value:reference_range.low::FLOAT
                THEN -1
            WHEN comp.value:result_value::FLOAT > comp.value:reference_range.high::FLOAT
                THEN 1
            ELSE 0
        END AS deviation_direction
    FROM lab_panels lp,
        LATERAL FLATTEN(input => lp.lab_panel_data:components) comp
    WHERE lp.collection_date >= DATEADD('day', -%(lookback_days)s, %(as_of_date)s::DATE)
      AND lp.collection_date <= %(as_of_date)s::DATE
),

-- ---------------------------------------------------------------------------
-- 4. Aggregate features
-- ---------------------------------------------------------------------------
patient_note_features AS (
    SELECT
        nf.patient_id,
        COUNT(DISTINCT nf.encounter_id)      AS note_encounter_count,
        COUNT(*)                              AS total_note_count,
        AVG(nf.note_length)                  AS avg_note_length,
        COUNT(DISTINCT nf.provider_specialty) AS distinct_specialties,
        SUM(CASE WHEN nf.priority = 'URGENT' THEN 1 ELSE 0 END)  AS urgent_notes,
        SUM(CASE WHEN nf.is_addendum THEN 1 ELSE 0 END)          AS addendum_count
    FROM note_flattened nf
    GROUP BY nf.patient_id
),

patient_dx_features AS (
    SELECT
        patient_id,
        COUNT(DISTINCT dx_code)              AS unique_dx_from_notes,
        COUNT(DISTINCT SUBSTR(dx_code, 1, 3)) AS unique_dx_categories,
        SUM(CASE WHEN is_primary THEN 1 ELSE 0 END) AS primary_dx_count
    FROM diagnoses_unnested
    GROUP BY patient_id
),

patient_allergy_features AS (
    SELECT
        patient_id,
        COUNT(*) AS total_allergies,
        SUM(CASE WHEN severity = 'SEVERE' THEN 1 ELSE 0 END) AS severe_allergies,
        SUM(CASE WHEN category = 'DRUG' THEN 1 ELSE 0 END)   AS drug_allergies,
        SUM(CASE WHEN category = 'FOOD' THEN 1 ELSE 0 END)   AS food_allergies,
        SUM(CASE WHEN verified THEN 1 ELSE 0 END)             AS verified_allergies
    FROM allergies_flat
    WHERE substance IS NOT NULL
    GROUP BY patient_id
),

patient_lab_panel_features AS (
    SELECT
        patient_id,
        COUNT(DISTINCT panel_name) AS distinct_panels_ordered,
        COUNT(DISTINCT component_name) AS distinct_components,
        SUM(CASE WHEN deviation_direction != 0 THEN 1 ELSE 0 END) AS abnormal_components,
        AVG(ABS(deviation_direction)) AS avg_deviation,
        COUNT(DISTINCT order_id) AS total_panel_orders
    FROM lab_panel_components
    GROUP BY patient_id
)

-- ---------------------------------------------------------------------------
-- Final assembly
-- ---------------------------------------------------------------------------
SELECT
    COALESCE(pnf.patient_id, pdf.patient_id, paf.patient_id, plf.patient_id) AS patient_id,

    -- Note features
    COALESCE(pnf.note_encounter_count, 0)      AS note_encounter_count,
    COALESCE(pnf.total_note_count, 0)           AS total_note_count,
    COALESCE(pnf.avg_note_length, 0)            AS avg_note_length,
    COALESCE(pnf.distinct_specialties, 0)       AS distinct_specialties,
    COALESCE(pnf.urgent_notes, 0)               AS urgent_notes,
    COALESCE(pnf.addendum_count, 0)             AS addendum_count,

    -- Diagnosis features from notes
    COALESCE(pdf.unique_dx_from_notes, 0)       AS unique_dx_from_notes,
    COALESCE(pdf.unique_dx_categories, 0)       AS unique_dx_categories,
    COALESCE(pdf.primary_dx_count, 0)           AS primary_dx_count,

    -- Allergy features
    COALESCE(paf.total_allergies, 0)            AS total_allergies,
    COALESCE(paf.severe_allergies, 0)           AS severe_allergies,
    COALESCE(paf.drug_allergies, 0)             AS drug_allergies,
    COALESCE(paf.food_allergies, 0)             AS food_allergies,
    COALESCE(paf.verified_allergies, 0)         AS verified_allergies,

    -- Lab panel features from VARIANT
    COALESCE(plf.distinct_panels_ordered, 0)    AS distinct_panels_ordered,
    COALESCE(plf.distinct_components, 0)        AS distinct_components,
    COALESCE(plf.abnormal_components, 0)        AS abnormal_components,
    COALESCE(plf.total_panel_orders, 0)         AS total_panel_orders,

    -- Social determinants
    sd.smoking_status,
    sd.alcohol_use,
    sd.exercise_freq,
    sd.housing_status,
    COALESCE(sd.food_insecurity, FALSE)         AS food_insecurity,
    COALESCE(sd.transportation_barrier, FALSE)  AS transportation_barrier,

    CURRENT_TIMESTAMP() AS feature_ts

FROM patient_note_features pnf
FULL OUTER JOIN patient_dx_features pdf ON pnf.patient_id = pdf.patient_id
FULL OUTER JOIN patient_allergy_features paf
    ON COALESCE(pnf.patient_id, pdf.patient_id) = paf.patient_id
FULL OUTER JOIN patient_lab_panel_features plf
    ON COALESCE(pnf.patient_id, pdf.patient_id, paf.patient_id) = plf.patient_id
LEFT JOIN social_determinants sd
    ON COALESCE(pnf.patient_id, pdf.patient_id, paf.patient_id, plf.patient_id) = sd.patient_id
ORDER BY 1;
