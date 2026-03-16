-- =============================================================================
-- Lab result features
-- Computes rolling statistics (mean, std, min, max, trend) for each lab test
-- per patient using Snowflake window functions. Includes out-of-range counts,
-- rate-of-change indicators, and critical value flags.
-- =============================================================================

WITH lab_base AS (
    SELECT
        lr.patient_id,
        lr.lab_test_code,
        lr.lab_test_name,
        lr.result_value,
        lr.unit,
        lr.reference_low,
        lr.reference_high,
        lr.collection_date,
        lr.abnormal_flag,
        -- Classify result relative to reference range
        CASE
            WHEN lr.result_value < lr.reference_low THEN 'LOW'
            WHEN lr.result_value > lr.reference_high THEN 'HIGH'
            ELSE 'NORMAL'
        END AS range_status,
        -- Critical value flags
        CASE
            WHEN lr.lab_test_code = 'POTASSIUM'
                 AND (lr.result_value < 2.5 OR lr.result_value > 6.5) THEN 1
            WHEN lr.lab_test_code = 'SODIUM'
                 AND (lr.result_value < 120 OR lr.result_value > 160) THEN 1
            WHEN lr.lab_test_code = 'GLUCOSE'
                 AND (lr.result_value < 40 OR lr.result_value > 500) THEN 1
            WHEN lr.lab_test_code = 'HEMOGLOBIN'
                 AND lr.result_value < 7.0 THEN 1
            WHEN lr.lab_test_code = 'PLATELET'
                 AND lr.result_value < 50 THEN 1
            ELSE 0
        END AS is_critical_value,
        -- Row numbering for trend computation
        ROW_NUMBER() OVER (
            PARTITION BY lr.patient_id, lr.lab_test_code
            ORDER BY lr.collection_date
        ) AS seq_num,
        COUNT(*) OVER (
            PARTITION BY lr.patient_id, lr.lab_test_code
        ) AS total_measurements
    FROM lab_results lr
    WHERE lr.collection_date >= DATEADD('day', -%(lookback_days)s, %(as_of_date)s::DATE)
      AND lr.collection_date <= %(as_of_date)s::DATE
      AND lr.result_value IS NOT NULL
      AND lr.status = 'FINAL'
),

-- Rolling window statistics per patient per lab test
lab_rolling AS (
    SELECT
        lb.patient_id,
        lb.lab_test_code,
        AVG(lb.result_value) AS mean_value,
        STDDEV(lb.result_value) AS std_value,
        MIN(lb.result_value) AS min_value,
        MAX(lb.result_value) AS max_value,
        MEDIAN(lb.result_value) AS median_value,
        COUNT(*) AS measurement_count,
        SUM(CASE WHEN lb.range_status != 'NORMAL' THEN 1 ELSE 0 END) AS out_of_range_count,
        SUM(lb.is_critical_value) AS critical_value_count,
        -- Coefficient of variation
        CASE
            WHEN AVG(lb.result_value) != 0
            THEN STDDEV(lb.result_value) / ABS(AVG(lb.result_value))
            ELSE NULL
        END AS coefficient_of_variation,
        -- Days between first and last measurement
        DATEDIFF('day', MIN(lb.collection_date), MAX(lb.collection_date)) AS measurement_span_days,
        -- Most recent value
        MAX(CASE WHEN lb.seq_num = lb.total_measurements THEN lb.result_value END)
            AS most_recent_value,
        -- Earliest value in window
        MAX(CASE WHEN lb.seq_num = 1 THEN lb.result_value END)
            AS earliest_value,
        -- Days since most recent measurement
        DATEDIFF('day', MAX(lb.collection_date), %(as_of_date)s::DATE)
            AS days_since_last_measurement
    FROM lab_base lb
    GROUP BY lb.patient_id, lb.lab_test_code
),

-- Compute trend slope using linear regression approximation
-- slope = (n * SUM(xy) - SUM(x)*SUM(y)) / (n * SUM(x^2) - (SUM(x))^2)
lab_trend AS (
    SELECT
        lb.patient_id,
        lb.lab_test_code,
        CASE
            WHEN COUNT(*) >= 3
                 AND (COUNT(*) * SUM(POWER(lb.seq_num, 2)) - POWER(SUM(lb.seq_num), 2)) != 0
            THEN (
                COUNT(*) * SUM(lb.seq_num * lb.result_value)
                - SUM(lb.seq_num) * SUM(lb.result_value)
            ) / (
                COUNT(*) * SUM(POWER(lb.seq_num, 2)) - POWER(SUM(lb.seq_num), 2)
            )
            ELSE 0
        END AS trend_slope
    FROM lab_base lb
    GROUP BY lb.patient_id, lb.lab_test_code
),

-- Pivot into wide format: one row per patient
pivoted AS (
    SELECT
        lr.patient_id,

        -- Hemoglobin
        MAX(CASE WHEN lr.lab_test_code = 'HEMOGLOBIN' THEN lr.mean_value END) AS hemoglobin_mean,
        MAX(CASE WHEN lr.lab_test_code = 'HEMOGLOBIN' THEN lr.std_value END) AS hemoglobin_std,
        MAX(CASE WHEN lr.lab_test_code = 'HEMOGLOBIN' THEN lr.most_recent_value END) AS hemoglobin_latest,
        MAX(CASE WHEN lr.lab_test_code = 'HEMOGLOBIN' THEN lt.trend_slope END) AS hemoglobin_trend,
        MAX(CASE WHEN lr.lab_test_code = 'HEMOGLOBIN' THEN lr.out_of_range_count END) AS hemoglobin_oor_count,

        -- Hematocrit
        MAX(CASE WHEN lr.lab_test_code = 'HEMATOCRIT' THEN lr.mean_value END) AS hematocrit_mean,
        MAX(CASE WHEN lr.lab_test_code = 'HEMATOCRIT' THEN lr.std_value END) AS hematocrit_std,

        -- Creatinine
        MAX(CASE WHEN lr.lab_test_code = 'CREATININE' THEN lr.mean_value END) AS creatinine_mean,
        MAX(CASE WHEN lr.lab_test_code = 'CREATININE' THEN lr.std_value END) AS creatinine_std,
        MAX(CASE WHEN lr.lab_test_code = 'CREATININE' THEN lr.most_recent_value END) AS creatinine_latest,
        MAX(CASE WHEN lr.lab_test_code = 'CREATININE' THEN lt.trend_slope END) AS creatinine_trend,
        MAX(CASE WHEN lr.lab_test_code = 'CREATININE' THEN lr.coefficient_of_variation END) AS creatinine_cv,

        -- BUN
        MAX(CASE WHEN lr.lab_test_code = 'BUN' THEN lr.mean_value END) AS bun_mean,
        MAX(CASE WHEN lr.lab_test_code = 'BUN' THEN lr.most_recent_value END) AS bun_latest,

        -- Glucose
        MAX(CASE WHEN lr.lab_test_code = 'GLUCOSE' THEN lr.mean_value END) AS glucose_mean,
        MAX(CASE WHEN lr.lab_test_code = 'GLUCOSE' THEN lr.std_value END) AS glucose_std,
        MAX(CASE WHEN lr.lab_test_code = 'GLUCOSE' THEN lr.max_value END) AS glucose_max,
        MAX(CASE WHEN lr.lab_test_code = 'GLUCOSE' THEN lr.out_of_range_count END) AS glucose_oor_count,
        MAX(CASE WHEN lr.lab_test_code = 'GLUCOSE' THEN lt.trend_slope END) AS glucose_trend,

        -- HbA1c
        MAX(CASE WHEN lr.lab_test_code = 'HBA1C' THEN lr.most_recent_value END) AS hba1c_latest,
        MAX(CASE WHEN lr.lab_test_code = 'HBA1C' THEN lt.trend_slope END) AS hba1c_trend,

        -- Electrolytes
        MAX(CASE WHEN lr.lab_test_code = 'SODIUM' THEN lr.mean_value END) AS sodium_mean,
        MAX(CASE WHEN lr.lab_test_code = 'POTASSIUM' THEN lr.mean_value END) AS potassium_mean,
        MAX(CASE WHEN lr.lab_test_code = 'CALCIUM' THEN lr.mean_value END) AS calcium_mean,
        MAX(CASE WHEN lr.lab_test_code = 'CHLORIDE' THEN lr.mean_value END) AS chloride_mean,

        -- WBC
        MAX(CASE WHEN lr.lab_test_code = 'WBC' THEN lr.mean_value END) AS wbc_mean,
        MAX(CASE WHEN lr.lab_test_code = 'WBC' THEN lr.std_value END) AS wbc_std,
        MAX(CASE WHEN lr.lab_test_code = 'WBC' THEN lr.out_of_range_count END) AS wbc_oor_count,

        -- Platelet
        MAX(CASE WHEN lr.lab_test_code = 'PLATELET' THEN lr.mean_value END) AS platelet_mean,
        MAX(CASE WHEN lr.lab_test_code = 'PLATELET' THEN lr.most_recent_value END) AS platelet_latest,

        -- Albumin
        MAX(CASE WHEN lr.lab_test_code = 'ALBUMIN' THEN lr.most_recent_value END) AS albumin_latest,
        MAX(CASE WHEN lr.lab_test_code = 'ALBUMIN' THEN lt.trend_slope END) AS albumin_trend,

        -- Aggregate across all lab tests
        SUM(lr.measurement_count) AS total_lab_count,
        SUM(lr.out_of_range_count) AS total_oor_count,
        SUM(lr.critical_value_count) AS total_critical_count,
        COUNT(DISTINCT lr.lab_test_code) AS distinct_lab_types,
        MIN(lr.days_since_last_measurement) AS days_since_any_lab

    FROM lab_rolling lr
    LEFT JOIN lab_trend lt
        ON lr.patient_id = lt.patient_id
        AND lr.lab_test_code = lt.lab_test_code
    GROUP BY lr.patient_id
)

SELECT
    p.*,
    -- Derived ratio features
    CASE
        WHEN p.hematocrit_mean IS NOT NULL AND p.hematocrit_mean > 0
        THEN p.hemoglobin_mean / p.hematocrit_mean
        ELSE NULL
    END AS hgb_hct_ratio,
    CASE
        WHEN p.creatinine_mean IS NOT NULL AND p.creatinine_mean > 0
        THEN p.bun_mean / p.creatinine_mean
        ELSE NULL
    END AS bun_creatinine_ratio,
    -- Out-of-range rate
    CASE
        WHEN p.total_lab_count > 0
        THEN p.total_oor_count::FLOAT / p.total_lab_count
        ELSE 0
    END AS overall_oor_rate,
    CURRENT_TIMESTAMP() AS feature_ts
FROM pivoted p
ORDER BY p.patient_id;
