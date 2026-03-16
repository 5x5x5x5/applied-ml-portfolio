/*******************************************************************************
 * PharmaDataVault - Clinical Analytics Queries
 *
 * Complex analytical queries for pharmaceutical data analysis:
 *   1. Drug safety signal detection (disproportionality analysis)
 *   2. Trial enrollment trends with window functions
 *   3. Adverse event severity by drug class (pivot)
 *   4. Patient cohort analysis
 *   5. Manufacturing quality metrics
 *
 * Oracle SQL compatible with analytical functions
 ******************************************************************************/

-- ============================================================================
-- 1. DRUG SAFETY SIGNAL DETECTION
-- Proportional Reporting Ratio (PRR) for disproportionality analysis.
-- Identifies drug-AE combinations that occur more frequently than expected.
-- PRR > 2 with chi-square > 4 and N >= 3 is a signal (Evans criteria).
-- ============================================================================

WITH ae_drug_counts AS (
    -- Count of each drug-AE combination
    SELECT
        dd.DRUG_NAME,
        dd.THERAPEUTIC_CLASS,
        fa.AE_TERM,
        fa.MEDDRA_SOC,
        COUNT(*)                                        AS A,   -- Drug + AE
        SUM(COUNT(*)) OVER (PARTITION BY dd.DRUG_NAME)  AS drug_total -- Drug total AEs
    FROM FACT_ADVERSE_EVENTS fa
    INNER JOIN DIM_DRUG dd ON fa.DIM_DRUG_KEY = dd.DIM_DRUG_KEY AND dd.IS_CURRENT = 'Y'
    GROUP BY dd.DRUG_NAME, dd.THERAPEUTIC_CLASS, fa.AE_TERM, fa.MEDDRA_SOC
),
ae_background AS (
    -- Background rate for each AE across all drugs
    SELECT
        fa.AE_TERM,
        COUNT(*)                                        AS C,   -- All drugs + AE
        SUM(COUNT(*)) OVER ()                           AS total_all -- Grand total AEs
    FROM FACT_ADVERSE_EVENTS fa
    GROUP BY fa.AE_TERM
),
prr_calc AS (
    SELECT
        adc.DRUG_NAME,
        adc.THERAPEUTIC_CLASS,
        adc.AE_TERM,
        adc.MEDDRA_SOC,
        adc.A,
        adc.drug_total                                  AS B_plus_A,
        ab.C,
        ab.total_all                                    AS D_plus_C,
        -- PRR = (A / (A+B)) / (C / (C+D))
        --     = (A / drug_total) / (C / total_all)
        ROUND(
            (adc.A / NULLIF(adc.drug_total, 0))
            / NULLIF((ab.C / NULLIF(ab.total_all, 0)), 0),
            3
        )                                               AS PRR,
        -- Chi-square approximation for 2x2 table
        ROUND(
            POWER(
                ABS(adc.A * (ab.total_all - ab.C) - (adc.drug_total - adc.A) * ab.C)
                - (ab.total_all / 2),
                2
            ) * ab.total_all
            / NULLIF(
                adc.drug_total * (ab.total_all - adc.drug_total) * ab.C * (ab.total_all - ab.C),
                0
            ),
            3
        )                                               AS CHI_SQUARE
    FROM ae_drug_counts adc
    INNER JOIN ae_background ab ON adc.AE_TERM = ab.AE_TERM
)
SELECT
    DRUG_NAME,
    THERAPEUTIC_CLASS,
    AE_TERM,
    MEDDRA_SOC,
    A                       AS EVENT_COUNT,
    PRR,
    CHI_SQUARE,
    CASE
        WHEN PRR >= 2 AND CHI_SQUARE >= 4 AND A >= 3 THEN 'SIGNAL DETECTED'
        WHEN PRR >= 1.5 AND A >= 3 THEN 'MONITOR'
        ELSE 'NO SIGNAL'
    END                     AS SIGNAL_STATUS,
    RANK() OVER (
        PARTITION BY DRUG_NAME
        ORDER BY PRR DESC, A DESC
    )                       AS SIGNAL_RANK
FROM prr_calc
WHERE A >= 3  -- Minimum case threshold
ORDER BY PRR DESC, A DESC;


-- ============================================================================
-- 2. TRIAL ENROLLMENT TRENDS WITH WINDOW FUNCTIONS
-- Running enrollment counts, month-over-month growth, and enrollment
-- velocity relative to target.
-- ============================================================================

WITH monthly_enrollment AS (
    SELECT
        fe.TRIAL_NCT_ID,
        fe.TRIAL_PHASE,
        dt.YEAR_NUMBER,
        dt.MONTH_NUMBER,
        dt.MONTH_NAME,
        dd.DRUG_NAME,
        COUNT(*)                                        AS MONTHLY_COUNT,
        COUNT(DISTINCT fe.DIM_PATIENT_KEY)              AS UNIQUE_PATIENTS,
        SUM(CASE WHEN fe.SCREEN_FAILURE_FLAG = 'Y'
            THEN 1 ELSE 0 END)                          AS SCREEN_FAILURES,
        SUM(CASE WHEN fe.WITHDRAWN_FLAG = 'Y'
            THEN 1 ELSE 0 END)                          AS WITHDRAWALS
    FROM FACT_TRIAL_ENROLLMENT fe
    INNER JOIN DIM_TIME dt ON fe.ENROLL_DATE_KEY = dt.TIME_KEY
    INNER JOIN DIM_DRUG dd ON fe.DIM_DRUG_KEY = dd.DIM_DRUG_KEY AND dd.IS_CURRENT = 'Y'
    GROUP BY
        fe.TRIAL_NCT_ID, fe.TRIAL_PHASE,
        dt.YEAR_NUMBER, dt.MONTH_NUMBER, dt.MONTH_NAME,
        dd.DRUG_NAME
)
SELECT
    TRIAL_NCT_ID,
    TRIAL_PHASE,
    DRUG_NAME,
    YEAR_NUMBER,
    MONTH_NAME,
    MONTHLY_COUNT,
    UNIQUE_PATIENTS,
    -- Running total of enrollments
    SUM(MONTHLY_COUNT) OVER (
        PARTITION BY TRIAL_NCT_ID
        ORDER BY YEAR_NUMBER, MONTH_NUMBER
        ROWS UNBOUNDED PRECEDING
    )                                                   AS CUMULATIVE_ENROLLED,
    -- Month-over-month change
    MONTHLY_COUNT - LAG(MONTHLY_COUNT, 1, 0) OVER (
        PARTITION BY TRIAL_NCT_ID
        ORDER BY YEAR_NUMBER, MONTH_NUMBER
    )                                                   AS MOM_CHANGE,
    -- 3-month rolling average
    ROUND(AVG(MONTHLY_COUNT) OVER (
        PARTITION BY TRIAL_NCT_ID
        ORDER BY YEAR_NUMBER, MONTH_NUMBER
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 1)                                               AS ROLLING_3M_AVG,
    -- Screen failure rate
    ROUND(
        SCREEN_FAILURES / NULLIF(MONTHLY_COUNT, 0) * 100,
        1
    )                                                   AS SCREEN_FAILURE_PCT,
    -- Withdrawal rate
    ROUND(
        WITHDRAWALS / NULLIF(MONTHLY_COUNT, 0) * 100,
        1
    )                                                   AS WITHDRAWAL_PCT,
    -- Rank months by enrollment within each trial
    RANK() OVER (
        PARTITION BY TRIAL_NCT_ID
        ORDER BY MONTHLY_COUNT DESC
    )                                                   AS ENROLLMENT_RANK
FROM monthly_enrollment
ORDER BY TRIAL_NCT_ID, YEAR_NUMBER, MONTH_NUMBER;


-- ============================================================================
-- 3. ADVERSE EVENT SEVERITY BY DRUG CLASS (PIVOT QUERY)
-- Cross-tabulation of severity counts by therapeutic class.
-- ============================================================================

SELECT *
FROM (
    SELECT
        dd.THERAPEUTIC_CLASS,
        fa.SEVERITY,
        fa.AE_FACT_KEY
    FROM FACT_ADVERSE_EVENTS fa
    INNER JOIN DIM_DRUG dd ON fa.DIM_DRUG_KEY = dd.DIM_DRUG_KEY AND dd.IS_CURRENT = 'Y'
    WHERE dd.THERAPEUTIC_CLASS IS NOT NULL
)
PIVOT (
    COUNT(AE_FACT_KEY)
    FOR SEVERITY IN (
        'mild'              AS MILD,
        'moderate'          AS MODERATE,
        'severe'            AS SEVERE,
        'life_threatening'  AS LIFE_THREATENING,
        'fatal'             AS FATAL
    )
)
ORDER BY THERAPEUTIC_CLASS;

-- Supplementary: percentage distribution per class
WITH severity_counts AS (
    SELECT
        dd.THERAPEUTIC_CLASS,
        fa.SEVERITY,
        COUNT(*)            AS AE_COUNT,
        SUM(COUNT(*)) OVER (PARTITION BY dd.THERAPEUTIC_CLASS) AS CLASS_TOTAL
    FROM FACT_ADVERSE_EVENTS fa
    INNER JOIN DIM_DRUG dd ON fa.DIM_DRUG_KEY = dd.DIM_DRUG_KEY AND dd.IS_CURRENT = 'Y'
    WHERE dd.THERAPEUTIC_CLASS IS NOT NULL
    GROUP BY dd.THERAPEUTIC_CLASS, fa.SEVERITY
)
SELECT
    THERAPEUTIC_CLASS,
    SEVERITY,
    AE_COUNT,
    CLASS_TOTAL,
    ROUND(AE_COUNT / NULLIF(CLASS_TOTAL, 0) * 100, 1) AS PCT_OF_CLASS,
    -- Sparkline-style bar (text-based visualization)
    RPAD('|', ROUND(AE_COUNT / NULLIF(CLASS_TOTAL, 0) * 50), '|') AS BAR
FROM severity_counts
ORDER BY THERAPEUTIC_CLASS,
    DECODE(SEVERITY, 'mild', 1, 'moderate', 2, 'severe', 3, 'life_threatening', 4, 'fatal', 5);


-- ============================================================================
-- 4. PATIENT COHORT ANALYSIS
-- Multi-dimensional patient cohort comparison: demographics, outcomes,
-- adverse event profiles across treatment arms.
-- ============================================================================

WITH patient_ae_profile AS (
    -- Aggregate AE profile per patient
    SELECT
        dp.DIM_PATIENT_KEY,
        dp.PATIENT_MRN,
        dp.AGE_GROUP,
        dp.SEX,
        dp.BMI_CATEGORY,
        COUNT(DISTINCT fa.AE_FACT_KEY)                  AS TOTAL_AES,
        MAX(fa.SEVERITY_SCORE)                          AS MAX_SEVERITY,
        AVG(fa.SEVERITY_SCORE)                          AS AVG_SEVERITY,
        SUM(CASE WHEN fa.SERIOUSNESS = 'Y' THEN 1 ELSE 0 END) AS SERIOUS_AE_COUNT,
        COUNT(DISTINCT fa.AE_TERM)                      AS DISTINCT_AE_TERMS,
        AVG(fa.DURATION_DAYS)                           AS AVG_AE_DURATION
    FROM DIM_PATIENT dp
    LEFT JOIN FACT_ADVERSE_EVENTS fa ON dp.DIM_PATIENT_KEY = fa.DIM_PATIENT_KEY
    WHERE dp.IS_CURRENT = 'Y'
    GROUP BY dp.DIM_PATIENT_KEY, dp.PATIENT_MRN, dp.AGE_GROUP, dp.SEX, dp.BMI_CATEGORY
),
patient_trial_profile AS (
    -- Trial participation profile per patient
    SELECT
        fe.DIM_PATIENT_KEY,
        COUNT(DISTINCT fe.TRIAL_NCT_ID)                 AS TRIALS_PARTICIPATED,
        AVG(fe.ENROLLMENT_DURATION_DAYS)                 AS AVG_ENROLLMENT_DAYS,
        SUM(fe.PROTOCOL_DEVIATIONS)                      AS TOTAL_DEVIATIONS,
        MAX(CASE WHEN fe.COMPLETED_FLAG = 'Y' THEN 1 ELSE 0 END) AS ANY_COMPLETED,
        MAX(CASE WHEN fe.WITHDRAWN_FLAG = 'Y' THEN 1 ELSE 0 END) AS ANY_WITHDRAWN
    FROM FACT_TRIAL_ENROLLMENT fe
    GROUP BY fe.DIM_PATIENT_KEY
)
SELECT
    pae.AGE_GROUP,
    pae.SEX,
    pae.BMI_CATEGORY,
    COUNT(*)                                            AS COHORT_SIZE,
    -- AE metrics per cohort
    ROUND(AVG(pae.TOTAL_AES), 1)                        AS AVG_AES_PER_PATIENT,
    ROUND(AVG(pae.AVG_SEVERITY), 2)                     AS AVG_SEVERITY_SCORE,
    ROUND(AVG(pae.SERIOUS_AE_COUNT), 2)                 AS AVG_SERIOUS_AES,
    ROUND(AVG(pae.AVG_AE_DURATION), 1)                  AS AVG_AE_DURATION_DAYS,
    -- Trial metrics per cohort
    ROUND(AVG(ptp.TRIALS_PARTICIPATED), 1)              AS AVG_TRIALS,
    ROUND(AVG(ptp.AVG_ENROLLMENT_DAYS), 0)              AS AVG_ENROLLMENT_DAYS,
    ROUND(
        SUM(ptp.ANY_COMPLETED) / NULLIF(COUNT(*), 0) * 100,
        1
    )                                                   AS COMPLETION_RATE_PCT,
    ROUND(
        SUM(ptp.ANY_WITHDRAWN) / NULLIF(COUNT(*), 0) * 100,
        1
    )                                                   AS WITHDRAWAL_RATE_PCT,
    -- Percentile ranks within the full population
    ROUND(
        PERCENT_RANK() OVER (ORDER BY AVG(pae.TOTAL_AES)) * 100,
        1
    )                                                   AS AE_BURDEN_PERCENTILE
FROM patient_ae_profile pae
LEFT JOIN patient_trial_profile ptp ON pae.DIM_PATIENT_KEY = ptp.DIM_PATIENT_KEY
GROUP BY pae.AGE_GROUP, pae.SEX, pae.BMI_CATEGORY
HAVING COUNT(*) >= 5  -- Minimum cohort size for statistical relevance
ORDER BY AVG(pae.TOTAL_AES) DESC;


-- ============================================================================
-- 5. MANUFACTURING QUALITY METRICS
-- Control chart data for manufacturing yield and QC failure analysis.
-- ============================================================================

WITH lot_data AS (
    SELECT
        dd.DRUG_NAME,
        dd.MANUFACTURER,
        sm.FACILITY_ID,
        sm.LOT_NUMBER,
        sm.MFG_DATE,
        sm.QC_STATUS,
        sm.YIELD_PERCENT,
        sm.DEVIATION_FLAG,
        sm.BATCH_SIZE,
        -- Moving averages and control limits
        AVG(sm.YIELD_PERCENT) OVER (
            PARTITION BY sm.DRUG_KEY, sm.FACILITY_ID
            ORDER BY sm.MFG_DATE
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        )                                               AS MOVING_AVG_YIELD,
        STDDEV(sm.YIELD_PERCENT) OVER (
            PARTITION BY sm.DRUG_KEY, sm.FACILITY_ID
            ORDER BY sm.MFG_DATE
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        )                                               AS YIELD_STDDEV,
        AVG(sm.YIELD_PERCENT) OVER (
            PARTITION BY sm.DRUG_KEY, sm.FACILITY_ID
        )                                               AS OVERALL_AVG_YIELD,
        -- Days since last QC failure
        sm.MFG_DATE - LAG(
            CASE WHEN sm.QC_STATUS = 'failed' THEN sm.MFG_DATE END
        ) IGNORE NULLS OVER (
            PARTITION BY sm.DRUG_KEY, sm.FACILITY_ID
            ORDER BY sm.MFG_DATE
        )                                               AS DAYS_SINCE_LAST_FAILURE
    FROM SAT_DRUG_MANUFACTURING sm
    INNER JOIN HUB_DRUG hd ON sm.DRUG_KEY = hd.DRUG_KEY
    INNER JOIN DIM_DRUG dd ON hd.DRUG_KEY = dd.DRUG_VAULT_KEY AND dd.IS_CURRENT = 'Y'
    WHERE sm.LOAD_END_DATE = TO_TIMESTAMP('9999-12-31 23:59:59', 'YYYY-MM-DD HH24:MI:SS')
)
SELECT
    DRUG_NAME,
    MANUFACTURER,
    FACILITY_ID,
    LOT_NUMBER,
    MFG_DATE,
    QC_STATUS,
    YIELD_PERCENT,
    MOVING_AVG_YIELD,
    -- Upper Control Limit (UCL) = mean + 3*sigma
    ROUND(OVERALL_AVG_YIELD + 3 * NVL(YIELD_STDDEV, 0), 2)  AS UCL,
    -- Lower Control Limit (LCL) = mean - 3*sigma
    ROUND(GREATEST(OVERALL_AVG_YIELD - 3 * NVL(YIELD_STDDEV, 0), 0), 2) AS LCL,
    -- Out of control flag
    CASE
        WHEN YIELD_PERCENT > OVERALL_AVG_YIELD + 3 * NVL(YIELD_STDDEV, 0) THEN 'OOC_HIGH'
        WHEN YIELD_PERCENT < OVERALL_AVG_YIELD - 3 * NVL(YIELD_STDDEV, 0) THEN 'OOC_LOW'
        ELSE 'IN_CONTROL'
    END                                                 AS CONTROL_STATUS,
    DEVIATION_FLAG,
    DAYS_SINCE_LAST_FAILURE,
    -- Cumulative failure count (useful for trend detection)
    SUM(CASE WHEN QC_STATUS = 'failed' THEN 1 ELSE 0 END) OVER (
        PARTITION BY DRUG_NAME, FACILITY_ID
        ORDER BY MFG_DATE
        ROWS UNBOUNDED PRECEDING
    )                                                   AS CUMULATIVE_FAILURES
FROM lot_data
ORDER BY DRUG_NAME, FACILITY_ID, MFG_DATE;

-- Summary: Facility OEE (Overall Equipment Effectiveness) proxy
SELECT
    dd.MANUFACTURER,
    sm.FACILITY_ID,
    TO_CHAR(sm.MFG_DATE, 'YYYY-Q')                     AS MFG_QUARTER,
    COUNT(DISTINCT sm.LOT_NUMBER)                       AS TOTAL_LOTS,
    ROUND(AVG(sm.YIELD_PERCENT), 2)                     AS AVG_YIELD,
    ROUND(
        SUM(CASE WHEN sm.QC_STATUS = 'passed' THEN 1 ELSE 0 END)
        / NULLIF(COUNT(*), 0) * 100,
        1
    )                                                   AS FIRST_PASS_YIELD_PCT,
    ROUND(
        SUM(CASE WHEN sm.DEVIATION_FLAG = 'Y' THEN 1 ELSE 0 END)
        / NULLIF(COUNT(*), 0) * 100,
        1
    )                                                   AS DEVIATION_RATE_PCT,
    ROUND(MEDIAN(sm.YIELD_PERCENT), 2)                  AS MEDIAN_YIELD,
    MIN(sm.YIELD_PERCENT)                               AS MIN_YIELD,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY sm.YIELD_PERCENT) AS YIELD_P25,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY sm.YIELD_PERCENT) AS YIELD_P75
FROM SAT_DRUG_MANUFACTURING sm
INNER JOIN HUB_DRUG hd ON sm.DRUG_KEY = hd.DRUG_KEY
INNER JOIN DIM_DRUG dd ON hd.DRUG_KEY = dd.DRUG_VAULT_KEY AND dd.IS_CURRENT = 'Y'
WHERE sm.LOAD_END_DATE = TO_TIMESTAMP('9999-12-31 23:59:59', 'YYYY-MM-DD HH24:MI:SS')
GROUP BY dd.MANUFACTURER, sm.FACILITY_ID, TO_CHAR(sm.MFG_DATE, 'YYYY-Q')
ORDER BY dd.MANUFACTURER, sm.FACILITY_ID, MFG_QUARTER;
