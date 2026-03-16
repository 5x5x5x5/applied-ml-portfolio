-- =============================================================================
-- DrugInteractionML: SnowSQL for Computing Feature Distribution Statistics
-- Used for drift detection by comparing production vs. training distributions.
-- =============================================================================

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. Production feature distribution statistics (for each numeric feature)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE TEMPORARY TABLE _production_feature_stats AS
WITH feature_values AS (
    SELECT
        feature_name,
        feature_value,
        prediction_date
    FROM PHARMA_DB.DRUG_INTERACTION.production_features
    WHERE prediction_date >= :analysis_start_date
      AND prediction_date < :analysis_end_date
),
basic_stats AS (
    SELECT
        feature_name,
        COUNT(*) AS sample_count,
        AVG(feature_value) AS mean_val,
        STDDEV(feature_value) AS std_val,
        VARIANCE(feature_value) AS var_val,
        MIN(feature_value) AS min_val,
        MAX(feature_value) AS max_val,
        MEDIAN(feature_value) AS median_val,
        -- Robust statistics
        PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY feature_value) AS p01,
        PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY feature_value) AS p05,
        PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY feature_value) AS p10,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY feature_value) AS p25,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY feature_value) AS p50,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY feature_value) AS p75,
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY feature_value) AS p90,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY feature_value) AS p95,
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY feature_value) AS p99,
        -- IQR for outlier detection
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY feature_value)
            - PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY feature_value) AS iqr,
        -- Skewness proxy
        (AVG(feature_value) - MEDIAN(feature_value)) / NULLIF(STDDEV(feature_value), 0) AS skewness_proxy,
        -- Count nulls and zeros
        SUM(CASE WHEN feature_value IS NULL THEN 1 ELSE 0 END) AS null_count,
        SUM(CASE WHEN feature_value = 0 THEN 1 ELSE 0 END) AS zero_count
    FROM feature_values
    GROUP BY feature_name
)
SELECT
    bs.*,
    bs.null_count::FLOAT / NULLIF(bs.sample_count, 0) AS null_rate,
    bs.zero_count::FLOAT / NULLIF(bs.sample_count, 0) AS zero_rate,
    -- Coefficient of variation
    bs.std_val / NULLIF(bs.mean_val, 0) AS coeff_of_variation
FROM basic_stats bs;


-- ─────────────────────────────────────────────────────────────────────────────
-- 2. Histogram bins for PSI calculation (10 equal-frequency bins per feature)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE TEMPORARY TABLE _production_histograms AS
WITH feature_with_deciles AS (
    SELECT
        feature_name,
        feature_value,
        NTILE(10) OVER (PARTITION BY feature_name ORDER BY feature_value) AS decile_bin
    FROM PHARMA_DB.DRUG_INTERACTION.production_features
    WHERE prediction_date >= :analysis_start_date
      AND prediction_date < :analysis_end_date
      AND feature_value IS NOT NULL
)
SELECT
    feature_name,
    decile_bin,
    COUNT(*) AS bin_count,
    MIN(feature_value) AS bin_min,
    MAX(feature_value) AS bin_max,
    AVG(feature_value) AS bin_mean
FROM feature_with_deciles
GROUP BY feature_name, decile_bin
ORDER BY feature_name, decile_bin;


-- ─────────────────────────────────────────────────────────────────────────────
-- 3. Training baseline statistics (reference period)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE TEMPORARY TABLE _baseline_feature_stats AS
SELECT
    feature_name,
    COUNT(*) AS sample_count,
    AVG(feature_value) AS mean_val,
    STDDEV(feature_value) AS std_val,
    VARIANCE(feature_value) AS var_val,
    MIN(feature_value) AS min_val,
    MAX(feature_value) AS max_val,
    MEDIAN(feature_value) AS median_val,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY feature_value) AS p25,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY feature_value) AS p50,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY feature_value) AS p75,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY feature_value)
        - PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY feature_value) AS iqr,
    SUM(CASE WHEN feature_value IS NULL THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS null_rate,
    SUM(CASE WHEN feature_value = 0 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS zero_rate
FROM PHARMA_DB.DRUG_INTERACTION.training_baseline_features
GROUP BY feature_name;


CREATE OR REPLACE TEMPORARY TABLE _baseline_histograms AS
WITH baseline_deciles AS (
    SELECT
        feature_name,
        feature_value,
        NTILE(10) OVER (PARTITION BY feature_name ORDER BY feature_value) AS decile_bin
    FROM PHARMA_DB.DRUG_INTERACTION.training_baseline_features
    WHERE feature_value IS NOT NULL
)
SELECT
    feature_name,
    decile_bin,
    COUNT(*) AS bin_count,
    MIN(feature_value) AS bin_min,
    MAX(feature_value) AS bin_max
FROM baseline_deciles
GROUP BY feature_name, decile_bin
ORDER BY feature_name, decile_bin;


-- ─────────────────────────────────────────────────────────────────────────────
-- 4. PSI calculation: compare production vs baseline histograms
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE TEMPORARY TABLE _psi_results AS
WITH bin_proportions AS (
    SELECT
        b.feature_name,
        b.decile_bin,
        b.bin_count::FLOAT / SUM(b.bin_count) OVER (PARTITION BY b.feature_name) AS baseline_pct,
        p.bin_count::FLOAT / SUM(p.bin_count) OVER (PARTITION BY p.feature_name) AS production_pct
    FROM _baseline_histograms b
    JOIN _production_histograms p
        ON b.feature_name = p.feature_name
        AND b.decile_bin = p.decile_bin
),
psi_per_bin AS (
    SELECT
        feature_name,
        decile_bin,
        baseline_pct,
        production_pct,
        -- Clip to avoid log(0)
        GREATEST(baseline_pct, 0.0001) AS safe_baseline,
        GREATEST(production_pct, 0.0001) AS safe_production,
        (GREATEST(production_pct, 0.0001) - GREATEST(baseline_pct, 0.0001))
            * LN(GREATEST(production_pct, 0.0001) / GREATEST(baseline_pct, 0.0001)) AS psi_bin
    FROM bin_proportions
)
SELECT
    feature_name,
    SUM(psi_bin) AS psi,
    CASE
        WHEN SUM(psi_bin) >= 0.5 THEN 'CRITICAL'
        WHEN SUM(psi_bin) >= 0.3 THEN 'HIGH'
        WHEN SUM(psi_bin) >= 0.2 THEN 'MODERATE'
        WHEN SUM(psi_bin) >= 0.1 THEN 'LOW'
        ELSE 'NONE'
    END AS drift_severity,
    SUM(psi_bin) >= 0.2 AS is_drifted
FROM psi_per_bin
GROUP BY feature_name
ORDER BY psi DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- 5. Statistical comparison: mean/std shift between baseline and production
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE TEMPORARY TABLE _statistical_comparison AS
SELECT
    b.feature_name,
    b.sample_count AS baseline_samples,
    p.sample_count AS production_samples,
    b.mean_val AS baseline_mean,
    p.mean_val AS production_mean,
    ABS(p.mean_val - b.mean_val) AS mean_shift,
    ABS(p.mean_val - b.mean_val) / NULLIF(b.std_val, 0) AS standardized_mean_shift,
    b.std_val AS baseline_std,
    p.std_val AS production_std,
    ABS(p.std_val - b.std_val) AS std_shift,
    -- Null rate change
    ABS(p.null_rate - b.null_rate) AS null_rate_shift,
    -- Zero rate change
    ABS(p.zero_rate - b.zero_rate) AS zero_rate_shift,
    -- Welch's t-test statistic approximation
    (p.mean_val - b.mean_val) / NULLIF(
        SQRT(
            (POWER(p.std_val, 2) / NULLIF(p.sample_count, 0))
            + (POWER(b.std_val, 2) / NULLIF(b.sample_count, 0))
        ), 0
    ) AS welch_t_stat,
    psi.psi,
    psi.drift_severity,
    psi.is_drifted
FROM _baseline_feature_stats b
JOIN _production_feature_stats p ON b.feature_name = p.feature_name
LEFT JOIN _psi_results psi ON b.feature_name = psi.feature_name
ORDER BY psi.psi DESC NULLS LAST;


-- ─────────────────────────────────────────────────────────────────────────────
-- 6. Prediction distribution drift
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE TEMPORARY TABLE _prediction_drift AS
WITH production_predictions AS (
    SELECT
        prediction_class,
        prediction_probability,
        prediction_date,
        NTILE(10) OVER (ORDER BY prediction_probability) AS prob_decile
    FROM PHARMA_DB.DRUG_INTERACTION.production_predictions
    WHERE prediction_date >= :analysis_start_date
      AND prediction_date < :analysis_end_date
),
baseline_predictions AS (
    SELECT
        prediction_class,
        prediction_probability,
        NTILE(10) OVER (ORDER BY prediction_probability) AS prob_decile
    FROM PHARMA_DB.DRUG_INTERACTION.training_baseline_predictions
),
class_distributions AS (
    SELECT
        'production' AS source,
        prediction_class,
        COUNT(*)::FLOAT / SUM(COUNT(*)) OVER () AS class_proportion
    FROM production_predictions
    GROUP BY prediction_class
    UNION ALL
    SELECT
        'baseline' AS source,
        prediction_class,
        COUNT(*)::FLOAT / SUM(COUNT(*)) OVER () AS class_proportion
    FROM baseline_predictions
    GROUP BY prediction_class
)
SELECT
    cd_prod.prediction_class,
    cd_prod.class_proportion AS production_proportion,
    cd_base.class_proportion AS baseline_proportion,
    ABS(cd_prod.class_proportion - cd_base.class_proportion) AS proportion_shift,
    -- PSI for each class
    (GREATEST(cd_prod.class_proportion, 0.0001) - GREATEST(cd_base.class_proportion, 0.0001))
        * LN(GREATEST(cd_prod.class_proportion, 0.0001) / GREATEST(cd_base.class_proportion, 0.0001)) AS class_psi
FROM class_distributions cd_prod
JOIN class_distributions cd_base
    ON cd_prod.prediction_class = cd_base.prediction_class
    AND cd_prod.source = 'production'
    AND cd_base.source = 'baseline'
ORDER BY proportion_shift DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- 7. Temporal drift analysis: feature stats over sliding windows
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE TEMPORARY TABLE _temporal_drift AS
WITH daily_stats AS (
    SELECT
        feature_name,
        prediction_date,
        AVG(feature_value) AS daily_mean,
        STDDEV(feature_value) AS daily_std,
        COUNT(*) AS daily_count,
        -- 7-day moving average
        AVG(AVG(feature_value)) OVER (
            PARTITION BY feature_name
            ORDER BY prediction_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS ma_7d_mean,
        -- Detect sudden shifts (z-score of daily mean vs 7-day window)
        (AVG(feature_value) - AVG(AVG(feature_value)) OVER (
            PARTITION BY feature_name
            ORDER BY prediction_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        )) / NULLIF(STDDEV(AVG(feature_value)) OVER (
            PARTITION BY feature_name
            ORDER BY prediction_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ), 0) AS daily_zscore
    FROM PHARMA_DB.DRUG_INTERACTION.production_features
    WHERE prediction_date >= :analysis_start_date
      AND prediction_date < :analysis_end_date
    GROUP BY feature_name, prediction_date
)
SELECT
    feature_name,
    prediction_date,
    daily_mean,
    daily_std,
    daily_count,
    ma_7d_mean,
    daily_zscore,
    ABS(daily_zscore) > 3.0 AS is_anomalous_day,
    -- Trend direction over the window
    REGR_SLOPE(daily_mean, DATEDIFF('day', :analysis_start_date, prediction_date))
        OVER (PARTITION BY feature_name) AS overall_trend_slope
FROM daily_stats
ORDER BY feature_name, prediction_date;


-- ─────────────────────────────────────────────────────────────────────────────
-- 8. Summary report: all drift metrics in a single view
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    sc.feature_name,
    sc.baseline_samples,
    sc.production_samples,
    sc.baseline_mean,
    sc.production_mean,
    sc.mean_shift,
    sc.standardized_mean_shift,
    sc.baseline_std,
    sc.production_std,
    sc.std_shift,
    sc.null_rate_shift,
    sc.welch_t_stat,
    sc.psi,
    sc.drift_severity,
    sc.is_drifted,
    -- Temporal anomaly count
    (SELECT COUNT(*) FROM _temporal_drift td
     WHERE td.feature_name = sc.feature_name AND td.is_anomalous_day) AS anomalous_days,
    -- Overall trend
    (SELECT MAX(overall_trend_slope) FROM _temporal_drift td
     WHERE td.feature_name = sc.feature_name) AS trend_slope
FROM _statistical_comparison sc
ORDER BY
    CASE sc.drift_severity
        WHEN 'CRITICAL' THEN 1
        WHEN 'HIGH' THEN 2
        WHEN 'MODERATE' THEN 3
        WHEN 'LOW' THEN 4
        ELSE 5
    END,
    sc.psi DESC NULLS LAST;
