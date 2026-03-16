-- =============================================================================
-- FeatureForge: Feature Store Schema DDL
-- Creates all metadata and storage tables for the feature store,
-- drift monitoring, and model tracking in Snowflake.
-- =============================================================================

-- Use a dedicated database and schema
CREATE DATABASE IF NOT EXISTS FEATURE_STORE;
CREATE SCHEMA IF NOT EXISTS FEATURE_STORE.FEATURE_FORGE;
USE SCHEMA FEATURE_STORE.FEATURE_FORGE;

-- ---------------------------------------------------------------------------
-- 1. Feature Registry - Catalog of all registered features
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS FEATURE_REGISTRY (
    feature_id          VARCHAR(16)     PRIMARY KEY,
    name                VARCHAR(128)    NOT NULL,
    version             INTEGER         NOT NULL DEFAULT 1,
    description         TEXT,
    data_type           VARCHAR(20)     NOT NULL,
    source_table        VARCHAR(256)    NOT NULL,
    source_query        TEXT,
    entity_key          VARCHAR(64)     DEFAULT 'patient_id',
    timestamp_column    VARCHAR(64)     DEFAULT 'feature_ts',
    freshness_sla_hours INTEGER         DEFAULT 24,
    owner               VARCHAR(128)    DEFAULT 'data-engineering',
    tags                VARIANT,            -- JSON array of tag strings
    dependencies        VARIANT,            -- JSON array of dependent feature names
    status              VARCHAR(20)     DEFAULT 'DRAFT',
    created_at          TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP(),
    updated_at          TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP(),
    CONSTRAINT uq_feature_name_version UNIQUE (name, version),
    CONSTRAINT chk_status CHECK (status IN ('DRAFT', 'ACTIVE', 'DEPRECATED', 'ARCHIVED'))
);

-- ---------------------------------------------------------------------------
-- 2. Feature Lineage - DAG of feature dependencies and transformations
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS FEATURE_LINEAGE (
    source_feature      VARCHAR(128)    NOT NULL,
    target_feature      VARCHAR(128)    NOT NULL,
    transformation      TEXT,
    created_at          TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (source_feature, target_feature)
);

-- ---------------------------------------------------------------------------
-- 3. Feature Version History - Audit log of all changes
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS FEATURE_VERSION_HISTORY (
    history_id          NUMBER          AUTOINCREMENT PRIMARY KEY,
    feature_id          VARCHAR(16)     NOT NULL,
    name                VARCHAR(128)    NOT NULL,
    version             INTEGER         NOT NULL,
    change_type         VARCHAR(20)     NOT NULL,   -- CREATE, UPDATE, DEPRECATE, ARCHIVE
    change_details      VARIANT,
    changed_by          VARCHAR(128),
    changed_at          TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP()
);

-- ---------------------------------------------------------------------------
-- 4. Feature Values - Materialized feature storage (per entity + timestamp)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS FEATURE_VALUES (
    entity_key          VARCHAR(64)     NOT NULL,
    entity_id           VARCHAR(128)    NOT NULL,
    feature_name        VARCHAR(128)    NOT NULL,
    feature_version     INTEGER         NOT NULL DEFAULT 1,
    feature_value       VARIANT,            -- Supports any data type
    feature_ts          TIMESTAMP_NTZ   NOT NULL,
    ingestion_ts        TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP(),
    batch_id            VARCHAR(64),
    CONSTRAINT pk_feature_values PRIMARY KEY (entity_id, feature_name, feature_version, feature_ts)
);

-- Clustering for efficient point-in-time lookups
ALTER TABLE FEATURE_VALUES CLUSTER BY (entity_id, feature_name, feature_ts);

-- ---------------------------------------------------------------------------
-- 5. Feature Groups - Logical grouping of related features
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS FEATURE_GROUPS (
    group_id            VARCHAR(16)     PRIMARY KEY,
    group_name          VARCHAR(128)    NOT NULL UNIQUE,
    description         TEXT,
    entity_key          VARCHAR(64)     DEFAULT 'patient_id',
    feature_names       VARIANT,            -- JSON array of feature names
    owner               VARCHAR(128),
    created_at          TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP(),
    updated_at          TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP()
);

-- ---------------------------------------------------------------------------
-- 6. Drift Reports - Historical drift detection results
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS DRIFT_REPORTS (
    report_id           VARCHAR(64)     PRIMARY KEY,
    generated_at        TIMESTAMP_NTZ   NOT NULL,
    total_features      INTEGER         NOT NULL,
    drifted_features    INTEGER         NOT NULL,
    overall_severity    VARCHAR(20)     NOT NULL,
    report_data         VARIANT,            -- Full JSON report
    pipeline_run_id     VARCHAR(64),
    created_at          TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP()
);

-- ---------------------------------------------------------------------------
-- 7. Drift Baselines - Stored baseline distributions for drift comparison
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS DRIFT_BASELINES (
    baseline_id         VARCHAR(64)     PRIMARY KEY,
    feature_name        VARCHAR(128)    NOT NULL,
    feature_version     INTEGER         NOT NULL DEFAULT 1,
    baseline_start      TIMESTAMP_NTZ   NOT NULL,
    baseline_end        TIMESTAMP_NTZ   NOT NULL,
    distribution_type   VARCHAR(20)     NOT NULL,   -- NUMERIC, CATEGORICAL
    statistics          VARIANT,            -- JSON: mean, std, histogram, etc.
    constraints         VARIANT,            -- JSON: min, max, allowed values
    sample_count        INTEGER,
    created_at          TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP(),
    CONSTRAINT uq_baseline UNIQUE (feature_name, feature_version, baseline_start, baseline_end)
);

-- ---------------------------------------------------------------------------
-- 8. Model Predictions Log - For model drift monitoring
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS MODEL_PREDICTIONS (
    prediction_id       VARCHAR(64)     PRIMARY KEY,
    patient_id          VARCHAR(128)    NOT NULL,
    model_name          VARCHAR(128)    NOT NULL,
    model_version       VARCHAR(32)     NOT NULL,
    prediction_score    FLOAT           NOT NULL,
    prediction_label    INTEGER,
    feature_vector      VARIANT,            -- Input features used (for explainability)
    prediction_ts       TIMESTAMP_NTZ   NOT NULL,
    endpoint_name       VARCHAR(128),
    latency_ms          FLOAT,
    created_at          TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP()
);

-- Index for efficient time-range queries
ALTER TABLE MODEL_PREDICTIONS CLUSTER BY (model_name, prediction_ts);

-- ---------------------------------------------------------------------------
-- 9. Ground Truth Labels - For accuracy monitoring
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS GROUND_TRUTH_LABELS (
    label_id            VARCHAR(64)     PRIMARY KEY,
    patient_id          VARCHAR(128)    NOT NULL,
    label_name          VARCHAR(128)    NOT NULL,
    label_value         INTEGER         NOT NULL,
    label_source        VARCHAR(64),        -- CLAIMS, EHR, MANUAL_REVIEW
    effective_date      DATE            NOT NULL,
    created_at          TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP(),
    CONSTRAINT uq_ground_truth UNIQUE (patient_id, label_name, effective_date)
);

-- ---------------------------------------------------------------------------
-- 10. Model Monitor Alerts - Track all drift alerts
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS DRIFT_ALERTS (
    alert_id            VARCHAR(64)     PRIMARY KEY,
    alert_type          VARCHAR(32)     NOT NULL,   -- FEATURE_DRIFT, MODEL_DRIFT, ACCURACY_DROP
    severity            VARCHAR(20)     NOT NULL,
    feature_name        VARCHAR(128),
    model_name          VARCHAR(128),
    model_version       VARCHAR(32),
    drift_statistic     FLOAT,
    drift_threshold     FLOAT,
    message             TEXT,
    acknowledged        BOOLEAN         DEFAULT FALSE,
    acknowledged_by     VARCHAR(128),
    acknowledged_at     TIMESTAMP_NTZ,
    action_taken        VARCHAR(32),        -- RETRAIN, ROLLBACK, INVESTIGATE, IGNORE
    created_at          TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP()
);

-- ---------------------------------------------------------------------------
-- 11. Retraining History - Track model retraining events
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS RETRAINING_HISTORY (
    retrain_id          VARCHAR(64)     PRIMARY KEY,
    model_name          VARCHAR(128)    NOT NULL,
    trigger_type        VARCHAR(32)     NOT NULL,   -- DRIFT, SCHEDULED, MANUAL
    trigger_details     VARIANT,
    pipeline_arn        VARCHAR(512),
    status              VARCHAR(20)     DEFAULT 'PENDING',
    started_at          TIMESTAMP_NTZ,
    completed_at        TIMESTAMP_NTZ,
    new_model_version   VARCHAR(32),
    metrics_before      VARIANT,
    metrics_after       VARIANT,
    created_at          TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP()
);

-- ---------------------------------------------------------------------------
-- Views for common queries
-- ---------------------------------------------------------------------------

-- Active features with their freshness status
CREATE OR REPLACE VIEW V_FEATURE_FRESHNESS AS
SELECT
    fr.name,
    fr.version,
    fr.source_table,
    fr.freshness_sla_hours,
    fr.owner,
    fv.latest_ts,
    DATEDIFF('hour', fv.latest_ts, CURRENT_TIMESTAMP()) AS hours_since_update,
    CASE
        WHEN fv.latest_ts IS NULL THEN 'NO_DATA'
        WHEN DATEDIFF('hour', fv.latest_ts, CURRENT_TIMESTAMP()) > fr.freshness_sla_hours THEN 'STALE'
        ELSE 'FRESH'
    END AS freshness_status
FROM FEATURE_REGISTRY fr
LEFT JOIN (
    SELECT feature_name, feature_version, MAX(feature_ts) AS latest_ts
    FROM FEATURE_VALUES
    GROUP BY feature_name, feature_version
) fv ON fr.name = fv.feature_name AND fr.version = fv.feature_version
WHERE fr.status = 'ACTIVE';

-- Recent drift summary
CREATE OR REPLACE VIEW V_DRIFT_SUMMARY AS
SELECT
    report_id,
    generated_at,
    total_features,
    drifted_features,
    overall_severity,
    ROUND(drifted_features::FLOAT / NULLIF(total_features, 0) * 100, 1) AS drift_pct,
    report_data:results AS detailed_results
FROM DRIFT_REPORTS
ORDER BY generated_at DESC;

-- Model performance over time
CREATE OR REPLACE VIEW V_MODEL_PERFORMANCE AS
SELECT
    mp.model_name,
    mp.model_version,
    DATE_TRUNC('day', mp.prediction_ts) AS prediction_date,
    COUNT(*) AS prediction_count,
    AVG(mp.prediction_score) AS avg_prediction_score,
    STDDEV(mp.prediction_score) AS std_prediction_score,
    AVG(mp.latency_ms) AS avg_latency_ms,
    -- Join with ground truth for accuracy
    AVG(CASE WHEN gt.label_value IS NOT NULL
        THEN CASE WHEN ROUND(mp.prediction_score) = gt.label_value THEN 1.0 ELSE 0.0 END
        ELSE NULL
    END) AS daily_accuracy
FROM MODEL_PREDICTIONS mp
LEFT JOIN GROUND_TRUTH_LABELS gt
    ON mp.patient_id = gt.patient_id
    AND gt.label_name = mp.model_name
GROUP BY mp.model_name, mp.model_version, DATE_TRUNC('day', mp.prediction_ts)
ORDER BY prediction_date DESC;

-- Grant permissions (adjust role names as needed)
GRANT SELECT ON ALL TABLES IN SCHEMA FEATURE_STORE.FEATURE_FORGE TO ROLE DATA_SCIENTIST;
GRANT SELECT ON ALL VIEWS IN SCHEMA FEATURE_STORE.FEATURE_FORGE TO ROLE DATA_SCIENTIST;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA FEATURE_STORE.FEATURE_FORGE TO ROLE DATA_ENGINEER;
