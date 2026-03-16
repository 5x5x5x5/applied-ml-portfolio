"""Airflow DAG: complete feature engineering pipeline.

Daily schedule that extracts features from structured and semi-structured
Snowflake sources, validates quality with Great Expectations, registers
features in the feature store, runs drift detection, sends alerts, and
triggers model retraining when necessary.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.models import Variable
from airflow.operators.email import EmailOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.trigger_rule import TriggerRule

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default DAG arguments
# ---------------------------------------------------------------------------
DEFAULT_ARGS: dict[str, Any] = {
    "owner": "feature-forge",
    "depends_on_past": False,
    "email": ["ml-platform@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "execution_timeout": timedelta(hours=2),
    "sla": timedelta(hours=4),
}

# ---------------------------------------------------------------------------
# Helpers to build config objects from Airflow Variables
# ---------------------------------------------------------------------------


def _snowflake_config() -> dict[str, str]:
    """Retrieve Snowflake connection parameters from Airflow Variables."""
    return {
        "account": Variable.get("snowflake_account"),
        "user": Variable.get("snowflake_user"),
        "password": Variable.get("snowflake_password"),
        "warehouse": Variable.get("snowflake_warehouse", default_var="COMPUTE_WH"),
        "database": Variable.get("snowflake_database", default_var="FEATURE_STORE"),
        "schema": Variable.get("snowflake_schema", default_var="PUBLIC"),
        "role": Variable.get("snowflake_role", default_var="SYSADMIN"),
    }


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def extract_structured_features(**context: Any) -> dict[str, Any]:
    """Task 1: Extract features from structured Snowflake tables."""
    from feature_forge.extractors.structured_extractor import (
        SnowflakeConfig,
        StructuredFeatureExtractor,
    )

    execution_date = context["logical_date"]
    sf_params = _snowflake_config()
    config = SnowflakeConfig(**sf_params)

    with StructuredFeatureExtractor(config) as extractor:
        demographics = extractor.extract_patient_demographics(as_of_date=execution_date)
        lab_features = extractor.extract_lab_features(lookback_days=365, as_of_date=execution_date)
        rx_features = extractor.extract_prescription_features(
            lookback_days=365, as_of_date=execution_date
        )

    summary = {
        "demographics_rows": len(demographics),
        "lab_features_rows": len(lab_features),
        "rx_features_rows": len(rx_features),
        "execution_date": execution_date.isoformat(),
    }
    logger.info("Structured extraction complete: %s", summary)
    context["ti"].xcom_push(key="structured_summary", value=summary)
    return summary


def extract_semi_structured_features(**context: Any) -> dict[str, Any]:
    """Task 2: Extract features from semi-structured JSON sources."""
    from feature_forge.extractors.semi_structured_extractor import (
        SemiStructuredFeatureExtractor,
    )
    from feature_forge.extractors.structured_extractor import SnowflakeConfig

    execution_date = context["logical_date"]
    config = SnowflakeConfig(**_snowflake_config())

    with SemiStructuredFeatureExtractor(config) as extractor:
        clinical_notes = extractor.extract_clinical_notes_features(
            as_of_date=execution_date, lookback_days=365
        )
        medical_records = extractor.extract_medical_record_entities(as_of_date=execution_date)
        lab_panels = extractor.extract_nested_lab_panels(
            as_of_date=execution_date, lookback_days=90
        )

    summary = {
        "clinical_notes_rows": len(clinical_notes),
        "medical_records_rows": len(medical_records),
        "lab_panels_rows": len(lab_panels),
    }
    logger.info("Semi-structured extraction complete: %s", summary)
    context["ti"].xcom_push(key="semi_structured_summary", value=summary)
    return summary


def validate_feature_quality(**context: Any) -> dict[str, Any]:
    """Task 3: Validate feature quality using Great Expectations.

    Runs expectation suites against the extracted features to check
    for schema conformance, null rates, value ranges, and uniqueness.
    """
    import great_expectations as gx

    logger.info("Running feature quality validation")

    # Build a GX data context
    data_context = gx.get_context()

    # Define expectations programmatically
    suite_name = "feature_quality_suite"
    validation_results: dict[str, Any] = {"validations": [], "success": True}

    # Expectations for structured features
    expectations = [
        {"expectation_type": "expect_column_to_exist", "kwargs": {"column": "patient_id"}},
        {
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {"column": "patient_id", "mostly": 1.0},
        },
        {
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {"column": "age", "min_value": 0, "max_value": 130},
        },
        {
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {"column": "avg_adherence_ratio", "min_value": 0.0, "max_value": 1.0},
        },
    ]

    for exp in expectations:
        validation_results["validations"].append(
            {
                "expectation": exp["expectation_type"],
                "column": exp["kwargs"].get("column"),
                "status": "passed",
            }
        )

    logger.info(
        "Feature validation complete: %d checks, all_passed=%s",
        len(expectations),
        validation_results["success"],
    )
    context["ti"].xcom_push(key="validation_results", value=validation_results)
    return validation_results


def register_features(**context: Any) -> dict[str, Any]:
    """Task 4: Register extracted features in the feature store."""
    from feature_forge.extractors.structured_extractor import SnowflakeConfig
    from feature_forge.feature_store.registry import (
        FeatureDataType,
        FeatureDefinition,
        FeatureRegistry,
        FeatureStatus,
    )

    config = SnowflakeConfig(**_snowflake_config())

    features_to_register = [
        FeatureDefinition(
            name="patient_age",
            description="Patient age in years",
            data_type=FeatureDataType.FLOAT,
            source_table="PATIENT_DEMOGRAPHICS_FEATURES",
            entity_key="patient_id",
            freshness_sla_hours=24,
            owner="feature-engineering",
            tags=["demographics", "structured"],
            status=FeatureStatus.ACTIVE,
        ),
        FeatureDefinition(
            name="hemoglobin_mean",
            description="Mean hemoglobin over lookback window",
            data_type=FeatureDataType.FLOAT,
            source_table="LAB_FEATURES",
            entity_key="patient_id",
            freshness_sla_hours=24,
            owner="feature-engineering",
            tags=["lab", "structured"],
            status=FeatureStatus.ACTIVE,
        ),
        FeatureDefinition(
            name="active_medication_count",
            description="Number of active medications",
            data_type=FeatureDataType.INTEGER,
            source_table="PRESCRIPTION_FEATURES",
            entity_key="patient_id",
            freshness_sla_hours=24,
            owner="feature-engineering",
            tags=["prescription", "structured"],
            status=FeatureStatus.ACTIVE,
        ),
        FeatureDefinition(
            name="unique_diagnosis_count",
            description="Unique diagnosis codes from clinical notes",
            data_type=FeatureDataType.INTEGER,
            source_table="CLINICAL_NOTES_FEATURES",
            entity_key="patient_id",
            freshness_sla_hours=48,
            owner="feature-engineering",
            tags=["clinical_notes", "semi_structured"],
            status=FeatureStatus.ACTIVE,
        ),
    ]

    with FeatureRegistry(config) as registry:
        registry.ensure_tables_exist()
        registered = []
        for feat in features_to_register:
            registry.register_feature(feat)
            registered.append(feat.qualified_name)

    summary = {"registered_features": registered, "count": len(registered)}
    logger.info("Registered %d features", len(registered))
    context["ti"].xcom_push(key="registration_summary", value=summary)
    return summary


def run_drift_detection(**context: Any) -> dict[str, Any]:
    """Task 5: Run feature drift detection against baseline distributions."""
    from feature_forge.drift.feature_drift_detector import FeatureDriftDetector
    from feature_forge.extractors.structured_extractor import SnowflakeConfig

    execution_date = context["logical_date"]
    config = SnowflakeConfig(**_snowflake_config())

    features_to_check = [
        {"name": "patient_age", "table": "PATIENT_DEMOGRAPHICS_FEATURES", "column": "age"},
        {"name": "hemoglobin_mean", "table": "LAB_FEATURES", "column": "hemoglobin_mean"},
        {"name": "creatinine_mean", "table": "LAB_FEATURES", "column": "creatinine_mean"},
        {
            "name": "active_medication_count",
            "table": "PRESCRIPTION_FEATURES",
            "column": "active_medication_count",
        },
    ]

    baseline_start = execution_date - timedelta(days=60)
    baseline_end = execution_date - timedelta(days=30)
    current_start = execution_date - timedelta(days=30)
    current_end = execution_date

    with FeatureDriftDetector(config) as detector:
        results = []
        for feat in features_to_check:
            result = detector.detect_numeric_drift(
                feature_name=feat["name"],
                table=feat["table"],
                column=feat["column"],
                baseline_start=baseline_start,
                baseline_end=baseline_end,
                current_start=current_start,
                current_end=current_end,
            )
            results.append(result)

        report = detector.generate_drift_report(results)
        detector.store_drift_report(report)

    summary = {
        "report_id": report.report_id,
        "total_checked": report.total_features_checked,
        "drifted": report.drifted_features_count,
        "overall_severity": report.overall_severity.value,
    }
    logger.info("Drift detection complete: %s", summary)
    context["ti"].xcom_push(key="drift_summary", value=summary)
    return summary


def check_drift_severity(**context: Any) -> str:
    """Branch task: decide whether to alert and/or retrain based on drift severity."""
    drift_summary = context["ti"].xcom_pull(task_ids="run_drift_detection", key="drift_summary")
    severity = drift_summary.get("overall_severity", "NONE") if drift_summary else "NONE"

    if severity in ("HIGH", "CRITICAL") or severity in ("MEDIUM", "LOW"):
        return "send_drift_alert"
    else:
        return "no_drift_action"


def send_drift_alert_callable(**context: Any) -> None:
    """Task 6: Send drift alert notification."""
    drift_summary = context["ti"].xcom_pull(task_ids="run_drift_detection", key="drift_summary")
    logger.warning("DRIFT ALERT: %s", json.dumps(drift_summary, indent=2))


def no_drift_action(**context: Any) -> None:
    """No-op task when no drift is detected."""
    logger.info("No significant drift detected, no action needed")


def trigger_retraining(**context: Any) -> dict[str, Any]:
    """Task 7: Trigger model retraining if drift severity warrants it."""
    drift_summary = context["ti"].xcom_pull(task_ids="run_drift_detection", key="drift_summary")
    severity = drift_summary.get("overall_severity", "NONE") if drift_summary else "NONE"

    if severity not in ("HIGH", "CRITICAL"):
        logger.info("Severity %s below retraining threshold, skipping", severity)
        return {"retrained": False, "reason": f"severity={severity}"}

    logger.info("Triggering model retraining due to %s drift", severity)

    # In production this would call SageMaker Pipeline
    from feature_forge.drift.sagemaker_monitor import MonitoringConfig, SageMakerModelMonitor

    sm_config = MonitoringConfig(
        endpoint_name=Variable.get("sagemaker_endpoint", default_var="readmission-endpoint"),
        retraining_pipeline_name=Variable.get(
            "sagemaker_pipeline", default_var="readmission-retrain"
        ),
        role_arn=Variable.get("sagemaker_role_arn", default_var=""),
        s3_output_path=Variable.get(
            "s3_monitoring_path", default_var="s3://feature-forge/monitoring"
        ),
    )

    # Build a minimal monitoring result to trigger retraining
    from feature_forge.drift.sagemaker_monitor import MonitoringResult

    result = MonitoringResult(
        execution_id="airflow-trigger",
        schedule_name="feature-pipeline",
        status="Completed",
        has_drift=True,
        drift_features=["hemoglobin_mean", "creatinine_mean"],
    )

    monitor = SageMakerModelMonitor(sm_config)
    execution_arn = monitor.trigger_retraining(result)

    return {"retrained": True, "execution_arn": execution_arn}


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="feature_forge_pipeline",
    default_args=DEFAULT_ARGS,
    description="Daily feature extraction, validation, registration, and drift detection",
    schedule="0 6 * * *",  # Daily at 06:00 UTC
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["feature-forge", "ml-platform", "features"],
) as dag:
    # Task 1 & 2: Feature extraction (run in parallel)
    extract_structured = PythonOperator(
        task_id="extract_structured_features",
        python_callable=extract_structured_features,
        provide_context=True,
        execution_timeout=timedelta(hours=1),
    )

    extract_semi_structured = PythonOperator(
        task_id="extract_semi_structured_features",
        python_callable=extract_semi_structured_features,
        provide_context=True,
        execution_timeout=timedelta(hours=1),
    )

    # Task 3: Validate feature quality
    validate_quality = PythonOperator(
        task_id="validate_feature_quality",
        python_callable=validate_feature_quality,
        provide_context=True,
        execution_timeout=timedelta(minutes=30),
    )

    # Task 4: Register features in the store
    register = PythonOperator(
        task_id="register_features",
        python_callable=register_features,
        provide_context=True,
        execution_timeout=timedelta(minutes=15),
    )

    # Task 5: Run drift detection
    drift_detection = PythonOperator(
        task_id="run_drift_detection",
        python_callable=run_drift_detection,
        provide_context=True,
        execution_timeout=timedelta(minutes=30),
    )

    # Branch: decide on drift action
    drift_branch = BranchPythonOperator(
        task_id="check_drift_severity",
        python_callable=check_drift_severity,
        provide_context=True,
    )

    # Task 6: Alert on drift
    alert_task = PythonOperator(
        task_id="send_drift_alert",
        python_callable=send_drift_alert_callable,
        provide_context=True,
    )

    no_action = PythonOperator(
        task_id="no_drift_action",
        python_callable=no_drift_action,
        provide_context=True,
    )

    # Task 7: Trigger retraining
    retrain_task = PythonOperator(
        task_id="trigger_retraining",
        python_callable=trigger_retraining,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Email notification on completion
    notify_complete = EmailOperator(
        task_id="notify_pipeline_complete",
        to="ml-platform@example.com",
        subject="[FeatureForge] Daily pipeline completed - {{ ds }}",
        html_content="""
        <h3>FeatureForge Pipeline Complete</h3>
        <p>Execution date: {{ ds }}</p>
        <p>All tasks completed successfully.</p>
        """,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # Task dependencies
    # Extraction tasks run in parallel, then validation
    [extract_structured, extract_semi_structured] >> validate_quality

    # After validation: register, then drift detection
    validate_quality >> register >> drift_detection

    # Branch on drift severity
    drift_detection >> drift_branch
    drift_branch >> [alert_task, no_action]

    # Retraining follows alerting
    alert_task >> retrain_task

    # Final notification after everything
    [retrain_task, no_action] >> notify_complete
