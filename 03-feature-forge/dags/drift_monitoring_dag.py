"""Airflow DAG: dedicated drift monitoring pipeline.

Runs hourly to monitor feature drift, model prediction drift, and
concept drift. Integrates with SageMaker Model Monitor and publishes
CloudWatch metrics.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import BranchPythonOperator, PythonOperator

logger = logging.getLogger(__name__)

DEFAULT_ARGS: dict[str, Any] = {
    "owner": "feature-forge",
    "depends_on_past": False,
    "email": ["ml-platform@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
    "execution_timeout": timedelta(minutes=45),
}


def _snowflake_config() -> dict[str, str]:
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


def check_feature_drift(**context: Any) -> dict[str, Any]:
    """Check all registered features for distributional drift."""
    from feature_forge.drift.feature_drift_detector import FeatureDriftDetector
    from feature_forge.extractors.structured_extractor import SnowflakeConfig
    from feature_forge.feature_store.registry import FeatureRegistry, FeatureStatus

    execution_date = context["logical_date"]
    config = SnowflakeConfig(**_snowflake_config())

    # Get active features from registry
    with FeatureRegistry(config) as registry:
        active_features = registry.list_features(status=FeatureStatus.ACTIVE)

    if not active_features:
        logger.info("No active features to monitor")
        return {"checked": 0, "drifted": 0, "severity": "NONE"}

    baseline_start = execution_date - timedelta(days=30)
    baseline_end = execution_date - timedelta(days=7)
    current_start = execution_date - timedelta(days=7)
    current_end = execution_date

    with FeatureDriftDetector(config) as detector:
        results = []
        for feat in active_features:
            try:
                result = detector.detect_numeric_drift(
                    feature_name=feat.name,
                    table=feat.source_table,
                    column=feat.name,
                    baseline_start=baseline_start,
                    baseline_end=baseline_end,
                    current_start=current_start,
                    current_end=current_end,
                    timestamp_col=feat.timestamp_column,
                )
                results.append(result)
            except Exception:
                logger.exception("Failed drift check for feature %s", feat.name)

        report = detector.generate_drift_report(results)
        detector.store_drift_report(report)

    summary = {
        "report_id": report.report_id,
        "checked": report.total_features_checked,
        "drifted": report.drifted_features_count,
        "severity": report.overall_severity.value,
    }
    context["ti"].xcom_push(key="feature_drift", value=summary)
    return summary


def check_model_drift(**context: Any) -> dict[str, Any]:
    """Check model prediction distribution and accuracy for drift."""
    from feature_forge.drift.feature_drift_detector import FeatureDriftDetector
    from feature_forge.drift.model_drift_detector import (
        ModelDriftDetector,
        ModelMonitorConfig,
    )
    from feature_forge.extractors.structured_extractor import SnowflakeConfig

    execution_date = context["logical_date"]
    config = SnowflakeConfig(**_snowflake_config())

    monitor_config = ModelMonitorConfig(
        model_name=Variable.get("model_name", default_var="readmission_model"),
        model_version=Variable.get("model_version", default_var="1.0"),
        predictions_table="MODEL_PREDICTIONS",
        ground_truth_table="GROUND_TRUTH_LABELS",
        prediction_column="prediction_score",
        label_column="readmitted",
        timestamp_column="prediction_ts",
        baseline_days=30,
        current_window_days=7,
    )

    feature_detector = FeatureDriftDetector(config)

    with ModelDriftDetector(config, monitor_config, feature_detector) as detector:
        results = detector.run_full_check(
            as_of_date=execution_date,
            feature_table="MODEL_INPUT_FEATURES",
            feature_columns=[
                "age",
                "hemoglobin_mean",
                "creatinine_mean",
                "active_medication_count",
                "unique_diagnosis_count",
            ],
        )

    any_retrain = any(r.should_retrain for r in results)
    max_severity = max((r.severity.value for r in results), default="NONE")

    summary = {
        "total_checks": len(results),
        "drifted": sum(1 for r in results if r.is_drifted),
        "retrain_recommended": any_retrain,
        "max_severity": max_severity,
    }
    context["ti"].xcom_push(key="model_drift", value=summary)
    return summary


def check_sagemaker_monitor(**context: Any) -> dict[str, Any]:
    """Process latest SageMaker Model Monitor execution results."""
    from feature_forge.drift.sagemaker_monitor import (
        MonitoringConfig,
        SageMakerModelMonitor,
    )

    sm_config = MonitoringConfig(
        endpoint_name=Variable.get("sagemaker_endpoint", default_var="readmission-endpoint"),
        s3_output_path=Variable.get(
            "s3_monitoring_path", default_var="s3://feature-forge/monitoring"
        ),
        role_arn=Variable.get("sagemaker_role_arn", default_var=""),
        sns_topic_arn=Variable.get("sns_drift_topic", default_var=""),
        retraining_pipeline_name=Variable.get("sagemaker_pipeline", default_var=""),
        cloudwatch_namespace="FeatureForge/DriftMonitoring",
    )

    schedule_name = f"{sm_config.endpoint_name}-monitor"
    monitor = SageMakerModelMonitor(sm_config)

    try:
        result = monitor.run_monitoring_cycle(schedule_name)
        summary = {
            "violations": result.constraint_violations_count,
            "has_drift": result.has_drift,
            "drift_features": result.drift_features,
        }
    except Exception:
        logger.exception("SageMaker monitor check failed")
        summary = {"violations": 0, "has_drift": False, "drift_features": [], "error": True}

    context["ti"].xcom_push(key="sagemaker_drift", value=summary)
    return summary


def evaluate_drift_action(**context: Any) -> str:
    """Evaluate all drift checks and decide on action."""
    feature_drift = context["ti"].xcom_pull(task_ids="check_feature_drift", key="feature_drift")
    model_drift = context["ti"].xcom_pull(task_ids="check_model_drift", key="model_drift")
    sm_drift = context["ti"].xcom_pull(task_ids="check_sagemaker_monitor", key="sagemaker_drift")

    should_alert = False
    should_retrain = False

    if feature_drift and feature_drift.get("severity") in ("HIGH", "CRITICAL"):
        should_alert = True
    if model_drift and model_drift.get("retrain_recommended"):
        should_retrain = True
        should_alert = True
    if sm_drift and sm_drift.get("has_drift"):
        should_alert = True

    if should_retrain:
        return "trigger_model_retrain"
    elif should_alert:
        return "send_drift_notification"
    return "log_no_drift"


def send_drift_notification(**context: Any) -> None:
    """Send notification about detected drift."""
    feature_drift = context["ti"].xcom_pull(task_ids="check_feature_drift", key="feature_drift")
    model_drift = context["ti"].xcom_pull(task_ids="check_model_drift", key="model_drift")

    logger.warning(
        "Drift notification: feature_drift=%s, model_drift=%s",
        feature_drift,
        model_drift,
    )


def trigger_model_retrain(**context: Any) -> dict[str, Any]:
    """Trigger model retraining through SageMaker pipeline."""
    from feature_forge.drift.sagemaker_monitor import (
        MonitoringConfig,
        MonitoringResult,
        SageMakerModelMonitor,
    )

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

    monitor = SageMakerModelMonitor(sm_config)
    result = MonitoringResult(
        execution_id="hourly-drift-trigger",
        schedule_name="drift-monitoring",
        status="Completed",
        has_drift=True,
        drift_features=["detected_via_hourly_monitor"],
    )

    execution_arn = monitor.trigger_retraining(result)
    logger.info("Retraining triggered: %s", execution_arn)
    return {"triggered": True, "execution_arn": execution_arn}


def log_no_drift(**context: Any) -> None:
    """Log that no significant drift was detected."""
    logger.info("Hourly drift check: no significant drift detected")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="drift_monitoring_hourly",
    default_args=DEFAULT_ARGS,
    description="Hourly drift monitoring for features and models",
    schedule="0 * * * *",  # Every hour
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["feature-forge", "drift-monitoring", "ml-platform"],
) as dag:
    feature_drift_task = PythonOperator(
        task_id="check_feature_drift",
        python_callable=check_feature_drift,
        provide_context=True,
    )

    model_drift_task = PythonOperator(
        task_id="check_model_drift",
        python_callable=check_model_drift,
        provide_context=True,
    )

    sagemaker_task = PythonOperator(
        task_id="check_sagemaker_monitor",
        python_callable=check_sagemaker_monitor,
        provide_context=True,
    )

    evaluate_task = BranchPythonOperator(
        task_id="evaluate_drift_action",
        python_callable=evaluate_drift_action,
        provide_context=True,
    )

    notify_task = PythonOperator(
        task_id="send_drift_notification",
        python_callable=send_drift_notification,
        provide_context=True,
    )

    retrain_task = PythonOperator(
        task_id="trigger_model_retrain",
        python_callable=trigger_model_retrain,
        provide_context=True,
    )

    no_drift_task = PythonOperator(
        task_id="log_no_drift",
        python_callable=log_no_drift,
        provide_context=True,
    )

    # All drift checks run in parallel, then evaluate
    [feature_drift_task, model_drift_task, sagemaker_task] >> evaluate_task

    # Branch to appropriate action
    evaluate_task >> [notify_task, retrain_task, no_drift_task]
