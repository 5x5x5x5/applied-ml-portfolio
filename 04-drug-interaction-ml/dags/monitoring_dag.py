"""Airflow DAG for continuous model monitoring.

Daily schedule that:
1. Runs drift detection (feature, prediction, label)
2. Checks model performance on recent labelled data
3. Generates monitoring reports
4. Alerts on anomalies and triggers retraining if needed
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DAG configuration
# ---------------------------------------------------------------------------

DEFAULT_ARGS = {
    "owner": "drug-interaction-ml",
    "depends_on_past": False,
    "email": ["ml-team@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=1),
}

S3_BUCKET = Variable.get("ml_artifacts_bucket", default_var="drug-interaction-ml-artifacts")
SNS_TOPIC_ARN = Variable.get("monitoring_sns_topic_arn", default_var="")
STEP_FUNCTIONS_ARN = Variable.get("training_step_functions_arn", default_var="")
ENDPOINT_NAME = Variable.get("sagemaker_endpoint_name", default_var="drug-interaction-endpoint")
MONITORING_SCHEDULE_NAME = Variable.get(
    "sagemaker_monitoring_schedule", default_var="drug-interaction-endpoint-monitoring-schedule"
)


# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------


def load_baseline_data(**context: Any) -> dict[str, str]:
    """Load training baseline feature and prediction distributions."""
    # In production, these paths would be stored as Variables or in a metadata DB
    baseline_features_path = f"s3://{S3_BUCKET}/baselines/latest/features.parquet"
    baseline_predictions_path = f"s3://{S3_BUCKET}/baselines/latest/predictions.parquet"
    baseline_labels_path = f"s3://{S3_BUCKET}/baselines/latest/labels.parquet"

    logger.info(
        "Baseline paths: features=%s, predictions=%s",
        baseline_features_path,
        baseline_predictions_path,
    )
    return {
        "baseline_features_path": baseline_features_path,
        "baseline_predictions_path": baseline_predictions_path,
        "baseline_labels_path": baseline_labels_path,
    }


def load_production_data(**context: Any) -> dict[str, str]:
    """Load recent production feature and prediction data."""
    from datetime import date

    end_date = date.fromisoformat(context["ds"])
    start_date = end_date - timedelta(days=1)

    production_features_path = (
        f"s3://{S3_BUCKET}/production/features/{start_date.isoformat()}/features.parquet"
    )
    production_predictions_path = (
        f"s3://{S3_BUCKET}/production/predictions/{start_date.isoformat()}/predictions.parquet"
    )
    production_labels_path = (
        f"s3://{S3_BUCKET}/production/labels/{start_date.isoformat()}/labels.parquet"
    )

    logger.info(
        "Production data paths: features=%s, predictions=%s",
        production_features_path,
        production_predictions_path,
    )
    return {
        "production_features_path": production_features_path,
        "production_predictions_path": production_predictions_path,
        "production_labels_path": production_labels_path,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }


def run_drift_detection(**context: Any) -> dict[str, Any]:
    """Execute comprehensive drift detection."""
    import numpy as np
    import pandas as pd

    from drug_interaction.monitoring.drift_detector import DriftDetector

    ti = context["ti"]
    baseline = ti.xcom_pull(task_ids="load_baseline_data")
    production = ti.xcom_pull(task_ids="load_production_data")

    # Load data
    baseline_features = pd.read_parquet(baseline["baseline_features_path"])
    production_features = pd.read_parquet(production["production_features_path"])

    baseline_predictions = None
    production_predictions = None
    baseline_labels = None
    production_labels = None

    try:
        pred_baseline = pd.read_parquet(baseline["baseline_predictions_path"])
        pred_production = pd.read_parquet(production["production_predictions_path"])
        baseline_predictions = pred_baseline["probability"].values.astype(np.float64)
        production_predictions = pred_production["probability"].values.astype(np.float64)
    except Exception:
        logger.warning("Could not load prediction data for drift analysis")

    try:
        lbl_baseline = pd.read_parquet(baseline["baseline_labels_path"])
        lbl_production = pd.read_parquet(production["production_labels_path"])
        baseline_labels = lbl_baseline["label"].values.astype(np.int64)
        production_labels = lbl_production["label"].values.astype(np.int64)
    except Exception:
        logger.warning("Could not load label data for drift analysis")

    detector = DriftDetector(psi_threshold=0.2, ks_alpha=0.05, max_drifted_features_pct=0.30)

    report = detector.generate_drift_report(
        baseline_features=baseline_features,
        current_features=production_features,
        baseline_predictions=baseline_predictions,
        current_predictions=production_predictions,
        baseline_labels=baseline_labels,
        current_labels=production_labels,
    )

    # Serialise report for XCom
    report_dict = report.model_dump()

    # Save report to S3
    report_path = f"s3://{S3_BUCKET}/monitoring/drift_reports/{context['ds']}/report.json"
    import boto3

    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=f"monitoring/drift_reports/{context['ds']}/report.json",
        Body=json.dumps(report_dict, indent=2, default=str),
        ContentType="application/json",
    )

    logger.info("Drift report: %s", report.summary)
    return report_dict


def check_model_performance(**context: Any) -> dict[str, Any]:
    """Check production model performance on recent labelled data."""
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    ti = context["ti"]
    production = ti.xcom_pull(task_ids="load_production_data")

    try:
        predictions_df = pd.read_parquet(production["production_predictions_path"])
        labels_df = pd.read_parquet(production["production_labels_path"])

        # Merge on common ID
        merged = predictions_df.merge(labels_df, on="prediction_id", how="inner")
        if len(merged) < 10:
            logger.warning("Too few labelled samples (%d) for performance check", len(merged))
            return {"sufficient_data": False, "sample_count": len(merged)}

        y_true = merged["label"]
        y_pred = merged["predicted_label"]

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_macro": float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "sample_count": len(merged),
            "sufficient_data": True,
        }
        logger.info("Production performance: %s", metrics)
        return metrics

    except Exception:
        logger.exception("Failed to compute production performance")
        return {"sufficient_data": False, "error": "Failed to load or process data"}


def check_sagemaker_monitor(**context: Any) -> dict[str, Any]:
    """Check SageMaker Model Monitor for violations."""
    from drug_interaction.monitoring.drift_detector import DriftDetector

    detector = DriftDetector()
    violations = detector.check_sagemaker_monitor_violations(
        monitoring_schedule_name=MONITORING_SCHEDULE_NAME,
        region="us-east-1",
    )

    return {
        "violation_count": len(violations),
        "violations": violations[:5],  # limit for XCom size
    }


def generate_monitoring_report(**context: Any) -> dict[str, Any]:
    """Compile all monitoring results into a single report."""
    ti = context["ti"]
    drift_result = ti.xcom_pull(task_ids="run_drift_detection")
    perf_result = ti.xcom_pull(task_ids="check_model_performance")
    monitor_result = ti.xcom_pull(task_ids="check_sagemaker_monitor")

    report = {
        "report_date": context["ds"],
        "drift_detection": {
            "overall_severity": drift_result.get("overall_severity", "unknown"),
            "features_drifted": drift_result.get("features_drifted", 0),
            "total_features": drift_result.get("total_features_analyzed", 0),
            "requires_retraining": drift_result.get("requires_retraining", False),
            "summary": drift_result.get("summary", ""),
        },
        "model_performance": perf_result,
        "sagemaker_monitor": {
            "violation_count": monitor_result.get("violation_count", 0),
        },
        "overall_status": "HEALTHY",
    }

    # Determine overall status
    if drift_result.get("requires_retraining"):
        report["overall_status"] = "RETRAINING_REQUIRED"
    elif drift_result.get("overall_severity") in ("high", "critical"):
        report["overall_status"] = "DEGRADED"
    elif monitor_result.get("violation_count", 0) > 0:
        report["overall_status"] = "MONITORING_VIOLATIONS"

    logger.info("Monitoring report status: %s", report["overall_status"])
    return report


def send_alerts(**context: Any) -> dict[str, Any]:
    """Process monitoring results and send alerts."""
    from drug_interaction.monitoring.alerting import AlertConfig, DriftAlertingSystem
    from drug_interaction.monitoring.drift_detector import DriftReport

    ti = context["ti"]
    drift_result = ti.xcom_pull(task_ids="run_drift_detection")
    perf_result = ti.xcom_pull(task_ids="check_model_performance")
    monitoring_report = ti.xcom_pull(task_ids="generate_monitoring_report")

    alert_config = AlertConfig(
        sns_topic_arn=SNS_TOPIC_ARN,
        step_functions_arn=STEP_FUNCTIONS_ARN,
        enable_auto_retrain=True,
        region="us-east-1",
    )
    alerting = DriftAlertingSystem(config=alert_config)

    # Process drift report
    drift_report = DriftReport(**drift_result)
    alerts = alerting.process_drift_report(drift_report)

    # Check performance degradation
    if perf_result.get("sufficient_data"):
        baseline_metrics = {"f1_macro": 0.85}  # Would come from model registry in production
        perf_alert = alerting.alert_on_performance_degradation(
            current_metrics=perf_result,
            baseline_metrics=baseline_metrics,
        )
        if perf_alert:
            alerts.append(perf_alert)

    alert_summary = {
        "total_alerts": len(alerts),
        "critical_alerts": sum(1 for a in alerts if a.level.value == "CRITICAL"),
        "warning_alerts": sum(1 for a in alerts if a.level.value == "WARNING"),
        "retraining_triggered": drift_report.requires_retraining,
    }
    logger.info("Alert summary: %s", alert_summary)
    return alert_summary


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="drug_interaction_monitoring",
    default_args=DEFAULT_ARGS,
    description="Daily model monitoring with drift detection and alerting",
    schedule="0 6 * * *",  # Every day at 06:00 UTC
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "drug-interaction", "monitoring"],
) as dag:
    load_baseline = PythonOperator(
        task_id="load_baseline_data",
        python_callable=load_baseline_data,
    )

    load_production = PythonOperator(
        task_id="load_production_data",
        python_callable=load_production_data,
    )

    drift_detection = PythonOperator(
        task_id="run_drift_detection",
        python_callable=run_drift_detection,
    )

    perf_check = PythonOperator(
        task_id="check_model_performance",
        python_callable=check_model_performance,
    )

    sm_monitor_check = PythonOperator(
        task_id="check_sagemaker_monitor",
        python_callable=check_sagemaker_monitor,
    )

    monitoring_report = PythonOperator(
        task_id="generate_monitoring_report",
        python_callable=generate_monitoring_report,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    alert_task = PythonOperator(
        task_id="send_alerts",
        python_callable=send_alerts,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # Dependencies
    [load_baseline, load_production] >> drift_detection
    load_production >> perf_check
    load_production >> sm_monitor_check
    [drift_detection, perf_check, sm_monitor_check] >> monitoring_report >> alert_task
