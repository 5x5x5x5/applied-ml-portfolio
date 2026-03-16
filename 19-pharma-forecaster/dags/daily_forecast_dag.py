"""Airflow DAG for daily pharmaceutical forecasting pipeline.

Runs the full forecast pipeline for all active drug demand series and
adverse event monitoring series. Includes data quality checks, model
training, forecast generation, and alerting.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

# --- DAG Configuration ---

DEFAULT_ARGS: dict[str, Any] = {
    "owner": "pharma-forecast-team",
    "depends_on_past": False,
    "email": ["pharma-forecast-alerts@company.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "execution_timeout": timedelta(hours=2),
}

# Series to forecast daily
DEMAND_SERIES_IDS: list[str] = [
    "acetaminophen_northeast",
    "acetaminophen_southeast",
    "acetaminophen_midwest",
    "acetaminophen_west",
    "amoxicillin_northeast",
    "amoxicillin_southeast",
    "amoxicillin_midwest",
    "amoxicillin_west",
    "lisinopril_national",
    "atorvastatin_national",
    "metformin_national",
    "albuterol_national",
]

ADVERSE_EVENT_SERIES_IDS: list[str] = [
    "ae_statins_cardiac",
    "ae_statins_hepatic",
    "ae_ssri_nervous_system",
    "ae_nsaid_gastrointestinal",
    "ae_immunotherapy_immune",
]


# --- Task Functions ---


def run_data_quality_check(series_ids: list[str], **kwargs: Any) -> dict[str, Any]:
    """Check data freshness and completeness for all series."""
    import structlog

    from pharma_forecast.pipeline.forecast_pipeline import PipelineConfig, S3DataStore

    logger = structlog.get_logger("airflow.data_quality")
    config = PipelineConfig()
    store = S3DataStore(bucket=config.s3_bucket, region=config.aws_region)

    results: dict[str, Any] = {"passed": [], "failed": [], "warnings": []}

    for series_id in series_ids:
        try:
            series = store.read_timeseries(f"{config.s3_input_prefix}{series_id}.csv")

            # Check data freshness: last observation should be within 2 days
            if hasattr(series.index, "max"):
                last_date = series.index.max()
                staleness = (datetime.now() - last_date).days
                if staleness > 2:
                    results["warnings"].append(
                        {"series_id": series_id, "issue": f"Data is {staleness} days stale"}
                    )

            # Check completeness
            missing_pct = series.isna().mean()
            if missing_pct > 0.1:
                results["warnings"].append(
                    {
                        "series_id": series_id,
                        "issue": f"Missing {missing_pct:.1%} of values",
                    }
                )

            # Check minimum length
            if len(series) < 180:
                results["failed"].append(
                    {
                        "series_id": series_id,
                        "issue": f"Only {len(series)} observations (need 180+)",
                    }
                )
            else:
                results["passed"].append(series_id)

        except Exception as exc:
            results["failed"].append({"series_id": series_id, "issue": str(exc)})

    logger.info(
        "data_quality_check_complete",
        passed=len(results["passed"]),
        warnings=len(results["warnings"]),
        failed=len(results["failed"]),
    )

    # Push valid series to XCom for downstream tasks
    kwargs["ti"].xcom_push(key="valid_series", value=results["passed"])
    return results


def run_demand_forecast(series_id: str, **kwargs: Any) -> dict[str, Any]:
    """Run the demand forecasting pipeline for a single series."""
    import structlog

    from pharma_forecast.pipeline.forecast_pipeline import ForecastPipeline, PipelineConfig

    logger = structlog.get_logger("airflow.demand_forecast")
    config = PipelineConfig(forecast_horizon=90)
    pipeline = ForecastPipeline(config=config)

    execution_date = kwargs.get("ds", datetime.now().strftime("%Y-%m-%d"))
    pipeline_id = f"daily_{execution_date}_{series_id}"

    result = pipeline.run(series_id=series_id, pipeline_id=pipeline_id)

    return {
        "series_id": series_id,
        "pipeline_id": pipeline_id,
        "stage": result.stage.value,
        "metrics": result.metrics,
        "n_alerts": len(result.alerts),
        "error": result.error,
    }


def run_adverse_event_forecast(series_id: str, **kwargs: Any) -> dict[str, Any]:
    """Run the adverse event forecasting pipeline for a single series."""
    import structlog

    from pharma_forecast.pipeline.forecast_pipeline import ForecastPipeline, PipelineConfig

    logger = structlog.get_logger("airflow.ae_forecast")
    config = PipelineConfig(forecast_horizon=90)
    pipeline = ForecastPipeline(config=config)

    execution_date = kwargs.get("ds", datetime.now().strftime("%Y-%m-%d"))
    pipeline_id = f"ae_daily_{execution_date}_{series_id}"

    result = pipeline.run(series_id=series_id, pipeline_id=pipeline_id)

    return {
        "series_id": series_id,
        "pipeline_id": pipeline_id,
        "stage": result.stage.value,
        "metrics": result.metrics,
        "n_alerts": len(result.alerts),
        "error": result.error,
    }


def run_monitor_check(**kwargs: Any) -> dict[str, Any]:
    """Run monitoring checks on all completed forecasts."""
    import structlog

    from pharma_forecast.monitoring.forecast_monitor import ForecastMonitor

    logger = structlog.get_logger("airflow.monitoring")
    monitor = ForecastMonitor()

    all_series = DEMAND_SERIES_IDS + ADVERSE_EVENT_SERIES_IDS
    retrain_candidates: list[str] = []

    for series_id in all_series:
        if monitor.should_retrain(series_id):
            retrain_candidates.append(series_id)

    active_alerts = monitor.get_active_alerts()

    summary = {
        "total_series": len(all_series),
        "retrain_candidates": retrain_candidates,
        "n_active_alerts": len(active_alerts),
        "critical_alerts": [a.series_id for a in active_alerts if a.severity.value == "critical"],
    }

    logger.info("monitoring_check_complete", summary=summary)

    # Push retrain candidates to XCom
    kwargs["ti"].xcom_push(key="retrain_candidates", value=retrain_candidates)
    return summary


def send_daily_report(**kwargs: Any) -> None:
    """Generate and send the daily forecast summary report."""
    import structlog

    logger = structlog.get_logger("airflow.daily_report")

    ti = kwargs["ti"]
    retrain_candidates = ti.xcom_pull(task_ids="monitor_forecasts", key="retrain_candidates") or []

    logger.info(
        "daily_report_generated",
        retrain_needed=len(retrain_candidates),
        execution_date=kwargs.get("ds"),
    )


# --- DAG Definition ---

with DAG(
    dag_id="pharma_daily_forecast",
    default_args=DEFAULT_ARGS,
    description="Daily pharmaceutical demand and adverse event forecasting",
    schedule_interval="0 6 * * *",  # Run at 6 AM UTC daily
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["pharma", "forecasting", "production"],
) as dag:
    start = DummyOperator(task_id="start")

    # Data quality check
    data_quality = PythonOperator(
        task_id="data_quality_check",
        python_callable=run_data_quality_check,
        op_kwargs={"series_ids": DEMAND_SERIES_IDS + ADVERSE_EVENT_SERIES_IDS},
        provide_context=True,
    )

    # Demand forecast tasks (one per series)
    demand_tasks = []
    for sid in DEMAND_SERIES_IDS:
        task = PythonOperator(
            task_id=f"demand_forecast_{sid}",
            python_callable=run_demand_forecast,
            op_kwargs={"series_id": sid},
            provide_context=True,
            pool="forecast_pool",
        )
        demand_tasks.append(task)

    # Adverse event forecast tasks
    ae_tasks = []
    for sid in ADVERSE_EVENT_SERIES_IDS:
        task = PythonOperator(
            task_id=f"ae_forecast_{sid}",
            python_callable=run_adverse_event_forecast,
            op_kwargs={"series_id": sid},
            provide_context=True,
            pool="forecast_pool",
        )
        ae_tasks.append(task)

    # Join point for all forecasts
    forecasts_complete = DummyOperator(
        task_id="forecasts_complete",
        trigger_rule=TriggerRule.ALL_DONE,  # Continue even if some forecasts fail
    )

    # Monitoring check
    monitor = PythonOperator(
        task_id="monitor_forecasts",
        python_callable=run_monitor_check,
        provide_context=True,
    )

    # Daily report
    report = PythonOperator(
        task_id="daily_report",
        python_callable=send_daily_report,
        provide_context=True,
    )

    end = DummyOperator(task_id="end")

    # Task dependencies
    start >> data_quality >> demand_tasks >> forecasts_complete
    start >> data_quality >> ae_tasks >> forecasts_complete
    forecasts_complete >> monitor >> report >> end
