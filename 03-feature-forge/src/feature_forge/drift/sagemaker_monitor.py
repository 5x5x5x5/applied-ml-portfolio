"""AWS SageMaker Model Monitor integration.

Creates monitoring schedules, defines baseline constraints, processes
monitoring results, and triggers retraining pipelines when drift is
detected. Uses boto3 SageMaker client for all AWS interactions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class MonitoringType(str, Enum):
    """Types of SageMaker monitoring."""

    DATA_QUALITY = "DataQuality"
    MODEL_QUALITY = "ModelQuality"
    MODEL_BIAS = "ModelBias"
    MODEL_EXPLAINABILITY = "ModelExplainability"


class ScheduleStatus(str, Enum):
    """Status of a monitoring schedule."""

    PENDING = "Pending"
    ACTIVE = "Scheduled"
    STOPPED = "Stopped"
    FAILED = "Failed"


@dataclass
class MonitoringConfig:
    """Configuration for a SageMaker monitoring schedule."""

    endpoint_name: str
    monitoring_type: MonitoringType = MonitoringType.DATA_QUALITY
    schedule_expression: str = "cron(0 * ? * * *)"  # Hourly
    instance_type: str = "ml.m5.xlarge"
    instance_count: int = 1
    volume_size_gb: int = 20
    max_runtime_seconds: int = 3600
    s3_output_path: str = ""
    role_arn: str = ""
    baseline_dataset_uri: str = ""
    baseline_statistics_uri: str = ""
    baseline_constraints_uri: str = ""
    # Alert configuration
    sns_topic_arn: str = ""
    cloudwatch_namespace: str = "FeatureForge/ModelMonitoring"
    drift_threshold: float = 0.2
    # Retraining
    retraining_pipeline_name: str = ""


@dataclass
class MonitoringResult:
    """Parsed result from a SageMaker monitoring execution."""

    execution_id: str
    schedule_name: str
    status: str
    start_time: datetime | None = None
    end_time: datetime | None = None
    violations: list[dict[str, Any]] = field(default_factory=list)
    constraint_violations_count: int = 0
    has_drift: bool = False
    drift_features: list[str] = field(default_factory=list)


class SageMakerModelMonitor:
    """Manage SageMaker Model Monitor schedules, baselines, and drift responses.

    Provides a high-level interface over the SageMaker Model Monitor API
    to create monitoring schedules, define baseline datasets and constraints,
    process monitoring execution results, and trigger retraining pipelines
    when drift is detected.
    """

    def __init__(
        self,
        config: MonitoringConfig,
        region_name: str = "us-east-1",
        session: boto3.Session | None = None,
    ) -> None:
        self._config = config
        self._session = session or boto3.Session(region_name=region_name)
        self._sm_client = self._session.client("sagemaker")
        self._s3_client = self._session.client("s3")
        self._cloudwatch = self._session.client("cloudwatch")
        self._sns_client = self._session.client("sns") if config.sns_topic_arn else None
        logger.info(
            "SageMakerModelMonitor initialised for endpoint=%s",
            config.endpoint_name,
        )

    # ------------------------------------------------------------------
    # Baseline creation
    # ------------------------------------------------------------------

    def create_data_quality_baseline(
        self,
        baseline_dataset_uri: str | None = None,
        output_s3_uri: str | None = None,
        job_name: str | None = None,
    ) -> str:
        """Create a data quality baseline job.

        Processes the baseline dataset to generate statistics and
        constraint suggestions that will be used for drift detection.
        Returns the baseline job name.
        """
        cfg = self._config
        effective_dataset = baseline_dataset_uri or cfg.baseline_dataset_uri
        effective_output = output_s3_uri or cfg.s3_output_path
        effective_job_name = (
            job_name or f"{cfg.endpoint_name}-baseline-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        )

        logger.info("Creating data quality baseline: %s", effective_job_name)

        try:
            self._sm_client.create_processing_job(
                ProcessingJobName=effective_job_name,
                ProcessingInputs=[
                    {
                        "InputName": "baseline_dataset",
                        "S3Input": {
                            "S3Uri": effective_dataset,
                            "LocalPath": "/opt/ml/processing/input",
                            "S3DataType": "S3Prefix",
                            "S3InputMode": "File",
                        },
                    }
                ],
                ProcessingOutputConfig={
                    "Outputs": [
                        {
                            "OutputName": "baseline_output",
                            "S3Output": {
                                "S3Uri": f"{effective_output}/baseline",
                                "LocalPath": "/opt/ml/processing/output",
                                "S3UploadMode": "EndOfJob",
                            },
                        }
                    ]
                },
                ProcessingResources={
                    "ClusterConfig": {
                        "InstanceCount": cfg.instance_count,
                        "InstanceType": cfg.instance_type,
                        "VolumeSizeInGB": cfg.volume_size_gb,
                    }
                },
                StoppingCondition={
                    "MaxRuntimeInSeconds": cfg.max_runtime_seconds,
                },
                AppSpecification={
                    "ImageUri": self._get_model_monitor_image_uri(),
                },
                RoleArn=cfg.role_arn,
                Environment={
                    "dataset_format": '{"csv": {"header": true}}',
                    "dataset_source": "/opt/ml/processing/input",
                    "output_path": "/opt/ml/processing/output",
                    "publish_cloudwatch_metrics": "Disabled",
                },
            )
            logger.info("Baseline job %s created successfully", effective_job_name)
            return effective_job_name

        except ClientError as exc:
            logger.error("Failed to create baseline job: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Monitoring schedule
    # ------------------------------------------------------------------

    def create_monitoring_schedule(
        self,
        schedule_name: str | None = None,
    ) -> str:
        """Create a SageMaker monitoring schedule for the endpoint.

        Configures data quality monitoring with the specified schedule
        expression, baseline constraints, and output location.
        """
        cfg = self._config
        effective_name = schedule_name or f"{cfg.endpoint_name}-monitor"

        logger.info("Creating monitoring schedule: %s", effective_name)

        monitoring_output = {
            "MonitoringOutputs": [
                {
                    "S3Output": {
                        "S3Uri": f"{cfg.s3_output_path}/monitoring-results",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "Continuous",
                    }
                }
            ]
        }

        baseline_config = {}
        if cfg.baseline_statistics_uri:
            baseline_config["StatisticsResource"] = {"S3Uri": cfg.baseline_statistics_uri}
        if cfg.baseline_constraints_uri:
            baseline_config["ConstraintsResource"] = {"S3Uri": cfg.baseline_constraints_uri}

        try:
            self._sm_client.create_monitoring_schedule(
                MonitoringScheduleName=effective_name,
                MonitoringScheduleConfig={
                    "ScheduleConfig": {
                        "ScheduleExpression": cfg.schedule_expression,
                    },
                    "MonitoringJobDefinition": {
                        "BaselineConfig": baseline_config,
                        "MonitoringInputs": [
                            {
                                "EndpointInput": {
                                    "EndpointName": cfg.endpoint_name,
                                    "LocalPath": "/opt/ml/processing/input",
                                    "S3DataDistributionType": "FullyReplicated",
                                    "S3InputMode": "File",
                                }
                            }
                        ],
                        "MonitoringOutputConfig": monitoring_output,
                        "MonitoringResources": {
                            "ClusterConfig": {
                                "InstanceCount": cfg.instance_count,
                                "InstanceType": cfg.instance_type,
                                "VolumeSizeInGB": cfg.volume_size_gb,
                            }
                        },
                        "MonitoringAppSpecification": {
                            "ImageUri": self._get_model_monitor_image_uri(),
                        },
                        "StoppingCondition": {
                            "MaxRuntimeInSeconds": cfg.max_runtime_seconds,
                        },
                        "RoleArn": cfg.role_arn,
                    },
                },
            )
            logger.info("Monitoring schedule %s created", effective_name)
            return effective_name

        except ClientError as exc:
            logger.error("Failed to create monitoring schedule: %s", exc)
            raise

    def describe_monitoring_schedule(self, schedule_name: str) -> dict[str, Any]:
        """Describe a monitoring schedule and its status."""
        try:
            response = self._sm_client.describe_monitoring_schedule(
                MonitoringScheduleName=schedule_name
            )
            return {
                "name": response["MonitoringScheduleName"],
                "status": response["MonitoringScheduleStatus"],
                "creation_time": response.get("CreationTime"),
                "last_modified": response.get("LastModifiedTime"),
                "last_execution": response.get("LastMonitoringExecutionSummary", {}),
            }
        except ClientError as exc:
            logger.error("Failed to describe schedule %s: %s", schedule_name, exc)
            raise

    def stop_monitoring_schedule(self, schedule_name: str) -> None:
        """Stop a running monitoring schedule."""
        try:
            self._sm_client.stop_monitoring_schedule(MonitoringScheduleName=schedule_name)
            logger.info("Stopped monitoring schedule: %s", schedule_name)
        except ClientError as exc:
            logger.error("Failed to stop schedule %s: %s", schedule_name, exc)
            raise

    def delete_monitoring_schedule(self, schedule_name: str) -> None:
        """Delete a monitoring schedule."""
        try:
            self._sm_client.delete_monitoring_schedule(MonitoringScheduleName=schedule_name)
            logger.info("Deleted monitoring schedule: %s", schedule_name)
        except ClientError as exc:
            logger.error("Failed to delete schedule %s: %s", schedule_name, exc)
            raise

    # ------------------------------------------------------------------
    # Processing monitoring results
    # ------------------------------------------------------------------

    def list_monitoring_executions(
        self,
        schedule_name: str,
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        """List recent monitoring executions for a schedule."""
        try:
            response = self._sm_client.list_monitoring_executions(
                MonitoringScheduleName=schedule_name,
                SortBy="CreationTime",
                SortOrder="Descending",
                MaxResults=max_results,
            )
            return [
                {
                    "execution_status": ex["MonitoringExecutionStatus"],
                    "scheduled_time": ex.get("ScheduledTime"),
                    "creation_time": ex.get("CreationTime"),
                    "last_modified": ex.get("LastModifiedTime"),
                    "failure_reason": ex.get("FailureReason"),
                }
                for ex in response.get("MonitoringExecutionSummaries", [])
            ]
        except ClientError as exc:
            logger.error("Failed to list executions: %s", exc)
            raise

    def process_monitoring_results(
        self,
        schedule_name: str,
        execution_index: int = 0,
    ) -> MonitoringResult:
        """Process the results of a monitoring execution.

        Downloads the constraint violations report from S3, parses it,
        and returns a structured MonitoringResult with drift information.
        """
        executions = self.list_monitoring_executions(schedule_name, max_results=execution_index + 1)
        if not executions or len(executions) <= execution_index:
            raise ValueError(f"No execution found at index {execution_index}")

        execution = executions[execution_index]
        cfg = self._config

        # Download violations report from S3
        violations_uri = f"{cfg.s3_output_path}/monitoring-results/constraint_violations.json"
        violations = self._download_s3_json(violations_uri)

        parsed_violations = []
        drift_features: list[str] = []

        for violation in violations.get("violations", []):
            feature_name = violation.get("feature_name", "")
            metric_name = violation.get("constraint_check_type", "")
            description = violation.get("description", "")

            parsed_violations.append(
                {
                    "feature_name": feature_name,
                    "metric_name": metric_name,
                    "description": description,
                }
            )

            if "drift" in metric_name.lower() or "distribution" in description.lower():
                drift_features.append(feature_name)

        has_drift = len(drift_features) > 0

        result = MonitoringResult(
            execution_id=f"{schedule_name}-{execution_index}",
            schedule_name=schedule_name,
            status=execution["execution_status"],
            violations=parsed_violations,
            constraint_violations_count=len(parsed_violations),
            has_drift=has_drift,
            drift_features=drift_features,
        )

        logger.info(
            "Processed monitoring result: %d violations, drift=%s",
            len(parsed_violations),
            has_drift,
        )
        return result

    # ------------------------------------------------------------------
    # Alerting
    # ------------------------------------------------------------------

    def send_drift_alert(self, result: MonitoringResult) -> None:
        """Send an SNS alert when drift is detected."""
        if not self._sns_client or not self._config.sns_topic_arn:
            logger.warning("SNS not configured, skipping drift alert")
            return

        message = {
            "source": "FeatureForge",
            "endpoint": self._config.endpoint_name,
            "schedule": result.schedule_name,
            "violations_count": result.constraint_violations_count,
            "drifted_features": result.drift_features,
            "detected_at": datetime.utcnow().isoformat(),
        }

        try:
            self._sns_client.publish(
                TopicArn=self._config.sns_topic_arn,
                Subject=f"[FeatureForge] Drift detected on {self._config.endpoint_name}",
                Message=json.dumps(message, indent=2),
                MessageAttributes={
                    "severity": {
                        "DataType": "String",
                        "StringValue": "HIGH" if len(result.drift_features) > 3 else "MEDIUM",
                    }
                },
            )
            logger.info("Drift alert sent to SNS topic")
        except ClientError as exc:
            logger.error("Failed to send drift alert: %s", exc)

    def publish_cloudwatch_metrics(self, result: MonitoringResult) -> None:
        """Publish monitoring metrics to CloudWatch."""
        namespace = self._config.cloudwatch_namespace
        dimensions = [
            {"Name": "Endpoint", "Value": self._config.endpoint_name},
            {"Name": "Schedule", "Value": result.schedule_name},
        ]

        metric_data = [
            {
                "MetricName": "ConstraintViolations",
                "Dimensions": dimensions,
                "Value": float(result.constraint_violations_count),
                "Unit": "Count",
            },
            {
                "MetricName": "DriftDetected",
                "Dimensions": dimensions,
                "Value": 1.0 if result.has_drift else 0.0,
                "Unit": "None",
            },
            {
                "MetricName": "DriftedFeaturesCount",
                "Dimensions": dimensions,
                "Value": float(len(result.drift_features)),
                "Unit": "Count",
            },
        ]

        try:
            self._cloudwatch.put_metric_data(
                Namespace=namespace,
                MetricData=metric_data,
            )
            logger.info("Published %d CloudWatch metrics", len(metric_data))
        except ClientError as exc:
            logger.error("Failed to publish CloudWatch metrics: %s", exc)

    # ------------------------------------------------------------------
    # Retraining trigger
    # ------------------------------------------------------------------

    def trigger_retraining(
        self,
        result: MonitoringResult,
        pipeline_parameters: dict[str, str] | None = None,
    ) -> str | None:
        """Trigger a SageMaker pipeline for model retraining.

        Only triggers if drift is detected and a retraining pipeline
        is configured. Returns the pipeline execution ARN or None.
        """
        cfg = self._config
        if not cfg.retraining_pipeline_name:
            logger.warning("No retraining pipeline configured")
            return None

        if not result.has_drift:
            logger.info("No drift detected, skipping retraining trigger")
            return None

        logger.info(
            "Triggering retraining pipeline: %s (drift in %d features)",
            cfg.retraining_pipeline_name,
            len(result.drift_features),
        )

        params = [
            {"Name": "EndpointName", "Value": cfg.endpoint_name},
            {"Name": "DriftFeatures", "Value": ",".join(result.drift_features)},
            {"Name": "TriggerReason", "Value": "drift_detected"},
            {"Name": "TriggerTimestamp", "Value": datetime.utcnow().isoformat()},
        ]
        if pipeline_parameters:
            for key, value in pipeline_parameters.items():
                params.append({"Name": key, "Value": value})

        try:
            response = self._sm_client.start_pipeline_execution(
                PipelineName=cfg.retraining_pipeline_name,
                PipelineExecutionDisplayName=f"drift-retrain-{datetime.utcnow().strftime('%Y%m%d%H%M')}",
                PipelineParameters=params,
                PipelineExecutionDescription=(
                    f"Automated retraining triggered by drift detection. "
                    f"Drifted features: {', '.join(result.drift_features[:5])}"
                ),
            )
            execution_arn = response["PipelineExecutionArn"]
            logger.info("Retraining pipeline started: %s", execution_arn)
            return execution_arn

        except ClientError as exc:
            logger.error("Failed to trigger retraining: %s", exc)
            raise

    # ------------------------------------------------------------------
    # End-to-end monitoring cycle
    # ------------------------------------------------------------------

    def run_monitoring_cycle(self, schedule_name: str) -> MonitoringResult:
        """Execute a full monitoring cycle: process results, alert, and retrain.

        1. Process the latest monitoring execution results
        2. Publish CloudWatch metrics
        3. Send SNS alert if drift detected
        4. Trigger retraining if drift exceeds threshold
        """
        logger.info("Running monitoring cycle for %s", schedule_name)

        result = self.process_monitoring_results(schedule_name)
        self.publish_cloudwatch_metrics(result)

        if result.has_drift:
            self.send_drift_alert(result)

            if result.constraint_violations_count >= 3:
                self.trigger_retraining(result)
            else:
                logger.info(
                    "Drift detected but below retraining threshold (%d violations)",
                    result.constraint_violations_count,
                )

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_model_monitor_image_uri(self) -> str:
        """Get the SageMaker Model Monitor container image URI for the region."""
        region = self._session.region_name or "us-east-1"
        account_map = {
            "us-east-1": "156813124566",
            "us-east-2": "777275614652",
            "us-west-1": "890145073186",
            "us-west-2": "159807026194",
            "eu-west-1": "468650794304",
            "eu-central-1": "048819808253",
            "ap-southeast-1": "245545462676",
            "ap-northeast-1": "574779866223",
        }
        account = account_map.get(region, "156813124566")
        return f"{account}.dkr.ecr.{region}.amazonaws.com/sagemaker-model-monitor-analyzer"

    def _download_s3_json(self, s3_uri: str) -> dict[str, Any]:
        """Download and parse a JSON file from S3."""
        if not s3_uri.startswith("s3://"):
            logger.warning("Invalid S3 URI: %s", s3_uri)
            return {}

        parts = s3_uri.replace("s3://", "").split("/", 1)
        if len(parts) != 2:
            return {}

        bucket, key = parts
        try:
            response = self._s3_client.get_object(Bucket=bucket, Key=key)
            content = response["Body"].read().decode("utf-8")
            return json.loads(content)
        except ClientError as exc:
            logger.error("Failed to download %s: %s", s3_uri, exc)
            return {}
        except json.JSONDecodeError:
            logger.error("Invalid JSON in %s", s3_uri)
            return {}
