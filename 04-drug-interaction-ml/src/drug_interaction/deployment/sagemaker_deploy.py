"""SageMaker deployment utilities for drug interaction models.

Manages endpoint creation, A/B testing configuration, auto-scaling
policies, and model monitoring baselines.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import boto3
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class EndpointConfig(BaseModel):
    """Configuration for a SageMaker endpoint."""

    endpoint_name: str
    model_name: str
    instance_type: str = "ml.m5.xlarge"
    initial_instance_count: int = 1
    variant_name: str = "primary"
    initial_weight: float = 1.0
    data_capture_percentage: int = 100
    data_capture_s3_uri: str = ""


class ABTestConfig(BaseModel):
    """Configuration for A/B testing between model variants."""

    endpoint_name: str
    variant_a_name: str = "champion"
    variant_b_name: str = "challenger"
    variant_a_model: str
    variant_b_model: str
    variant_a_weight: float = Field(default=0.9, ge=0.0, le=1.0)
    variant_b_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    instance_type: str = "ml.m5.xlarge"
    instance_count: int = 1


class AutoScalingConfig(BaseModel):
    """Auto-scaling policy configuration."""

    endpoint_name: str
    variant_name: str = "primary"
    min_capacity: int = 1
    max_capacity: int = 4
    target_invocations_per_instance: int = 1000
    scale_in_cooldown: int = 300
    scale_out_cooldown: int = 60


class MonitoringBaselineConfig(BaseModel):
    """Configuration for SageMaker Model Monitor baseline."""

    endpoint_name: str
    baseline_dataset_s3_uri: str
    baseline_results_s3_uri: str
    role_arn: str
    instance_type: str = "ml.m5.xlarge"
    schedule_expression: str = "cron(0 * ? * * *)"  # hourly


# ---------------------------------------------------------------------------
# Deployer
# ---------------------------------------------------------------------------


@dataclass
class SageMakerDeployer:
    """Manage SageMaker model deployments.

    Parameters
    ----------
    region : str
        AWS region name.
    execution_role_arn : str
        IAM role ARN for SageMaker operations.
    """

    region: str = "us-east-1"
    execution_role_arn: str = ""
    _sm_client: Any = field(default=None, init=False, repr=False)
    _sm_runtime: Any = field(default=None, init=False, repr=False)
    _autoscaling: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._sm_client = boto3.client("sagemaker", region_name=self.region)
        self._sm_runtime = boto3.client("sagemaker-runtime", region_name=self.region)
        self._autoscaling = boto3.client("application-autoscaling", region_name=self.region)

    # -- Model creation -----------------------------------------------------

    def create_model(
        self,
        model_name: str,
        image_uri: str,
        model_data_url: str,
        *,
        environment: dict[str, str] | None = None,
    ) -> str:
        """Create a SageMaker model.

        Parameters
        ----------
        model_name : str
            Name for the SageMaker model.
        image_uri : str
            Docker image URI for inference.
        model_data_url : str
            S3 path to the model artifact (model.tar.gz).
        environment : dict, optional
            Environment variables for the container.

        Returns
        -------
        str
            Model ARN.
        """
        container = {
            "Image": image_uri,
            "ModelDataUrl": model_data_url,
            "Environment": environment or {},
        }

        response = self._sm_client.create_model(
            ModelName=model_name,
            PrimaryContainer=container,
            ExecutionRoleArn=self.execution_role_arn,
            Tags=[
                {"Key": "Project", "Value": "DrugInteractionML"},
                {"Key": "CreatedAt", "Value": datetime.now(tz=UTC).isoformat()},
            ],
        )
        logger.info("Created SageMaker model: %s", model_name)
        return response["ModelArn"]

    # -- Endpoint creation --------------------------------------------------

    def create_endpoint(self, config: EndpointConfig) -> str:
        """Create a SageMaker endpoint with data capture.

        Parameters
        ----------
        config : EndpointConfig
            Endpoint configuration.

        Returns
        -------
        str
            Endpoint ARN.
        """
        endpoint_config_name = f"{config.endpoint_name}-config-{int(time.time())}"

        # Data capture configuration
        data_capture_config = {}
        if config.data_capture_s3_uri:
            data_capture_config = {
                "EnableCapture": True,
                "InitialSamplingPercentage": config.data_capture_percentage,
                "DestinationS3Uri": config.data_capture_s3_uri,
                "CaptureOptions": [
                    {"CaptureMode": "Input"},
                    {"CaptureMode": "Output"},
                ],
                "CaptureContentTypeHeader": {
                    "CsvContentTypes": ["text/csv"],
                    "JsonContentTypes": ["application/json"],
                },
            }

        # Create endpoint config
        production_variant = {
            "VariantName": config.variant_name,
            "ModelName": config.model_name,
            "InstanceType": config.instance_type,
            "InitialInstanceCount": config.initial_instance_count,
            "InitialVariantWeight": config.initial_weight,
        }

        create_config_kwargs: dict[str, Any] = {
            "EndpointConfigName": endpoint_config_name,
            "ProductionVariants": [production_variant],
        }
        if data_capture_config:
            create_config_kwargs["DataCaptureConfig"] = data_capture_config

        self._sm_client.create_endpoint_config(**create_config_kwargs)
        logger.info("Created endpoint config: %s", endpoint_config_name)

        # Create or update endpoint
        try:
            response = self._sm_client.create_endpoint(
                EndpointName=config.endpoint_name,
                EndpointConfigName=endpoint_config_name,
                Tags=[
                    {"Key": "Project", "Value": "DrugInteractionML"},
                ],
            )
            logger.info("Created endpoint: %s", config.endpoint_name)
        except self._sm_client.exceptions.ClientError as e:
            if "already exists" in str(e).lower():
                response = self._sm_client.update_endpoint(
                    EndpointName=config.endpoint_name,
                    EndpointConfigName=endpoint_config_name,
                )
                logger.info("Updated endpoint: %s", config.endpoint_name)
            else:
                raise

        return response.get("EndpointArn", config.endpoint_name)

    # -- A/B Testing --------------------------------------------------------

    def create_ab_test_endpoint(self, config: ABTestConfig) -> str:
        """Create an endpoint with two model variants for A/B testing.

        Parameters
        ----------
        config : ABTestConfig
            A/B test configuration.

        Returns
        -------
        str
            Endpoint name.
        """
        endpoint_config_name = f"{config.endpoint_name}-ab-{int(time.time())}"

        variants = [
            {
                "VariantName": config.variant_a_name,
                "ModelName": config.variant_a_model,
                "InstanceType": config.instance_type,
                "InitialInstanceCount": config.instance_count,
                "InitialVariantWeight": config.variant_a_weight,
            },
            {
                "VariantName": config.variant_b_name,
                "ModelName": config.variant_b_model,
                "InstanceType": config.instance_type,
                "InitialInstanceCount": config.instance_count,
                "InitialVariantWeight": config.variant_b_weight,
            },
        ]

        self._sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=variants,
        )

        try:
            self._sm_client.create_endpoint(
                EndpointName=config.endpoint_name,
                EndpointConfigName=endpoint_config_name,
            )
        except self._sm_client.exceptions.ClientError as e:
            if "already exists" in str(e).lower():
                self._sm_client.update_endpoint(
                    EndpointName=config.endpoint_name,
                    EndpointConfigName=endpoint_config_name,
                )
            else:
                raise

        logger.info(
            "Created A/B test endpoint: %s (A=%s@%.0f%%, B=%s@%.0f%%)",
            config.endpoint_name,
            config.variant_a_name,
            config.variant_a_weight * 100,
            config.variant_b_name,
            config.variant_b_weight * 100,
        )
        return config.endpoint_name

    def update_variant_weights(
        self,
        endpoint_name: str,
        weights: dict[str, float],
    ) -> None:
        """Update traffic distribution across variants.

        Parameters
        ----------
        endpoint_name : str
            Name of the endpoint.
        weights : dict
            Mapping of variant name to desired weight.
        """
        desired_weights = [
            {"VariantName": name, "DesiredWeight": weight} for name, weight in weights.items()
        ]
        self._sm_client.update_endpoint_weights_and_capacities(
            EndpointName=endpoint_name,
            DesiredWeightsAndCapacities=desired_weights,
        )
        logger.info("Updated variant weights for %s: %s", endpoint_name, weights)

    # -- Auto-scaling -------------------------------------------------------

    def configure_autoscaling(self, config: AutoScalingConfig) -> None:
        """Set up auto-scaling for an endpoint variant.

        Uses a target-tracking scaling policy based on
        invocations per instance.

        Parameters
        ----------
        config : AutoScalingConfig
            Scaling configuration.
        """
        resource_id = f"endpoint/{config.endpoint_name}/variant/{config.variant_name}"

        # Register scalable target
        self._autoscaling.register_scalable_target(
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            MinCapacity=config.min_capacity,
            MaxCapacity=config.max_capacity,
        )

        # Target tracking policy
        self._autoscaling.put_scaling_policy(
            PolicyName=f"{config.endpoint_name}-scaling-policy",
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            PolicyType="TargetTrackingScaling",
            TargetTrackingScalingPolicyConfiguration={
                "TargetValue": float(config.target_invocations_per_instance),
                "CustomizedMetricSpecification": {
                    "MetricName": "InvocationsPerInstance",
                    "Namespace": "AWS/SageMaker",
                    "Dimensions": [
                        {"Name": "EndpointName", "Value": config.endpoint_name},
                        {"Name": "VariantName", "Value": config.variant_name},
                    ],
                    "Statistic": "Sum",
                },
                "ScaleInCooldown": config.scale_in_cooldown,
                "ScaleOutCooldown": config.scale_out_cooldown,
            },
        )
        logger.info(
            "Configured auto-scaling for %s/%s: min=%d, max=%d, target=%d",
            config.endpoint_name,
            config.variant_name,
            config.min_capacity,
            config.max_capacity,
            config.target_invocations_per_instance,
        )

    # -- Model Monitoring Baseline ------------------------------------------

    def create_monitoring_baseline(self, config: MonitoringBaselineConfig) -> str:
        """Create a SageMaker Model Monitor baseline job.

        Parameters
        ----------
        config : MonitoringBaselineConfig
            Baseline configuration.

        Returns
        -------
        str
            Baseline job name.
        """
        job_name = f"drug-interaction-baseline-{int(time.time())}"

        self._sm_client.create_processing_job(
            ProcessingJobName=job_name,
            ProcessingResources={
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": config.instance_type,
                    "VolumeSizeInGB": 20,
                }
            },
            AppSpecification={
                "ImageUri": self._get_model_monitor_image_uri(),
            },
            RoleArn=config.role_arn,
            ProcessingInputs=[
                {
                    "InputName": "baseline_dataset",
                    "S3Input": {
                        "S3Uri": config.baseline_dataset_s3_uri,
                        "LocalPath": "/opt/ml/processing/input",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                    },
                }
            ],
            ProcessingOutputConfig={
                "Outputs": [
                    {
                        "OutputName": "baseline_results",
                        "S3Output": {
                            "S3Uri": config.baseline_results_s3_uri,
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob",
                        },
                    }
                ]
            },
            Environment={
                "dataset_format": '{"csv": {"header": true}}',
                "dataset_source": "/opt/ml/processing/input",
                "output_path": "/opt/ml/processing/output",
                "publish_cloudwatch_metrics": "Enabled",
            },
        )
        logger.info("Created monitoring baseline job: %s", job_name)
        return job_name

    def create_monitoring_schedule(
        self,
        config: MonitoringBaselineConfig,
        baseline_constraints_uri: str,
        baseline_statistics_uri: str,
    ) -> str:
        """Create a recurring monitoring schedule.

        Parameters
        ----------
        config : MonitoringBaselineConfig
            Base monitoring configuration.
        baseline_constraints_uri : str
            S3 URI of the constraints.json from the baseline job.
        baseline_statistics_uri : str
            S3 URI of the statistics.json from the baseline job.

        Returns
        -------
        str
            Monitoring schedule name.
        """
        schedule_name = f"{config.endpoint_name}-monitoring-schedule"

        self._sm_client.create_monitoring_schedule(
            MonitoringScheduleName=schedule_name,
            MonitoringScheduleConfig={
                "ScheduleConfig": {
                    "ScheduleExpression": config.schedule_expression,
                },
                "MonitoringJobDefinition": {
                    "BaselineConfig": {
                        "ConstraintsResource": {"S3Uri": baseline_constraints_uri},
                        "StatisticsResource": {"S3Uri": baseline_statistics_uri},
                    },
                    "MonitoringInputs": [
                        {
                            "EndpointInput": {
                                "EndpointName": config.endpoint_name,
                                "LocalPath": "/opt/ml/processing/input",
                            }
                        }
                    ],
                    "MonitoringOutputConfig": {
                        "MonitoringOutputs": [
                            {
                                "S3Output": {
                                    "S3Uri": f"{config.baseline_results_s3_uri}/monitoring",
                                    "LocalPath": "/opt/ml/processing/output",
                                    "S3UploadMode": "Continuous",
                                }
                            }
                        ]
                    },
                    "MonitoringResources": {
                        "ClusterConfig": {
                            "InstanceCount": 1,
                            "InstanceType": config.instance_type,
                            "VolumeSizeInGB": 20,
                        }
                    },
                    "MonitoringAppSpecification": {
                        "ImageUri": self._get_model_monitor_image_uri(),
                    },
                    "RoleArn": config.role_arn,
                },
            },
        )
        logger.info("Created monitoring schedule: %s", schedule_name)
        return schedule_name

    # -- Utilities ----------------------------------------------------------

    def wait_for_endpoint(
        self,
        endpoint_name: str,
        *,
        timeout_minutes: int = 30,
        poll_interval: int = 30,
    ) -> str:
        """Wait for an endpoint to become InService.

        Returns
        -------
        str
            Final endpoint status.
        """
        deadline = time.time() + timeout_minutes * 60
        while time.time() < deadline:
            response = self._sm_client.describe_endpoint(EndpointName=endpoint_name)
            status = response["EndpointStatus"]
            if status == "InService":
                logger.info("Endpoint %s is InService", endpoint_name)
                return status
            if status in ("Failed", "RollbackFailed"):
                reason = response.get("FailureReason", "Unknown")
                raise RuntimeError(f"Endpoint {endpoint_name} failed: {reason}")
            logger.info("Endpoint %s status: %s. Waiting...", endpoint_name, status)
            time.sleep(poll_interval)
        raise TimeoutError(
            f"Endpoint {endpoint_name} did not become InService within {timeout_minutes} minutes"
        )

    def delete_endpoint(self, endpoint_name: str) -> None:
        """Delete a SageMaker endpoint and its configuration."""
        try:
            desc = self._sm_client.describe_endpoint(EndpointName=endpoint_name)
            config_name = desc["EndpointConfigName"]
            self._sm_client.delete_endpoint(EndpointName=endpoint_name)
            self._sm_client.delete_endpoint_config(EndpointConfigName=config_name)
            logger.info("Deleted endpoint %s and config %s", endpoint_name, config_name)
        except self._sm_client.exceptions.ClientError:
            logger.warning("Endpoint %s not found for deletion", endpoint_name)

    def invoke_endpoint(
        self,
        endpoint_name: str,
        payload: str,
        content_type: str = "text/csv",
        target_variant: str | None = None,
    ) -> dict[str, Any]:
        """Invoke a SageMaker endpoint.

        Parameters
        ----------
        endpoint_name : str
            Endpoint to invoke.
        payload : str
            Request body.
        content_type : str
            Content type of the payload.
        target_variant : str, optional
            Specific variant to target (for A/B testing).

        Returns
        -------
        dict
            Response body and invoked variant.
        """
        invoke_kwargs: dict[str, Any] = {
            "EndpointName": endpoint_name,
            "Body": payload,
            "ContentType": content_type,
        }
        if target_variant:
            invoke_kwargs["TargetVariant"] = target_variant

        response = self._sm_runtime.invoke_endpoint(**invoke_kwargs)
        body = response["Body"].read().decode("utf-8")

        return {
            "body": json.loads(body) if content_type == "application/json" else body,
            "invoked_variant": response.get("InvokedProductionVariant", ""),
            "content_type": response["ContentType"],
        }

    def _get_model_monitor_image_uri(self) -> str:
        """Return the SageMaker Model Monitor built-in image URI."""
        region = self.region
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
