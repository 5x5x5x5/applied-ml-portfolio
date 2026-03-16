"""AWS Step Functions state machine definition for the training workflow.

Orchestrates: data extraction -> feature engineering -> model training ->
evaluation -> conditional deployment -> monitoring setup. Includes error
handling, retry logic, and parallel branches.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import boto3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State Machine Definition
# ---------------------------------------------------------------------------


def build_training_workflow_definition(
    *,
    extraction_lambda_arn: str,
    feature_eng_lambda_arn: str,
    training_job_lambda_arn: str,
    evaluation_lambda_arn: str,
    deployment_lambda_arn: str,
    monitoring_lambda_arn: str,
    rollback_lambda_arn: str,
    sns_topic_arn: str,
    performance_threshold: float = 0.80,
) -> dict[str, Any]:
    """Build the Step Functions state machine definition as a Python dict.

    Parameters
    ----------
    extraction_lambda_arn : str
        ARN for the data extraction Lambda function.
    feature_eng_lambda_arn : str
        ARN for the feature engineering Lambda.
    training_job_lambda_arn : str
        ARN for the SageMaker training job orchestrator Lambda.
    evaluation_lambda_arn : str
        ARN for the model evaluation Lambda.
    deployment_lambda_arn : str
        ARN for the SageMaker deployment Lambda.
    monitoring_lambda_arn : str
        ARN for the monitoring setup Lambda.
    rollback_lambda_arn : str
        ARN for the deployment rollback Lambda.
    sns_topic_arn : str
        SNS topic ARN for notifications.
    performance_threshold : float
        Minimum F1 score required for deployment.

    Returns
    -------
    dict
        Complete Step Functions state machine definition.
    """
    return {
        "Comment": "DrugInteractionML Training and Deployment Pipeline",
        "StartAt": "DataExtraction",
        "States": {
            # ── Stage 1: Parallel Data Extraction ─────────────────────
            "DataExtraction": {
                "Type": "Parallel",
                "Comment": "Extract molecular and patient features in parallel",
                "Branches": [
                    {
                        "StartAt": "ExtractMolecularFeatures",
                        "States": {
                            "ExtractMolecularFeatures": {
                                "Type": "Task",
                                "Resource": "arn:aws:states:::lambda:invoke",
                                "Parameters": {
                                    "FunctionName": extraction_lambda_arn,
                                    "Payload": {
                                        "extraction_type": "molecular",
                                        "input.$": "$.molecular_config",
                                    },
                                },
                                "ResultPath": "$.molecular_features",
                                "Retry": [
                                    {
                                        "ErrorEquals": [
                                            "Lambda.ServiceException",
                                            "Lambda.TooManyRequestsException",
                                            "States.TaskFailed",
                                        ],
                                        "IntervalSeconds": 30,
                                        "MaxAttempts": 3,
                                        "BackoffRate": 2.0,
                                    }
                                ],
                                "Catch": [
                                    {
                                        "ErrorEquals": ["States.ALL"],
                                        "ResultPath": "$.error",
                                        "Next": "MolecularExtractionFailed",
                                    }
                                ],
                                "End": True,
                            },
                            "MolecularExtractionFailed": {
                                "Type": "Task",
                                "Resource": "arn:aws:states:::sns:publish",
                                "Parameters": {
                                    "TopicArn": sns_topic_arn,
                                    "Subject": "DrugInteractionML: Molecular Extraction Failed",
                                    "Message.$": "States.Format('Molecular feature extraction failed: {}', $.error.Cause)",
                                },
                                "End": True,
                            },
                        },
                    },
                    {
                        "StartAt": "ExtractSnowflakeFeatures",
                        "States": {
                            "ExtractSnowflakeFeatures": {
                                "Type": "Task",
                                "Resource": "arn:aws:states:::lambda:invoke",
                                "Parameters": {
                                    "FunctionName": extraction_lambda_arn,
                                    "Payload": {
                                        "extraction_type": "snowflake",
                                        "input.$": "$.snowflake_config",
                                    },
                                },
                                "ResultPath": "$.snowflake_features",
                                "Retry": [
                                    {
                                        "ErrorEquals": [
                                            "Lambda.ServiceException",
                                            "States.TaskFailed",
                                        ],
                                        "IntervalSeconds": 60,
                                        "MaxAttempts": 2,
                                        "BackoffRate": 2.0,
                                    }
                                ],
                                "Catch": [
                                    {
                                        "ErrorEquals": ["States.ALL"],
                                        "ResultPath": "$.error",
                                        "Next": "SnowflakeExtractionFailed",
                                    }
                                ],
                                "End": True,
                            },
                            "SnowflakeExtractionFailed": {
                                "Type": "Task",
                                "Resource": "arn:aws:states:::sns:publish",
                                "Parameters": {
                                    "TopicArn": sns_topic_arn,
                                    "Subject": "DrugInteractionML: Snowflake Extraction Failed",
                                    "Message.$": "States.Format('Snowflake feature extraction failed: {}', $.error.Cause)",
                                },
                                "End": True,
                            },
                        },
                    },
                ],
                "ResultPath": "$.extraction_results",
                "Next": "FeatureEngineering",
                "Catch": [
                    {
                        "ErrorEquals": ["States.ALL"],
                        "ResultPath": "$.error",
                        "Next": "PipelineFailedNotification",
                    }
                ],
            },
            # ── Stage 2: Feature Engineering & Validation ─────────────
            "FeatureEngineering": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "Parameters": {
                    "FunctionName": feature_eng_lambda_arn,
                    "Payload": {
                        "extraction_results.$": "$.extraction_results",
                        "feature_config.$": "$.feature_config",
                    },
                },
                "ResultPath": "$.engineered_features",
                "Retry": [
                    {
                        "ErrorEquals": ["States.TaskFailed"],
                        "IntervalSeconds": 30,
                        "MaxAttempts": 2,
                        "BackoffRate": 2.0,
                    }
                ],
                "Catch": [
                    {
                        "ErrorEquals": ["States.ALL"],
                        "ResultPath": "$.error",
                        "Next": "PipelineFailedNotification",
                    }
                ],
                "Next": "ValidateFeatures",
            },
            "ValidateFeatures": {
                "Type": "Choice",
                "Choices": [
                    {
                        "Variable": "$.engineered_features.Payload.validation_passed",
                        "BooleanEquals": True,
                        "Next": "ModelTraining",
                    },
                    {
                        "Variable": "$.engineered_features.Payload.sample_count",
                        "NumericLessThan": 100,
                        "Next": "InsufficientDataNotification",
                    },
                ],
                "Default": "PipelineFailedNotification",
            },
            "InsufficientDataNotification": {
                "Type": "Task",
                "Resource": "arn:aws:states:::sns:publish",
                "Parameters": {
                    "TopicArn": sns_topic_arn,
                    "Subject": "DrugInteractionML: Insufficient Training Data",
                    "Message": "Feature validation failed due to insufficient samples. Pipeline halted.",
                },
                "Next": "PipelineFailed",
            },
            # ── Stage 3: Model Training (SageMaker) ──────────────────
            "ModelTraining": {
                "Type": "Task",
                "Resource": "arn:aws:states:::sagemaker:createTrainingJob.sync",
                "Parameters": {
                    "TrainingJobName.$": "States.Format('drug-interaction-{}', $$.Execution.StartTime)",
                    "AlgorithmSpecification": {
                        "TrainingImage.$": "$.training_config.image_uri",
                        "TrainingInputMode": "File",
                    },
                    "RoleArn.$": "$.training_config.role_arn",
                    "InputDataConfig": [
                        {
                            "ChannelName": "train",
                            "DataSource": {
                                "S3DataSource": {
                                    "S3DataType": "S3Prefix",
                                    "S3Uri.$": "$.engineered_features.Payload.train_s3_uri",
                                    "S3DataDistributionType": "FullyReplicated",
                                }
                            },
                            "ContentType": "text/csv",
                        },
                        {
                            "ChannelName": "validation",
                            "DataSource": {
                                "S3DataSource": {
                                    "S3DataType": "S3Prefix",
                                    "S3Uri.$": "$.engineered_features.Payload.val_s3_uri",
                                    "S3DataDistributionType": "FullyReplicated",
                                }
                            },
                            "ContentType": "text/csv",
                        },
                    ],
                    "OutputDataConfig": {
                        "S3OutputPath.$": "$.training_config.output_s3_uri",
                    },
                    "ResourceConfig": {
                        "InstanceCount": 1,
                        "InstanceType": "ml.m5.xlarge",
                        "VolumeSizeInGB": 50,
                    },
                    "StoppingCondition": {"MaxRuntimeInSeconds": 7200},
                    "HyperParameters.$": "$.training_config.hyperparameters",
                },
                "ResultPath": "$.training_result",
                "Retry": [
                    {
                        "ErrorEquals": ["SageMaker.AmazonSageMakerException"],
                        "IntervalSeconds": 120,
                        "MaxAttempts": 2,
                        "BackoffRate": 2.0,
                    }
                ],
                "Catch": [
                    {
                        "ErrorEquals": ["States.ALL"],
                        "ResultPath": "$.error",
                        "Next": "TrainingFailedNotification",
                    }
                ],
                "Next": "ModelEvaluation",
            },
            "TrainingFailedNotification": {
                "Type": "Task",
                "Resource": "arn:aws:states:::sns:publish",
                "Parameters": {
                    "TopicArn": sns_topic_arn,
                    "Subject": "DrugInteractionML: Training Job Failed",
                    "Message.$": "States.Format('SageMaker training job failed: {}', $.error.Cause)",
                },
                "Next": "PipelineFailed",
            },
            # ── Stage 4: Model Evaluation ─────────────────────────────
            "ModelEvaluation": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "Parameters": {
                    "FunctionName": evaluation_lambda_arn,
                    "Payload": {
                        "training_result.$": "$.training_result",
                        "model_artifact.$": "$.training_result.ModelArtifacts.S3ModelArtifacts",
                    },
                },
                "ResultPath": "$.evaluation_result",
                "Next": "DeploymentDecision",
            },
            # ── Stage 5: Conditional Deployment ───────────────────────
            "DeploymentDecision": {
                "Type": "Choice",
                "Comment": "Deploy only if model meets performance threshold",
                "Choices": [
                    {
                        "And": [
                            {
                                "Variable": "$.evaluation_result.Payload.f1_macro",
                                "NumericGreaterThanEquals": performance_threshold,
                            },
                            {
                                "Variable": "$.evaluation_result.Payload.is_improvement",
                                "BooleanEquals": True,
                            },
                        ],
                        "Next": "DeployModel",
                    },
                    {
                        "Variable": "$.evaluation_result.Payload.f1_macro",
                        "NumericLessThan": performance_threshold,
                        "Next": "BelowThresholdNotification",
                    },
                ],
                "Default": "NoImprovementNotification",
            },
            "BelowThresholdNotification": {
                "Type": "Task",
                "Resource": "arn:aws:states:::sns:publish",
                "Parameters": {
                    "TopicArn": sns_topic_arn,
                    "Subject": "DrugInteractionML: Model Below Threshold",
                    "Message.$": "States.Format('Model f1={} below threshold={}', $.evaluation_result.Payload.f1_macro, '{}')",
                },
                "Next": "PipelineComplete",
            },
            "NoImprovementNotification": {
                "Type": "Task",
                "Resource": "arn:aws:states:::sns:publish",
                "Parameters": {
                    "TopicArn": sns_topic_arn,
                    "Subject": "DrugInteractionML: No Improvement Over Production",
                    "Message": "New model does not improve over the current production model.",
                },
                "Next": "PipelineComplete",
            },
            # ── Stage 6: Deployment ───────────────────────────────────
            "DeployModel": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "Parameters": {
                    "FunctionName": deployment_lambda_arn,
                    "Payload": {
                        "model_artifact.$": "$.training_result.ModelArtifacts.S3ModelArtifacts",
                        "endpoint_name": "drug-interaction-endpoint",
                        "deployment_config.$": "$.deployment_config",
                    },
                },
                "ResultPath": "$.deployment_result",
                "Retry": [
                    {
                        "ErrorEquals": ["States.TaskFailed"],
                        "IntervalSeconds": 60,
                        "MaxAttempts": 2,
                        "BackoffRate": 2.0,
                    }
                ],
                "Catch": [
                    {
                        "ErrorEquals": ["States.ALL"],
                        "ResultPath": "$.error",
                        "Next": "DeploymentRollback",
                    }
                ],
                "Next": "SetupMonitoring",
            },
            "DeploymentRollback": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "Parameters": {
                    "FunctionName": rollback_lambda_arn,
                    "Payload": {
                        "endpoint_name": "drug-interaction-endpoint",
                        "error.$": "$.error",
                    },
                },
                "ResultPath": "$.rollback_result",
                "Next": "DeploymentFailedNotification",
            },
            "DeploymentFailedNotification": {
                "Type": "Task",
                "Resource": "arn:aws:states:::sns:publish",
                "Parameters": {
                    "TopicArn": sns_topic_arn,
                    "Subject": "DrugInteractionML: Deployment Failed - Rolled Back",
                    "Message.$": "States.Format('Deployment failed and was rolled back: {}', $.error.Cause)",
                },
                "Next": "PipelineFailed",
            },
            # ── Stage 7: Monitoring Setup ─────────────────────────────
            "SetupMonitoring": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "Parameters": {
                    "FunctionName": monitoring_lambda_arn,
                    "Payload": {
                        "endpoint_name": "drug-interaction-endpoint",
                        "baseline_s3_uri.$": "$.engineered_features.Payload.baseline_s3_uri",
                    },
                },
                "ResultPath": "$.monitoring_result",
                "Next": "DeploymentSuccessNotification",
            },
            "DeploymentSuccessNotification": {
                "Type": "Task",
                "Resource": "arn:aws:states:::sns:publish",
                "Parameters": {
                    "TopicArn": sns_topic_arn,
                    "Subject": "DrugInteractionML: Deployment Successful",
                    "Message.$": "States.Format('Model deployed successfully. Endpoint: drug-interaction-endpoint. F1: {}', $.evaluation_result.Payload.f1_macro)",
                },
                "Next": "PipelineComplete",
            },
            # ── Terminal states ───────────────────────────────────────
            "PipelineComplete": {
                "Type": "Succeed",
            },
            "PipelineFailed": {
                "Type": "Fail",
                "Error": "PipelineError",
                "Cause": "Pipeline failed. Check SNS notifications for details.",
            },
            "PipelineFailedNotification": {
                "Type": "Task",
                "Resource": "arn:aws:states:::sns:publish",
                "Parameters": {
                    "TopicArn": sns_topic_arn,
                    "Subject": "DrugInteractionML: Pipeline Failed",
                    "Message.$": "States.Format('Pipeline execution failed: {}', $.error.Cause)",
                },
                "Next": "PipelineFailed",
            },
        },
    }


# ---------------------------------------------------------------------------
# Step Functions Manager
# ---------------------------------------------------------------------------


@dataclass
class StepFunctionsManager:
    """Create and manage AWS Step Functions state machines.

    Parameters
    ----------
    region : str
        AWS region name.
    role_arn : str
        IAM role ARN for the state machine execution.
    """

    region: str = "us-east-1"
    role_arn: str = ""
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = boto3.client("stepfunctions", region_name=self.region)

    def create_state_machine(
        self,
        name: str,
        definition: dict[str, Any],
        *,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Create or update a Step Functions state machine.

        Parameters
        ----------
        name : str
            State machine name.
        definition : dict
            State machine definition.
        tags : dict, optional
            Tags to apply.

        Returns
        -------
        str
            State machine ARN.
        """
        definition_json = json.dumps(definition, indent=2)

        # Check if it already exists
        existing_arn = self._find_state_machine(name)
        if existing_arn:
            logger.info("Updating existing state machine: %s", name)
            self._client.update_state_machine(
                stateMachineArn=existing_arn,
                definition=definition_json,
                roleArn=self.role_arn,
            )
            return existing_arn

        tag_list = [{"key": k, "value": v} for k, v in (tags or {}).items()]
        response = self._client.create_state_machine(
            name=name,
            definition=definition_json,
            roleArn=self.role_arn,
            type="STANDARD",
            tags=tag_list,
        )
        arn = response["stateMachineArn"]
        logger.info("Created state machine: %s (arn=%s)", name, arn)
        return arn

    def start_execution(
        self,
        state_machine_arn: str,
        execution_name: str,
        input_data: dict[str, Any],
    ) -> str:
        """Start a state machine execution.

        Returns
        -------
        str
            Execution ARN.
        """
        response = self._client.start_execution(
            stateMachineArn=state_machine_arn,
            name=execution_name,
            input=json.dumps(input_data),
        )
        execution_arn = response["executionArn"]
        logger.info("Started execution: %s", execution_arn)
        return execution_arn

    def get_execution_status(self, execution_arn: str) -> dict[str, Any]:
        """Get the current status of an execution."""
        response = self._client.describe_execution(executionArn=execution_arn)
        return {
            "status": response["status"],
            "startDate": str(response["startDate"]),
            "stopDate": str(response.get("stopDate", "")),
            "output": response.get("output"),
            "error": response.get("error"),
            "cause": response.get("cause"),
        }

    def _find_state_machine(self, name: str) -> str | None:
        """Find a state machine ARN by name."""
        paginator = self._client.get_paginator("list_state_machines")
        for page in paginator.paginate():
            for sm in page["stateMachines"]:
                if sm["name"] == name:
                    return sm["stateMachineArn"]
        return None

    def export_definition_json(
        self,
        definition: dict[str, Any],
        output_path: str,
    ) -> None:
        """Export the state machine definition to a JSON file."""
        from pathlib import Path

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(definition, indent=2))
        logger.info("Exported state machine definition to %s", output_path)
