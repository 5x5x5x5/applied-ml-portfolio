"""AWS Step Functions workflow definition for the CloudGenomics pipeline.

Generates the Step Functions state machine definition as Python-generated
JSON. The workflow orchestrates:
  VCF Upload -> Validation -> Annotation -> Classification -> Reporting

Designed for integration with CloudFormation and the CloudGenomics
ECS/Lambda infrastructure.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StepFunctionConfig:
    """Configuration for the Step Functions state machine."""

    region: str = "us-east-1"
    account_id: str = "123456789012"
    vcf_bucket: str = "cloudgenomics-vcf-files"
    results_bucket: str = "cloudgenomics-results"
    validation_lambda_arn: str = ""
    annotation_lambda_arn: str = ""
    classification_ecs_cluster: str = ""
    classification_task_def: str = ""
    classification_subnet_ids: list[str] | None = None
    classification_security_group_id: str = ""
    notification_topic_arn: str = ""
    dynamodb_table: str = "cloudgenomics-jobs"
    max_variants_per_batch: int = 1000
    classification_timeout_seconds: int = 3600
    retry_max_attempts: int = 3

    def __post_init__(self) -> None:
        base = f"arn:aws:lambda:{self.region}:{self.account_id}:function"
        if not self.validation_lambda_arn:
            self.validation_lambda_arn = f"{base}:cloudgenomics-vcf-validator"
        if not self.annotation_lambda_arn:
            self.annotation_lambda_arn = f"{base}:cloudgenomics-variant-annotator"
        if not self.classification_ecs_cluster:
            self.classification_ecs_cluster = (
                f"arn:aws:ecs:{self.region}:{self.account_id}:cluster/cloudgenomics-cluster"
            )
        if not self.classification_task_def:
            self.classification_task_def = (
                f"arn:aws:ecs:{self.region}:{self.account_id}:"
                f"task-definition/cloudgenomics-classifier"
            )
        if not self.notification_topic_arn:
            self.notification_topic_arn = (
                f"arn:aws:sns:{self.region}:{self.account_id}:cloudgenomics-notifications"
            )
        if self.classification_subnet_ids is None:
            self.classification_subnet_ids = []


def build_state_machine(config: StepFunctionConfig | None = None) -> dict[str, Any]:
    """Build the complete Step Functions state machine definition.

    Args:
        config: Step Functions configuration. Uses defaults if None.

    Returns:
        Complete state machine definition as a dictionary (ASL format).
    """
    if config is None:
        config = StepFunctionConfig()

    state_machine: dict[str, Any] = {
        "Comment": (
            "CloudGenomics Variant Classification Pipeline - "
            "Processes VCF files through validation, annotation, "
            "ML classification, and reporting."
        ),
        "StartAt": "InitializeJob",
        "States": {
            **_initialize_job_state(),
            **_validate_vcf_state(config),
            **_check_validation_state(),
            **_split_variants_state(config),
            **_annotate_variants_state(config),
            **_classify_variants_state(config),
            **_aggregate_results_state(),
            **_generate_report_state(config),
            **_notify_completion_state(config),
            **_handle_failure_state(config),
            **_validation_failed_state(config),
        },
    }

    logger.info("Built state machine with %d states", len(state_machine["States"]))
    return state_machine


def _initialize_job_state() -> dict[str, Any]:
    """Create the job initialization state.

    Sets up the job record in DynamoDB with initial metadata.
    """
    return {
        "InitializeJob": {
            "Type": "Pass",
            "Comment": "Initialize job metadata and set processing parameters",
            "Parameters": {
                "jobId.$": "$$.Execution.Id",
                "startTime.$": "$$.Execution.StartTime",
                "status": "INITIALIZING",
                "input.$": "$",
                "processingConfig": {
                    "maxVariantsPerBatch": 1000,
                    "qualityThresholds": {
                        "minQual": 30,
                        "minDepth": 10,
                        "minGQ": 20,
                    },
                    "enableAnnotation": True,
                    "enableSpliceAnalysis": True,
                },
            },
            "ResultPath": "$.jobContext",
            "Next": "ValidateVCF",
        }
    }


def _validate_vcf_state(config: StepFunctionConfig) -> dict[str, Any]:
    """Create the VCF validation Lambda invocation state.

    Validates file format, checks for corruption, and extracts metadata.
    """
    return {
        "ValidateVCF": {
            "Type": "Task",
            "Comment": "Validate VCF file format, check integrity, extract metadata",
            "Resource": "arn:aws:states:::lambda:invoke",
            "Parameters": {
                "FunctionName": config.validation_lambda_arn,
                "Payload": {
                    "s3Bucket.$": "$.jobContext.input.s3Bucket",
                    "s3Key.$": "$.jobContext.input.s3Key",
                    "jobId.$": "$.jobContext.jobId",
                    "qualityThresholds.$": "$.jobContext.processingConfig.qualityThresholds",
                },
            },
            "ResultPath": "$.validationResult",
            "ResultSelector": {
                "isValid.$": "$.Payload.isValid",
                "variantCount.$": "$.Payload.variantCount",
                "sampleCount.$": "$.Payload.sampleCount",
                "referenceGenome.$": "$.Payload.referenceGenome",
                "errors.$": "$.Payload.errors",
                "fileSize.$": "$.Payload.fileSize",
                "md5Checksum.$": "$.Payload.md5Checksum",
            },
            "Retry": [
                {
                    "ErrorEquals": [
                        "Lambda.ServiceException",
                        "Lambda.AWSLambdaException",
                        "Lambda.TooManyRequestsException",
                    ],
                    "IntervalSeconds": 2,
                    "MaxAttempts": config.retry_max_attempts,
                    "BackoffRate": 2.0,
                }
            ],
            "Catch": [
                {
                    "ErrorEquals": ["States.ALL"],
                    "Next": "HandleFailure",
                    "ResultPath": "$.error",
                }
            ],
            "TimeoutSeconds": 300,
            "Next": "CheckValidation",
        }
    }


def _check_validation_state() -> dict[str, Any]:
    """Create the validation result check (Choice state)."""
    return {
        "CheckValidation": {
            "Type": "Choice",
            "Comment": "Route based on VCF validation result",
            "Choices": [
                {
                    "Variable": "$.validationResult.isValid",
                    "BooleanEquals": True,
                    "Next": "SplitVariants",
                },
                {
                    "Variable": "$.validationResult.isValid",
                    "BooleanEquals": False,
                    "Next": "ValidationFailed",
                },
            ],
            "Default": "ValidationFailed",
        }
    }


def _split_variants_state(config: StepFunctionConfig) -> dict[str, Any]:
    """Create the variant splitting state for parallel processing.

    Divides large VCF files into batches for parallel annotation
    and classification.
    """
    return {
        "SplitVariants": {
            "Type": "Task",
            "Comment": (
                "Split VCF variants into batches for parallel processing. "
                f"Max {config.max_variants_per_batch} variants per batch."
            ),
            "Resource": "arn:aws:states:::lambda:invoke",
            "Parameters": {
                "FunctionName": config.annotation_lambda_arn,
                "Payload": {
                    "action": "split",
                    "s3Bucket.$": "$.jobContext.input.s3Bucket",
                    "s3Key.$": "$.jobContext.input.s3Key",
                    "jobId.$": "$.jobContext.jobId",
                    "maxBatchSize": config.max_variants_per_batch,
                    "variantCount.$": "$.validationResult.variantCount",
                },
            },
            "ResultPath": "$.batchInfo",
            "ResultSelector": {
                "batches.$": "$.Payload.batches",
                "totalBatches.$": "$.Payload.totalBatches",
            },
            "Retry": [
                {
                    "ErrorEquals": ["Lambda.ServiceException"],
                    "IntervalSeconds": 5,
                    "MaxAttempts": 2,
                    "BackoffRate": 2.0,
                }
            ],
            "Catch": [
                {
                    "ErrorEquals": ["States.ALL"],
                    "Next": "HandleFailure",
                    "ResultPath": "$.error",
                }
            ],
            "Next": "AnnotateVariants",
        }
    }


def _annotate_variants_state(config: StepFunctionConfig) -> dict[str, Any]:
    """Create the Map state for parallel variant annotation.

    Processes each batch through annotation Lambda functions in parallel.
    """
    return {
        "AnnotateVariants": {
            "Type": "Map",
            "Comment": "Annotate variant batches in parallel with population frequencies and predictions",
            "ItemsPath": "$.batchInfo.batches",
            "MaxConcurrency": 10,
            "Parameters": {
                "batchIndex.$": "$$.Map.Item.Index",
                "batch.$": "$$.Map.Item.Value",
                "jobId.$": "$.jobContext.jobId",
                "s3Bucket.$": "$.jobContext.input.s3Bucket",
                "processingConfig.$": "$.jobContext.processingConfig",
            },
            "Iterator": {
                "StartAt": "AnnotateBatch",
                "States": {
                    "AnnotateBatch": {
                        "Type": "Task",
                        "Resource": "arn:aws:states:::lambda:invoke",
                        "Parameters": {
                            "FunctionName": config.annotation_lambda_arn,
                            "Payload": {
                                "action": "annotate",
                                "batchIndex.$": "$.batchIndex",
                                "batch.$": "$.batch",
                                "jobId.$": "$.jobId",
                                "s3Bucket.$": "$.s3Bucket",
                                "processingConfig.$": "$.processingConfig",
                            },
                        },
                        "ResultSelector": {
                            "annotatedVariants.$": "$.Payload.annotatedVariants",
                            "batchIndex.$": "$.Payload.batchIndex",
                            "annotationStats.$": "$.Payload.annotationStats",
                        },
                        "Retry": [
                            {
                                "ErrorEquals": [
                                    "Lambda.ServiceException",
                                    "Lambda.TooManyRequestsException",
                                ],
                                "IntervalSeconds": 5,
                                "MaxAttempts": config.retry_max_attempts,
                                "BackoffRate": 2.0,
                            }
                        ],
                        "End": True,
                    }
                },
            },
            "ResultPath": "$.annotationResults",
            "Catch": [
                {
                    "ErrorEquals": ["States.ALL"],
                    "Next": "HandleFailure",
                    "ResultPath": "$.error",
                }
            ],
            "Next": "ClassifyVariants",
        }
    }


def _classify_variants_state(config: StepFunctionConfig) -> dict[str, Any]:
    """Create the ECS Fargate classification task state.

    Runs the ML classification model on annotated variants using
    ECS Fargate for GPU/compute-intensive workloads.
    """
    return {
        "ClassifyVariants": {
            "Type": "Task",
            "Comment": "Run ML variant classification on ECS Fargate",
            "Resource": "arn:aws:states:::ecs:runTask.sync",
            "Parameters": {
                "LaunchType": "FARGATE",
                "Cluster": config.classification_ecs_cluster,
                "TaskDefinition": config.classification_task_def,
                "NetworkConfiguration": {
                    "AwsvpcConfiguration": {
                        "Subnets": config.classification_subnet_ids or [],
                        "SecurityGroups": [config.classification_security_group_id],
                        "AssignPublicIp": "DISABLED",
                    }
                },
                "Overrides": {
                    "ContainerOverrides": [
                        {
                            "Name": "classifier",
                            "Environment": [
                                {
                                    "Name": "JOB_ID",
                                    "Value.$": "$.jobContext.jobId",
                                },
                                {
                                    "Name": "S3_BUCKET",
                                    "Value.$": "$.jobContext.input.s3Bucket",
                                },
                                {
                                    "Name": "ANNOTATION_RESULTS",
                                    "Value.$": "States.JsonToString($.annotationResults)",
                                },
                            ],
                        }
                    ]
                },
            },
            "ResultPath": "$.classificationResult",
            "TimeoutSeconds": config.classification_timeout_seconds,
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
                    "Next": "HandleFailure",
                    "ResultPath": "$.error",
                }
            ],
            "Next": "AggregateResults",
        }
    }


def _aggregate_results_state() -> dict[str, Any]:
    """Create the results aggregation state.

    Combines classification results from all batches and computes
    summary statistics.
    """
    return {
        "AggregateResults": {
            "Type": "Pass",
            "Comment": "Aggregate classification results and compute summary statistics",
            "Parameters": {
                "jobId.$": "$.jobContext.jobId",
                "status": "AGGREGATING",
                "totalVariants.$": "$.validationResult.variantCount",
                "classificationResult.$": "$.classificationResult",
                "annotationResults.$": "$.annotationResults",
                "summary": {
                    "completedAt.$": "$$.State.EnteredTime",
                    "pipelineVersion": "1.0.0",
                },
            },
            "ResultPath": "$.aggregation",
            "Next": "GenerateReport",
        }
    }


def _generate_report_state(config: StepFunctionConfig) -> dict[str, Any]:
    """Create the report generation state.

    Generates a clinical-grade variant classification report and
    stores it in S3.
    """
    return {
        "GenerateReport": {
            "Type": "Task",
            "Comment": "Generate clinical variant classification report and store in S3",
            "Resource": "arn:aws:states:::lambda:invoke",
            "Parameters": {
                "FunctionName": config.annotation_lambda_arn,
                "Payload": {
                    "action": "generate_report",
                    "jobId.$": "$.jobContext.jobId",
                    "aggregation.$": "$.aggregation",
                    "outputBucket": config.results_bucket,
                    "outputKey.$": "States.Format('reports/{}/classification_report.json', $.jobContext.jobId)",
                },
            },
            "ResultPath": "$.reportResult",
            "ResultSelector": {
                "reportS3Key.$": "$.Payload.reportS3Key",
                "reportUrl.$": "$.Payload.reportUrl",
                "generatedAt.$": "$.Payload.generatedAt",
            },
            "Retry": [
                {
                    "ErrorEquals": ["Lambda.ServiceException"],
                    "IntervalSeconds": 5,
                    "MaxAttempts": 2,
                    "BackoffRate": 2.0,
                }
            ],
            "Catch": [
                {
                    "ErrorEquals": ["States.ALL"],
                    "Next": "HandleFailure",
                    "ResultPath": "$.error",
                }
            ],
            "Next": "NotifyCompletion",
        }
    }


def _notify_completion_state(config: StepFunctionConfig) -> dict[str, Any]:
    """Create the completion notification state.

    Sends SNS notification with pipeline results.
    """
    return {
        "NotifyCompletion": {
            "Type": "Task",
            "Comment": "Send completion notification via SNS",
            "Resource": "arn:aws:states:::sns:publish",
            "Parameters": {
                "TopicArn": config.notification_topic_arn,
                "Subject": "CloudGenomics - Classification Complete",
                "Message": {
                    "jobId.$": "$.jobContext.jobId",
                    "status": "COMPLETED",
                    "totalVariants.$": "$.validationResult.variantCount",
                    "reportUrl.$": "$.reportResult.reportUrl",
                    "completedAt.$": "$$.State.EnteredTime",
                },
            },
            "ResultPath": "$.notificationResult",
            "End": True,
        }
    }


def _handle_failure_state(config: StepFunctionConfig) -> dict[str, Any]:
    """Create the failure handling state.

    Captures error details and sends failure notification.
    """
    return {
        "HandleFailure": {
            "Type": "Task",
            "Comment": "Handle pipeline failure - log error and notify",
            "Resource": "arn:aws:states:::sns:publish",
            "Parameters": {
                "TopicArn": config.notification_topic_arn,
                "Subject": "CloudGenomics - Pipeline FAILED",
                "Message": {
                    "jobId.$": "$.jobContext.jobId",
                    "status": "FAILED",
                    "error.$": "$.error",
                    "failedAt.$": "$$.State.EnteredTime",
                },
            },
            "End": True,
        }
    }


def _validation_failed_state(config: StepFunctionConfig) -> dict[str, Any]:
    """Create the validation failure state.

    Handles cases where the VCF file fails validation.
    """
    return {
        "ValidationFailed": {
            "Type": "Task",
            "Comment": "Handle VCF validation failure - notify user of issues",
            "Resource": "arn:aws:states:::sns:publish",
            "Parameters": {
                "TopicArn": config.notification_topic_arn,
                "Subject": "CloudGenomics - VCF Validation Failed",
                "Message": {
                    "jobId.$": "$.jobContext.jobId",
                    "status": "VALIDATION_FAILED",
                    "errors.$": "$.validationResult.errors",
                    "failedAt.$": "$$.State.EnteredTime",
                },
            },
            "End": True,
        }
    }


def export_state_machine_json(
    config: StepFunctionConfig | None = None,
    output_path: str | None = None,
    indent: int = 2,
) -> str:
    """Export the state machine definition as a JSON string.

    Args:
        config: Step Functions configuration.
        output_path: Optional file path to write the JSON.
        indent: JSON indentation level.

    Returns:
        JSON string of the state machine definition.
    """
    definition = build_state_machine(config)
    json_str = json.dumps(definition, indent=indent, default=str)

    if output_path:
        from pathlib import Path

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_str)
        logger.info("State machine definition written to %s", output_path)

    return json_str


def build_iam_policy(config: StepFunctionConfig | None = None) -> dict[str, Any]:
    """Build the IAM policy for the Step Functions execution role.

    Args:
        config: Step Functions configuration.

    Returns:
        IAM policy document as a dictionary.
    """
    if config is None:
        config = StepFunctionConfig()

    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "InvokeLambdaFunctions",
                "Effect": "Allow",
                "Action": ["lambda:InvokeFunction"],
                "Resource": [
                    config.validation_lambda_arn,
                    config.annotation_lambda_arn,
                ],
            },
            {
                "Sid": "RunECSTasks",
                "Effect": "Allow",
                "Action": [
                    "ecs:RunTask",
                    "ecs:StopTask",
                    "ecs:DescribeTasks",
                ],
                "Resource": [config.classification_task_def],
            },
            {
                "Sid": "PassRoleToECS",
                "Effect": "Allow",
                "Action": ["iam:PassRole"],
                "Resource": [
                    f"arn:aws:iam::{config.account_id}:role/cloudgenomics-ecs-task-role",
                    f"arn:aws:iam::{config.account_id}:role/cloudgenomics-ecs-execution-role",
                ],
            },
            {
                "Sid": "PublishSNSNotifications",
                "Effect": "Allow",
                "Action": ["sns:Publish"],
                "Resource": [config.notification_topic_arn],
            },
            {
                "Sid": "AccessS3Buckets",
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:ListBucket",
                ],
                "Resource": [
                    f"arn:aws:s3:::{config.vcf_bucket}",
                    f"arn:aws:s3:::{config.vcf_bucket}/*",
                    f"arn:aws:s3:::{config.results_bucket}",
                    f"arn:aws:s3:::{config.results_bucket}/*",
                ],
            },
            {
                "Sid": "CloudWatchLogs",
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogDelivery",
                    "logs:GetLogDelivery",
                    "logs:UpdateLogDelivery",
                    "logs:DeleteLogDelivery",
                    "logs:ListLogDeliveries",
                    "logs:PutResourcePolicy",
                    "logs:DescribeResourcePolicies",
                    "logs:DescribeLogGroups",
                ],
                "Resource": "*",
            },
            {
                "Sid": "XRayTracing",
                "Effect": "Allow",
                "Action": [
                    "xray:PutTraceSegments",
                    "xray:PutTelemetryRecords",
                    "xray:GetSamplingRules",
                    "xray:GetSamplingTargets",
                ],
                "Resource": "*",
            },
            {
                "Sid": "ManageECSEvents",
                "Effect": "Allow",
                "Action": [
                    "events:PutTargets",
                    "events:PutRule",
                    "events:DescribeRule",
                ],
                "Resource": [
                    f"arn:aws:events:{config.region}:{config.account_id}:rule/StepFunctions*"
                ],
            },
        ],
    }
