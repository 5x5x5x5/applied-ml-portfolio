#!/bin/bash
# =============================================================================
# LocalStack initialization script
# Creates S3 buckets and SQS queues for local development
# =============================================================================

set -euo pipefail

echo "Initializing LocalStack resources..."

# Create S3 buckets
awslocal s3 mb s3://pharma-sentinel-input
awslocal s3 mb s3://pharma-sentinel-output
awslocal s3 mb s3://pharma-sentinel-models

# Create SQS queues (with dead letter queue)
awslocal sqs create-queue --queue-name pharma-sentinel-critical-dlq
awslocal sqs create-queue \
    --queue-name pharma-sentinel-critical \
    --attributes '{
        "RedrivePolicy": "{\"deadLetterTargetArn\":\"arn:aws:sqs:us-east-1:000000000000:pharma-sentinel-critical-dlq\",\"maxReceiveCount\":\"3\"}",
        "VisibilityTimeout": "300",
        "MessageRetentionPeriod": "1209600"
    }'

awslocal sqs create-queue --queue-name pharma-sentinel-processing-dlq
awslocal sqs create-queue \
    --queue-name pharma-sentinel-processing \
    --attributes '{
        "RedrivePolicy": "{\"deadLetterTargetArn\":\"arn:aws:sqs:us-east-1:000000000000:pharma-sentinel-processing-dlq\",\"maxReceiveCount\":\"3\"}",
        "VisibilityTimeout": "600",
        "MessageRetentionPeriod": "1209600"
    }'

# Create sample S3 prefixes
awslocal s3api put-object --bucket pharma-sentinel-input --key incoming/
awslocal s3api put-object --bucket pharma-sentinel-input --key faers/
awslocal s3api put-object --bucket pharma-sentinel-input --key archive/
awslocal s3api put-object --bucket pharma-sentinel-output --key classified/
awslocal s3api put-object --bucket pharma-sentinel-output --key processed/
awslocal s3api put-object --bucket pharma-sentinel-models --key models/

echo "LocalStack initialization complete."
echo "S3 buckets:"
awslocal s3 ls
echo "SQS queues:"
awslocal sqs list-queues
