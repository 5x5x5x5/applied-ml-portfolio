#!/bin/bash
# CloudGenomics - LocalStack initialization script
# Creates AWS resources for local development

set -euo pipefail

echo "Initializing LocalStack resources for CloudGenomics..."

# Create S3 buckets
awslocal s3 mb s3://cloudgenomics-vcf-files
awslocal s3 mb s3://cloudgenomics-results

# Enable versioning on VCF bucket
awslocal s3api put-bucket-versioning \
    --bucket cloudgenomics-vcf-files \
    --versioning-configuration Status=Enabled

# Create KMS key
KEY_ID=$(awslocal kms create-key --description "CloudGenomics data encryption key" \
    --query 'KeyMetadata.KeyId' --output text)
awslocal kms create-alias \
    --alias-name alias/cloudgenomics-key \
    --target-key-id "$KEY_ID"

# Create SNS topic for notifications
awslocal sns create-topic --name cloudgenomics-notifications

echo "LocalStack initialization complete."
echo "  S3 buckets: cloudgenomics-vcf-files, cloudgenomics-results"
echo "  KMS key: $KEY_ID (alias: cloudgenomics-key)"
echo "  SNS topic: cloudgenomics-notifications"
