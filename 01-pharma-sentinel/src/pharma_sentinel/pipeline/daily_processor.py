"""Daily file processing job for FAERS adverse event data.

Lambda-compatible handler that reads new files from S3, runs NLP
preprocessing, classifies events by severity, writes results to
output bucket, and sends SQS notifications for critical events.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import UTC, datetime
from typing import Any

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from pharma_sentinel.config import AppSettings, get_settings
from pharma_sentinel.models.adverse_event_classifier import (
    AdverseEventClassifier,
    SeverityLevel,
)
from pharma_sentinel.pipeline.data_ingestion import FAERSDataIngester, FAERSRecord

logger = logging.getLogger(__name__)


# Common drug name patterns for entity extraction
DRUG_NAME_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b([A-Z][a-z]+(?:mab|nib|lib|zumab|tinib|ciclib|rafenib))\b"),
    re.compile(r"\b([A-Z][a-z]+(?:statin|prazole|sartan|olol|pril|floxacin))\b"),
    re.compile(r"\b([A-Z][a-z]+(?:azepam|zolam|barbital|etine|amine))\b"),
    re.compile(r"\b(aspirin|ibuprofen|acetaminophen|metformin|lisinopril)\b", re.IGNORECASE),
    re.compile(r"\b(warfarin|heparin|insulin|morphine|fentanyl)\b", re.IGNORECASE),
]

# MedDRA-like adverse event term patterns
ADVERSE_EVENT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(nausea|vomiting|diarrhea|headache|dizziness|fatigue)\b", re.IGNORECASE),
    re.compile(r"\b(rash|pruritus|edema|dyspnea|chest pain|palpitations)\b", re.IGNORECASE),
    re.compile(r"\b(seizure|syncope|hemorrhage|thrombosis|anaphylaxis)\b", re.IGNORECASE),
    re.compile(
        r"\b(hepatotoxicity|nephrotoxicity|cardiotoxicity|neurotoxicity)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(stevens-johnson|toxic epidermal necrolysis|agranulocytosis)\b",
        re.IGNORECASE,
    ),
]


class NLPPreprocessor:
    """NLP preprocessing for adverse event report text.

    Handles tokenization, entity extraction for drug names and
    adverse event terms, and text normalization.
    """

    def tokenize(self, text: str) -> list[str]:
        """Tokenize clinical text into words.

        Args:
            text: Raw clinical text.

        Returns:
            List of tokens.
        """
        if not text:
            return []

        # Basic clinical text tokenization
        cleaned = re.sub(r"[^\w\s\-/]", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        tokens = cleaned.lower().split()

        # Remove very short tokens and stopwords
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "was",
            "were",
            "are",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "shall",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "and",
            "but",
            "or",
            "nor",
            "not",
            "no",
            "so",
            "if",
            "then",
            "than",
            "too",
            "very",
            "just",
            "about",
            "above",
            "below",
            "between",
        }

        return [t for t in tokens if len(t) > 1 and t not in stopwords]

    def extract_drug_names(self, text: str) -> list[str]:
        """Extract drug name entities from clinical text.

        Args:
            text: Clinical text possibly containing drug names.

        Returns:
            List of unique extracted drug names.
        """
        drugs: set[str] = set()

        for pattern in DRUG_NAME_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                drugs.add(match.strip().lower())

        return sorted(drugs)

    def extract_adverse_events(self, text: str) -> list[str]:
        """Extract adverse event terms from clinical text.

        Args:
            text: Clinical text possibly containing adverse event terms.

        Returns:
            List of unique extracted adverse event terms.
        """
        events: set[str] = set()

        for pattern in ADVERSE_EVENT_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                events.add(match.strip().lower())

        return sorted(events)

    def preprocess_report(self, record: FAERSRecord) -> dict[str, Any]:
        """Run full NLP preprocessing on a FAERS record.

        Args:
            record: A validated FAERS record.

        Returns:
            Dictionary with preprocessed text and extracted entities.
        """
        # Combine relevant text fields
        text_parts = [record.reaction_description]
        if record.drug_indication:
            text_parts.append(record.drug_indication)

        combined_text = " ".join(text_parts)

        tokens = self.tokenize(combined_text)
        extracted_drugs = self.extract_drug_names(combined_text)
        extracted_events = self.extract_adverse_events(combined_text)

        # Add the known drug name
        all_drugs = list(set([record.drug_name.lower()] + extracted_drugs))

        return {
            "report_id": record.report_id,
            "original_text": combined_text,
            "tokens": tokens,
            "token_count": len(tokens),
            "extracted_drugs": all_drugs,
            "extracted_adverse_events": extracted_events,
            "drug_name": record.drug_name,
            "patient_age": record.patient_age,
            "patient_sex": record.patient_sex,
        }


class DailyProcessor:
    """Daily batch processor for FAERS adverse event files.

    Orchestrates the daily processing pipeline: reads new files
    from S3, runs NLP preprocessing, classifies events, stores
    results, and sends notifications for critical events.

    Attributes:
        settings: Application configuration.
        ingester: FAERS data ingestion component.
        classifier: Adverse event severity classifier.
        nlp: NLP preprocessing component.
        s3_client: Boto3 S3 client.
        sqs_client: Boto3 SQS client.
    """

    def __init__(self, settings: AppSettings | None = None) -> None:
        """Initialize the daily processor.

        Args:
            settings: Application settings. Uses default if None.
        """
        self.settings = settings or get_settings()
        self.ingester = FAERSDataIngester(self.settings)
        self.classifier = AdverseEventClassifier()
        self.nlp = NLPPreprocessor()

        boto_config = BotoConfig(
            retries={"max_attempts": 3, "mode": "adaptive"},
            connect_timeout=10,
            read_timeout=30,
        )

        client_kwargs: dict[str, Any] = {
            "config": boto_config,
            "region_name": self.settings.aws.region,
        }

        if self.settings.aws.endpoint_url:
            client_kwargs["endpoint_url"] = self.settings.aws.endpoint_url
        if self.settings.aws.access_key_id:
            client_kwargs["aws_access_key_id"] = self.settings.aws.access_key_id
        if self.settings.aws.secret_access_key:
            client_kwargs["aws_secret_access_key"] = self.settings.aws.secret_access_key

        self.s3_client = boto3.client("s3", **client_kwargs)
        self.sqs_client = boto3.client("sqs", **client_kwargs)

        logger.info("DailyProcessor initialized")

    def load_model(self) -> None:
        """Load the classifier model from S3 or local path.

        Downloads the model artifact from S3 if not available locally,
        then loads it into the classifier.
        """
        local_path = self.settings.model.artifact_path

        try:
            # Try loading from local path first
            self.classifier.load_model(local_path)
            logger.info("Model loaded from local path: %s", local_path)
        except FileNotFoundError:
            logger.info("Local model not found, downloading from S3")
            try:
                response = self.s3_client.get_object(
                    Bucket=self.settings.aws.s3_model_bucket,
                    Key=self.settings.model.s3_artifact_key,
                )
                model_data = response["Body"].read()

                # Save locally
                import pathlib

                pathlib.Path(local_path).parent.mkdir(parents=True, exist_ok=True)
                with open(local_path, "wb") as f:
                    f.write(model_data)

                self.classifier.load_model(local_path)
                logger.info("Model downloaded from S3 and loaded")
            except ClientError:
                logger.exception("Failed to download model from S3")
                raise

    def process_file(self, s3_key: str) -> list[dict[str, Any]]:
        """Process a single FAERS file through the full pipeline.

        Args:
            s3_key: S3 key of the input file.

        Returns:
            List of classified event results.
        """
        start_time = time.monotonic()
        logger.info("Processing file: %s", s3_key)

        # Download and parse
        data = self.ingester.download_file(s3_key)

        if s3_key.lower().endswith(".csv"):
            records = self.ingester.parse_csv_file(data, s3_key)
        elif s3_key.lower().endswith(".xml"):
            records = self.ingester.parse_xml_file(data, s3_key)
        else:
            logger.warning("Unsupported file format: %s", s3_key)
            return []

        if not records:
            logger.warning("No valid records found in %s", s3_key)
            return []

        # NLP preprocessing
        preprocessed = [self.nlp.preprocess_report(record) for record in records]

        # Classify events
        texts = [p["original_text"] for p in preprocessed]

        if not self.classifier.is_fitted:
            logger.warning("Classifier not fitted; skipping classification")
            return []

        predictions = self.classifier.predict(texts)

        # Combine results
        results: list[dict[str, Any]] = []
        critical_events: list[dict[str, Any]] = []

        for record, prep, pred in zip(records, preprocessed, predictions):
            result = {
                "report_id": record.report_id,
                "case_id": record.case_id,
                "drug_name": record.drug_name,
                "reaction_description": record.reaction_description,
                "predicted_severity": pred["severity"],
                "confidence": pred["confidence"],
                "probabilities": pred["probabilities"],
                "extracted_drugs": prep["extracted_drugs"],
                "extracted_adverse_events": prep["extracted_adverse_events"],
                "patient_age": record.patient_age,
                "patient_sex": record.patient_sex,
                "processed_at": datetime.now(tz=UTC).isoformat(),
                "source_file": s3_key,
            }
            results.append(result)

            # Track critical events for notification
            if pred["severity"] in (SeverityLevel.CRITICAL.value, SeverityLevel.SEVERE.value):
                critical_events.append(result)

        # Store results
        self._store_results(results, s3_key)

        # Send notifications for critical events
        if critical_events:
            self._send_critical_notifications(critical_events)

        elapsed = time.monotonic() - start_time
        logger.info(
            "Processed %d records from %s in %.2fs (%d critical)",
            len(results),
            s3_key,
            elapsed,
            len(critical_events),
        )

        return results

    def _store_results(self, results: list[dict[str, Any]], source_key: str) -> None:
        """Store classified results to S3 output bucket.

        Args:
            results: List of classified event results.
            source_key: Original source file S3 key.
        """
        if not results:
            return

        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        source_name = source_key.rsplit("/", maxsplit=1)[-1].rsplit(".", maxsplit=1)[0]
        output_key = f"classified/{timestamp}/{source_name}.json"

        try:
            self.s3_client.put_object(
                Bucket=self.settings.aws.s3_output_bucket,
                Key=output_key,
                Body=json.dumps(results, default=str).encode("utf-8"),
                ContentType="application/json",
                ServerSideEncryption="aws:kms",
            )
            logger.info(
                "Results stored to s3://%s/%s", self.settings.aws.s3_output_bucket, output_key
            )
        except ClientError:
            logger.exception("Failed to store results to S3: %s", output_key)
            raise

    def _send_critical_notifications(self, critical_events: list[dict[str, Any]]) -> None:
        """Send SQS notifications for critical adverse events.

        Args:
            critical_events: List of events classified as critical or severe.
        """
        if not self.settings.aws.sqs_critical_queue_url:
            logger.warning("SQS critical queue URL not configured; skipping notifications")
            return

        for event in critical_events:
            message = {
                "event_type": "CRITICAL_ADVERSE_EVENT",
                "report_id": event["report_id"],
                "drug_name": event["drug_name"],
                "severity": event["predicted_severity"],
                "confidence": event["confidence"],
                "reaction": event["reaction_description"][:500],
                "timestamp": datetime.now(tz=UTC).isoformat(),
            }

            try:
                self.sqs_client.send_message(
                    QueueUrl=self.settings.aws.sqs_critical_queue_url,
                    MessageBody=json.dumps(message, default=str),
                    MessageGroupId="critical-events",
                    MessageAttributes={
                        "severity": {
                            "DataType": "String",
                            "StringValue": event["predicted_severity"],
                        },
                        "drug_name": {
                            "DataType": "String",
                            "StringValue": event["drug_name"][:256],
                        },
                    },
                )
            except ClientError:
                logger.exception(
                    "Failed to send SQS notification for report %s",
                    event["report_id"],
                )

        logger.info("Sent %d critical event notifications", len(critical_events))

    def process_new_files(self, prefix: str = "incoming/") -> dict[str, Any]:
        """Process all new files in the incoming prefix.

        Args:
            prefix: S3 prefix for incoming files.

        Returns:
            Summary of processing results.
        """
        start_time = time.monotonic()
        logger.info("Starting daily processing for prefix: %s", prefix)

        files = self.ingester.list_input_files(prefix=prefix)

        summary: dict[str, Any] = {
            "files_found": len(files),
            "files_processed": 0,
            "total_records": 0,
            "critical_events": 0,
            "errors": [],
            "started_at": datetime.now(tz=UTC).isoformat(),
        }

        for file_key in files:
            try:
                results = self.process_file(file_key)
                summary["files_processed"] += 1
                summary["total_records"] += len(results)
                summary["critical_events"] += sum(
                    1
                    for r in results
                    if r["predicted_severity"]
                    in (SeverityLevel.CRITICAL.value, SeverityLevel.SEVERE.value)
                )

                # Move processed file to archive
                self._archive_file(file_key)

            except Exception as exc:
                logger.exception("Failed to process file: %s", file_key)
                summary["errors"].append({"file": file_key, "error": str(exc)})

        elapsed = time.monotonic() - start_time
        summary["elapsed_seconds"] = round(elapsed, 2)
        summary["completed_at"] = datetime.now(tz=UTC).isoformat()

        logger.info("Daily processing complete: %s", summary)
        return summary

    def _archive_file(self, source_key: str) -> None:
        """Move a processed file to the archive prefix.

        Args:
            source_key: S3 key of the processed file.
        """
        archive_key = source_key.replace("incoming/", "archive/", 1)

        try:
            self.s3_client.copy_object(
                Bucket=self.settings.aws.s3_input_bucket,
                CopySource={
                    "Bucket": self.settings.aws.s3_input_bucket,
                    "Key": source_key,
                },
                Key=archive_key,
            )
            self.s3_client.delete_object(
                Bucket=self.settings.aws.s3_input_bucket,
                Key=source_key,
            )
            logger.info("Archived %s -> %s", source_key, archive_key)
        except ClientError:
            logger.exception("Failed to archive file: %s", source_key)


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """AWS Lambda handler for daily FAERS processing.

    Triggered by CloudWatch Events (daily schedule) or S3 event
    notifications for new file uploads.

    Args:
        event: Lambda event payload (CloudWatch or S3 trigger).
        context: Lambda execution context.

    Returns:
        Processing summary with status code.
    """
    logger.info("Lambda handler invoked with event: %s", json.dumps(event, default=str))

    processor = DailyProcessor()

    try:
        processor.load_model()
    except Exception:
        logger.exception("Failed to load model in Lambda handler")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Model loading failed"}),
        }

    # Determine trigger type
    if "Records" in event:
        # S3 event notification trigger
        results: list[dict[str, Any]] = []
        for record in event["Records"]:
            s3_info = record.get("s3", {})
            bucket = s3_info.get("bucket", {}).get("name", "")
            key = s3_info.get("object", {}).get("key", "")

            if bucket and key:
                logger.info("Processing S3 event: s3://%s/%s", bucket, key)
                file_results = processor.process_file(key)
                results.extend(file_results)

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": f"Processed {len(results)} records from S3 events",
                    "record_count": len(results),
                }
            ),
        }

    # Scheduled trigger - process all new files
    prefix = event.get("prefix", "incoming/")
    summary = processor.process_new_files(prefix=prefix)

    status_code = 200 if not summary["errors"] else 207  # 207 = Multi-Status

    return {
        "statusCode": status_code,
        "body": json.dumps(summary, default=str),
    }
