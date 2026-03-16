"""AWS S3-based data ingestion pipeline for FDA FAERS data.

Downloads FAERS quarterly data from S3, parses XML/CSV files,
validates data quality, and stores processed records.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from xml.etree import ElementTree

import boto3
import pandas as pd
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from pydantic import BaseModel, field_validator
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from pharma_sentinel.config import AppSettings, get_settings

logger = logging.getLogger(__name__)


class FAERSRecord(BaseModel):
    """Validated FAERS adverse event record.

    Represents a single adverse event report from the FDA FAERS system
    with all required fields validated.
    """

    report_id: str
    case_id: str
    report_date: str
    patient_age: float | None = None
    patient_sex: str | None = None
    drug_name: str
    drug_indication: str | None = None
    reaction_description: str
    outcome: str | None = None
    reporter_type: str | None = None
    source_file: str = ""

    @field_validator("report_id", "case_id")
    @classmethod
    def validate_ids(cls, v: str) -> str:
        """Ensure IDs are non-empty strings."""
        if not v or not v.strip():
            raise ValueError("ID fields cannot be empty")
        return v.strip()

    @field_validator("drug_name")
    @classmethod
    def validate_drug_name(cls, v: str) -> str:
        """Ensure drug name is present and reasonable."""
        if not v or not v.strip():
            raise ValueError("Drug name cannot be empty")
        cleaned = v.strip()
        if len(cleaned) > 500:
            raise ValueError("Drug name exceeds maximum length of 500 characters")
        return cleaned

    @field_validator("reaction_description")
    @classmethod
    def validate_reaction(cls, v: str) -> str:
        """Ensure reaction description is present."""
        if not v or not v.strip():
            raise ValueError("Reaction description cannot be empty")
        return v.strip()

    @field_validator("patient_age")
    @classmethod
    def validate_age(cls, v: float | None) -> float | None:
        """Validate patient age is within reasonable range."""
        if v is not None and (v < 0 or v > 150):
            logger.warning("Suspicious patient age: %.1f, setting to None", v)
            return None
        return v

    @field_validator("patient_sex")
    @classmethod
    def validate_sex(cls, v: str | None) -> str | None:
        """Normalize patient sex field."""
        if v is None:
            return None
        normalized = v.strip().upper()
        if normalized in ("M", "MALE"):
            return "M"
        if normalized in ("F", "FEMALE"):
            return "F"
        return "UNK"


@dataclass
class DataQualityReport:
    """Report on data quality metrics after ingestion."""

    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    missing_drug_name: int = 0
    missing_reaction: int = 0
    duplicate_report_ids: int = 0
    files_processed: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def validity_rate(self) -> float:
        """Calculate the percentage of valid records."""
        if self.total_records == 0:
            return 0.0
        return (self.valid_records / self.total_records) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "validity_rate": round(self.validity_rate, 2),
            "missing_drug_name": self.missing_drug_name,
            "missing_reaction": self.missing_reaction,
            "duplicate_report_ids": self.duplicate_report_ids,
            "files_processed": self.files_processed,
            "errors": self.errors[:50],  # Limit error list size
        }


class FAERSDataIngester:
    """Ingests and processes FDA FAERS data from S3.

    Downloads FAERS quarterly data files (CSV and XML formats) from an
    S3 input bucket, parses and validates records, and stores processed
    data back to S3 as Parquet files for downstream consumption.

    Attributes:
        settings: Application configuration.
        s3_client: Boto3 S3 client with retry configuration.
        quality_report: Data quality metrics from the last ingestion run.
    """

    # FAERS CSV column mappings
    CSV_COLUMN_MAP: dict[str, str] = {
        "primaryid": "report_id",
        "caseid": "case_id",
        "event_dt": "report_date",
        "age": "patient_age",
        "sex": "patient_sex",
        "drugname": "drug_name",
        "drug_ind": "drug_indication",
        "pt": "reaction_description",
        "outc_cod": "outcome",
        "rept_cod": "reporter_type",
    }

    def __init__(self, settings: AppSettings | None = None) -> None:
        """Initialize the data ingester.

        Args:
            settings: Application settings. Uses default if None.
        """
        self.settings = settings or get_settings()
        self.quality_report = DataQualityReport()

        boto_config = BotoConfig(
            retries={"max_attempts": 3, "mode": "adaptive"},
            connect_timeout=10,
            read_timeout=30,
            max_pool_connections=25,
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
        logger.info(
            "FAERSDataIngester initialized for bucket: %s",
            self.settings.aws.s3_input_bucket,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(ClientError),
        before_sleep=lambda retry_state: logger.warning(
            "Retrying S3 operation (attempt %d): %s",
            retry_state.attempt_number,
            retry_state.outcome.exception() if retry_state.outcome else "unknown",
        ),
    )
    def list_input_files(self, prefix: str = "faers/") -> list[str]:
        """List available FAERS data files in the input bucket.

        Args:
            prefix: S3 key prefix to filter files.

        Returns:
            List of S3 object keys matching the prefix.
        """
        keys: list[str] = []
        paginator = self.s3_client.get_paginator("list_objects_v2")

        for page in paginator.paginate(
            Bucket=self.settings.aws.s3_input_bucket,
            Prefix=prefix,
        ):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith((".csv", ".xml", ".CSV", ".XML")):
                    keys.append(key)

        logger.info("Found %d FAERS files with prefix '%s'", len(keys), prefix)
        return keys

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(ClientError),
    )
    def download_file(self, key: str) -> bytes:
        """Download a file from the input S3 bucket.

        Args:
            key: S3 object key.

        Returns:
            File contents as bytes.

        Raises:
            ClientError: If the S3 download fails after retries.
        """
        logger.info("Downloading s3://%s/%s", self.settings.aws.s3_input_bucket, key)
        response = self.s3_client.get_object(
            Bucket=self.settings.aws.s3_input_bucket,
            Key=key,
        )
        return response["Body"].read()

    def parse_csv_file(self, data: bytes, source_file: str) -> list[FAERSRecord]:
        """Parse a FAERS CSV file into validated records.

        Args:
            data: Raw CSV file bytes.
            source_file: Source file name for provenance tracking.

        Returns:
            List of validated FAERSRecord instances.
        """
        records: list[FAERSRecord] = []
        seen_ids: set[str] = set()

        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            logger.exception("Failed to decode CSV file: %s", source_file)
            self.quality_report.errors.append(f"Decode error: {source_file}")
            return records

        reader = csv.DictReader(io.StringIO(text), delimiter="$")

        for row_num, row in enumerate(reader, start=1):
            self.quality_report.total_records += 1

            try:
                # Map CSV columns to record fields
                mapped: dict[str, Any] = {}
                for csv_col, record_field in self.CSV_COLUMN_MAP.items():
                    value = row.get(csv_col, "")
                    if value is None:
                        value = ""
                    mapped[record_field] = value.strip() if isinstance(value, str) else value

                # Handle numeric fields
                age_str = mapped.get("patient_age", "")
                if age_str and str(age_str).strip():
                    try:
                        mapped["patient_age"] = float(str(age_str))
                    except (ValueError, TypeError):
                        mapped["patient_age"] = None
                else:
                    mapped["patient_age"] = None

                mapped["source_file"] = source_file

                # Validate required fields before creating record
                if not mapped.get("drug_name"):
                    self.quality_report.missing_drug_name += 1
                    self.quality_report.invalid_records += 1
                    continue

                if not mapped.get("reaction_description"):
                    self.quality_report.missing_reaction += 1
                    self.quality_report.invalid_records += 1
                    continue

                # Deduplicate by report ID
                report_id = mapped.get("report_id", "")
                if report_id in seen_ids:
                    self.quality_report.duplicate_report_ids += 1
                    continue
                seen_ids.add(report_id)

                record = FAERSRecord(**mapped)
                records.append(record)
                self.quality_report.valid_records += 1

            except Exception as exc:
                self.quality_report.invalid_records += 1
                if row_num <= 10:  # Only log first few errors
                    logger.warning(
                        "Failed to parse row %d in %s: %s",
                        row_num,
                        source_file,
                        str(exc),
                    )
                self.quality_report.errors.append(f"Row {row_num} in {source_file}: {exc}")

        logger.info(
            "Parsed %d valid records from %s (%d total rows)",
            len(records),
            source_file,
            self.quality_report.total_records,
        )
        return records

    def parse_xml_file(self, data: bytes, source_file: str) -> list[FAERSRecord]:
        """Parse a FAERS XML file into validated records.

        Args:
            data: Raw XML file bytes.
            source_file: Source file name for provenance tracking.

        Returns:
            List of validated FAERSRecord instances.
        """
        records: list[FAERSRecord] = []

        try:
            root = ElementTree.fromstring(data)  # noqa: S314
        except ElementTree.ParseError:
            logger.exception("Failed to parse XML file: %s", source_file)
            self.quality_report.errors.append(f"XML parse error: {source_file}")
            return records

        # Handle FAERS XML format (ichicsr format)
        safety_reports = root.findall(".//safetyreport")
        if not safety_reports:
            # Try alternative structure
            safety_reports = root.findall(".//SafetyReport")

        for report_elem in safety_reports:
            self.quality_report.total_records += 1

            try:
                # Extract fields from XML elements
                report_id = self._xml_text(report_elem, "safetyreportid", "")
                case_id = self._xml_text(report_elem, "companynumb", report_id)

                # Patient information
                patient_elem = report_elem.find(".//patient")
                age_str = (
                    self._xml_text(patient_elem, "patientonsetage", "") if patient_elem else ""
                )
                sex = self._xml_text(patient_elem, "patientsex", None) if patient_elem else None

                # Drug information
                drug_elem = report_elem.find(".//drug")
                drug_name = self._xml_text(drug_elem, "medicinalproduct", "") if drug_elem else ""
                drug_indication = (
                    self._xml_text(drug_elem, "drugindication", None) if drug_elem else None
                )

                # Reaction information
                reaction_elem = report_elem.find(".//reaction")
                reaction = (
                    self._xml_text(reaction_elem, "reactionmeddrapt", "") if reaction_elem else ""
                )

                # Date
                report_date = self._xml_text(report_elem, "receiptdate", "")

                # Outcome
                outcome = self._xml_text(report_elem, "serious", None)

                if not drug_name:
                    self.quality_report.missing_drug_name += 1
                    self.quality_report.invalid_records += 1
                    continue

                if not reaction:
                    self.quality_report.missing_reaction += 1
                    self.quality_report.invalid_records += 1
                    continue

                patient_age: float | None = None
                if age_str:
                    try:
                        patient_age = float(age_str)
                    except (ValueError, TypeError):
                        patient_age = None

                record = FAERSRecord(
                    report_id=report_id or f"XML-{self.quality_report.total_records}",
                    case_id=case_id or report_id or "UNKNOWN",
                    report_date=report_date,
                    patient_age=patient_age,
                    patient_sex=sex,
                    drug_name=drug_name,
                    drug_indication=drug_indication,
                    reaction_description=reaction,
                    outcome=outcome,
                    source_file=source_file,
                )
                records.append(record)
                self.quality_report.valid_records += 1

            except Exception as exc:
                self.quality_report.invalid_records += 1
                logger.warning("Failed to parse XML report in %s: %s", source_file, exc)
                self.quality_report.errors.append(f"XML report in {source_file}: {exc}")

        logger.info("Parsed %d valid records from XML file %s", len(records), source_file)
        return records

    @staticmethod
    def _xml_text(
        element: ElementTree.Element | None,
        tag: str,
        default: str | None = None,
    ) -> str | None:
        """Safely extract text from an XML element.

        Args:
            element: Parent XML element.
            tag: Child tag name to find.
            default: Default value if element/tag not found.

        Returns:
            Text content of the child element, or default.
        """
        if element is None:
            return default
        child = element.find(tag)
        if child is not None and child.text:
            return child.text.strip()
        return default

    def store_processed_records(
        self,
        records: list[FAERSRecord],
        output_key: str,
    ) -> str:
        """Store processed records to S3 as Parquet.

        Args:
            records: List of validated FAERS records.
            output_key: S3 key for the output file.

        Returns:
            S3 URI of the stored file.

        Raises:
            ClientError: If the S3 upload fails.
        """
        if not records:
            logger.warning("No records to store for key: %s", output_key)
            return ""

        df = pd.DataFrame([r.model_dump() for r in records])

        # Write to Parquet in memory
        buffer = io.BytesIO()
        df.to_parquet(buffer, engine="pyarrow", index=False, compression="snappy")
        buffer.seek(0)

        s3_key = f"processed/{output_key}.parquet"

        self.s3_client.put_object(
            Bucket=self.settings.aws.s3_output_bucket,
            Key=s3_key,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream",
            ServerSideEncryption="aws:kms",
        )

        s3_uri = f"s3://{self.settings.aws.s3_output_bucket}/{s3_key}"
        logger.info("Stored %d records to %s", len(records), s3_uri)
        return s3_uri

    def ingest_quarterly_data(self, quarter_prefix: str) -> DataQualityReport:
        """Ingest a full quarter of FAERS data.

        Downloads all files for the specified quarter, parses them,
        validates records, and stores processed results in S3.

        Args:
            quarter_prefix: S3 prefix for the quarter data
                           (e.g., "faers/2024Q1/").

        Returns:
            DataQualityReport with ingestion metrics.
        """
        self.quality_report = DataQualityReport()
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")

        logger.info("Starting quarterly data ingestion for prefix: %s", quarter_prefix)

        try:
            files = self.list_input_files(prefix=quarter_prefix)
        except ClientError:
            logger.exception("Failed to list files for prefix: %s", quarter_prefix)
            self.quality_report.errors.append(f"Failed to list files: {quarter_prefix}")
            return self.quality_report

        all_records: list[FAERSRecord] = []

        for file_key in files:
            try:
                data = self.download_file(file_key)
                self.quality_report.files_processed += 1

                if file_key.lower().endswith(".csv"):
                    records = self.parse_csv_file(data, file_key)
                elif file_key.lower().endswith(".xml"):
                    records = self.parse_xml_file(data, file_key)
                else:
                    logger.warning("Unsupported file format: %s", file_key)
                    continue

                all_records.extend(records)

            except ClientError:
                logger.exception("Failed to download file: %s", file_key)
                self.quality_report.errors.append(f"Download failed: {file_key}")
            except Exception:
                logger.exception("Unexpected error processing file: %s", file_key)
                self.quality_report.errors.append(f"Processing failed: {file_key}")

        # Store all processed records
        if all_records:
            quarter_name = quarter_prefix.rstrip("/").split("/")[-1]
            output_key = f"{quarter_name}/{timestamp}"
            self.store_processed_records(all_records, output_key)

        logger.info(
            "Quarterly ingestion complete: %s",
            self.quality_report.to_dict(),
        )

        return self.quality_report

    def validate_data_quality(
        self,
        records: list[FAERSRecord],
        min_validity_rate: float = 80.0,
    ) -> bool:
        """Validate that ingested data meets quality thresholds.

        Args:
            records: Processed records to validate.
            min_validity_rate: Minimum acceptable validity rate (percent).

        Returns:
            True if data quality meets thresholds, False otherwise.
        """
        if not records:
            logger.warning("No records to validate")
            return False

        validity_rate = self.quality_report.validity_rate

        if validity_rate < min_validity_rate:
            logger.warning(
                "Data quality below threshold: %.1f%% < %.1f%%",
                validity_rate,
                min_validity_rate,
            )
            return False

        # Check for excessive duplicates
        duplicate_rate = (
            self.quality_report.duplicate_report_ids / max(self.quality_report.total_records, 1)
        ) * 100
        if duplicate_rate > 20.0:
            logger.warning("High duplicate rate: %.1f%%", duplicate_rate)
            return False

        logger.info(
            "Data quality validation passed: %.1f%% validity, %.1f%% duplicates",
            validity_rate,
            duplicate_rate,
        )
        return True
