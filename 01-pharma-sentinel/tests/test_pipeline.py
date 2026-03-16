"""Tests for the data ingestion and processing pipelines."""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from pharma_sentinel.config import AppSettings
from pharma_sentinel.pipeline.daily_processor import NLPPreprocessor
from pharma_sentinel.pipeline.data_ingestion import (
    DataQualityReport,
    FAERSDataIngester,
    FAERSRecord,
)

# ─────────────────────────────────────────────────────────────────────
# FAERSRecord validation tests
# ─────────────────────────────────────────────────────────────────────


class TestFAERSRecord:
    """Tests for FAERS record validation."""

    def test_valid_record(self) -> None:
        """Test creating a valid FAERS record."""
        record = FAERSRecord(
            report_id="R001",
            case_id="C001",
            report_date="20240101",
            patient_age=55.0,
            patient_sex="F",
            drug_name="Ibuprofen",
            drug_indication="Pain",
            reaction_description="Nausea and headache",
            outcome="DE",
        )
        assert record.report_id == "R001"
        assert record.drug_name == "Ibuprofen"
        assert record.patient_sex == "F"

    def test_empty_report_id_raises(self) -> None:
        """Test that empty report ID raises validation error."""
        with pytest.raises(ValueError):
            FAERSRecord(
                report_id="",
                case_id="C001",
                report_date="20240101",
                drug_name="Aspirin",
                reaction_description="Headache",
            )

    def test_empty_drug_name_raises(self) -> None:
        """Test that empty drug name raises validation error."""
        with pytest.raises(ValueError):
            FAERSRecord(
                report_id="R001",
                case_id="C001",
                report_date="20240101",
                drug_name="",
                reaction_description="Headache",
            )

    def test_empty_reaction_raises(self) -> None:
        """Test that empty reaction description raises validation error."""
        with pytest.raises(ValueError):
            FAERSRecord(
                report_id="R001",
                case_id="C001",
                report_date="20240101",
                drug_name="Aspirin",
                reaction_description="",
            )

    def test_age_validation(self) -> None:
        """Test that unreasonable age is set to None."""
        record = FAERSRecord(
            report_id="R001",
            case_id="C001",
            report_date="20240101",
            patient_age=200.0,
            drug_name="Aspirin",
            reaction_description="Headache",
        )
        assert record.patient_age is None

    def test_sex_normalization(self) -> None:
        """Test that sex field is normalized."""
        for raw, expected in [("male", "M"), ("Female", "F"), ("Other", "UNK"), ("M", "M")]:
            record = FAERSRecord(
                report_id="R001",
                case_id="C001",
                report_date="20240101",
                patient_sex=raw,
                drug_name="Aspirin",
                reaction_description="Headache",
            )
            assert record.patient_sex == expected

    def test_drug_name_max_length(self) -> None:
        """Test that excessively long drug names are rejected."""
        with pytest.raises(ValueError):
            FAERSRecord(
                report_id="R001",
                case_id="C001",
                report_date="20240101",
                drug_name="x" * 501,
                reaction_description="Headache",
            )


# ─────────────────────────────────────────────────────────────────────
# DataQualityReport tests
# ─────────────────────────────────────────────────────────────────────


class TestDataQualityReport:
    """Tests for the data quality report."""

    def test_validity_rate_calculation(self) -> None:
        """Test validity rate calculation."""
        report = DataQualityReport(total_records=100, valid_records=80)
        assert report.validity_rate == 80.0

    def test_validity_rate_zero_records(self) -> None:
        """Test validity rate with zero records."""
        report = DataQualityReport(total_records=0)
        assert report.validity_rate == 0.0

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        report = DataQualityReport(
            total_records=100,
            valid_records=80,
            invalid_records=20,
            files_processed=5,
        )
        d = report.to_dict()
        assert d["total_records"] == 100
        assert d["validity_rate"] == 80.0
        assert d["files_processed"] == 5


# ─────────────────────────────────────────────────────────────────────
# FAERSDataIngester tests
# ─────────────────────────────────────────────────────────────────────


class TestFAERSDataIngester:
    """Tests for the FAERS data ingestion pipeline."""

    @mock_aws
    def test_list_input_files(self, test_settings: AppSettings) -> None:
        """Test listing input files from S3."""
        # Create mock S3 bucket and files
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=test_settings.aws.s3_input_bucket)
        s3.put_object(
            Bucket=test_settings.aws.s3_input_bucket,
            Key="faers/2024Q1/DRUG24Q1.csv",
            Body=b"data",
        )
        s3.put_object(
            Bucket=test_settings.aws.s3_input_bucket,
            Key="faers/2024Q1/REAC24Q1.csv",
            Body=b"data",
        )
        s3.put_object(
            Bucket=test_settings.aws.s3_input_bucket,
            Key="faers/2024Q1/README.txt",
            Body=b"readme",
        )

        ingester = FAERSDataIngester(test_settings)
        files = ingester.list_input_files(prefix="faers/2024Q1/")

        assert len(files) == 2
        assert all(f.endswith(".csv") for f in files)

    def test_parse_csv_file(self, test_settings: AppSettings, sample_csv_data: bytes) -> None:
        """Test parsing a FAERS CSV file."""
        ingester = FAERSDataIngester(test_settings)
        records = ingester.parse_csv_file(sample_csv_data, "test.csv")

        assert len(records) == 3
        assert records[0].report_id == "100001"
        assert records[0].drug_name == "Ibuprofen"
        assert records[0].patient_sex == "F"
        assert records[0].patient_age == 55.0

    def test_parse_csv_handles_missing_fields(self, test_settings: AppSettings) -> None:
        """Test CSV parsing handles rows with missing required fields."""
        csv_data = (
            b"primaryid$caseid$event_dt$age$sex$drugname$drug_ind$pt$outc_cod$rept_cod\n"
            b"1$C1$20240101$55$F$$Pain$Nausea$DE$HP\n"  # Missing drug name
            b"2$C2$20240102$72$M$Aspirin$$$OT$HP\n"  # Missing reaction
        )

        ingester = FAERSDataIngester(test_settings)
        records = ingester.parse_csv_file(csv_data, "test.csv")

        assert len(records) == 0
        assert ingester.quality_report.missing_drug_name == 1
        assert ingester.quality_report.missing_reaction == 1

    def test_parse_csv_deduplication(self, test_settings: AppSettings) -> None:
        """Test CSV parsing deduplicates by report ID."""
        csv_data = (
            b"primaryid$caseid$event_dt$age$sex$drugname$drug_ind$pt$outc_cod$rept_cod\n"
            b"1$C1$20240101$55$F$Aspirin$Pain$Nausea$DE$HP\n"
            b"1$C1$20240101$55$F$Aspirin$Pain$Nausea$DE$HP\n"  # Duplicate
        )

        ingester = FAERSDataIngester(test_settings)
        records = ingester.parse_csv_file(csv_data, "test.csv")

        assert len(records) == 1
        assert ingester.quality_report.duplicate_report_ids == 1

    def test_parse_xml_file(self, test_settings: AppSettings, sample_xml_data: bytes) -> None:
        """Test parsing a FAERS XML file."""
        ingester = FAERSDataIngester(test_settings)
        records = ingester.parse_xml_file(sample_xml_data, "test.xml")

        assert len(records) == 2
        assert records[0].report_id == "XML001"
        assert records[0].drug_name == "Warfarin"
        assert records[0].patient_age == 65.0

    def test_parse_xml_invalid(self, test_settings: AppSettings) -> None:
        """Test XML parsing handles invalid XML gracefully."""
        ingester = FAERSDataIngester(test_settings)
        records = ingester.parse_xml_file(b"not xml data", "bad.xml")

        assert len(records) == 0
        assert len(ingester.quality_report.errors) > 0

    @mock_aws
    def test_store_processed_records(self, test_settings: AppSettings) -> None:
        """Test storing processed records to S3."""
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=test_settings.aws.s3_output_bucket)

        ingester = FAERSDataIngester(test_settings)

        records = [
            FAERSRecord(
                report_id="R001",
                case_id="C001",
                report_date="20240101",
                drug_name="Aspirin",
                reaction_description="Headache",
            ),
        ]

        result = ingester.store_processed_records(records, "test/output")
        assert result.startswith("s3://")
        assert "processed/" in result

    def test_store_empty_records(self, test_settings: AppSettings) -> None:
        """Test storing empty records returns empty string."""
        ingester = FAERSDataIngester(test_settings)
        result = ingester.store_processed_records([], "test/output")
        assert result == ""

    def test_validate_data_quality_pass(self, test_settings: AppSettings) -> None:
        """Test data quality validation passes with good data."""
        ingester = FAERSDataIngester(test_settings)
        ingester.quality_report = DataQualityReport(
            total_records=100,
            valid_records=90,
            invalid_records=10,
            duplicate_report_ids=2,
        )

        records = [
            FAERSRecord(
                report_id=f"R{i}",
                case_id=f"C{i}",
                report_date="20240101",
                drug_name="Aspirin",
                reaction_description="Headache",
            )
            for i in range(90)
        ]

        assert ingester.validate_data_quality(records) is True

    def test_validate_data_quality_fail(self, test_settings: AppSettings) -> None:
        """Test data quality validation fails with low validity rate."""
        ingester = FAERSDataIngester(test_settings)
        ingester.quality_report = DataQualityReport(
            total_records=100,
            valid_records=50,
            invalid_records=50,
        )

        records = [
            FAERSRecord(
                report_id="R001",
                case_id="C001",
                report_date="20240101",
                drug_name="Aspirin",
                reaction_description="Headache",
            ),
        ]

        assert ingester.validate_data_quality(records) is False


# ─────────────────────────────────────────────────────────────────────
# NLPPreprocessor tests
# ─────────────────────────────────────────────────────────────────────


class TestNLPPreprocessor:
    """Tests for the NLP preprocessing component."""

    def test_tokenize(self) -> None:
        """Test basic tokenization."""
        nlp = NLPPreprocessor()
        tokens = nlp.tokenize("Patient had severe nausea and vomiting")
        assert "patient" in tokens
        assert "severe" in tokens
        assert "nausea" in tokens
        # Stopwords should be removed
        assert "and" not in tokens
        assert "had" not in tokens

    def test_tokenize_empty(self) -> None:
        """Test tokenization of empty string."""
        nlp = NLPPreprocessor()
        assert nlp.tokenize("") == []
        assert nlp.tokenize("   ") == []

    def test_extract_drug_names(self) -> None:
        """Test drug name extraction from text."""
        nlp = NLPPreprocessor()
        drugs = nlp.extract_drug_names(
            "Patient was taking Aspirin and Warfarin for heart condition"
        )
        assert "aspirin" in drugs
        assert "warfarin" in drugs

    def test_extract_adverse_events(self) -> None:
        """Test adverse event term extraction."""
        nlp = NLPPreprocessor()
        events = nlp.extract_adverse_events("Patient experienced nausea, rash, and seizure")
        assert "nausea" in events
        assert "rash" in events
        assert "seizure" in events

    def test_preprocess_report(self) -> None:
        """Test full report preprocessing."""
        nlp = NLPPreprocessor()
        record = FAERSRecord(
            report_id="R001",
            case_id="C001",
            report_date="20240101",
            patient_age=55.0,
            patient_sex="F",
            drug_name="Ibuprofen",
            drug_indication="Pain management",
            reaction_description="Patient experienced nausea and severe headache",
        )

        result = nlp.preprocess_report(record)

        assert result["report_id"] == "R001"
        assert "ibuprofen" in result["extracted_drugs"]
        assert result["token_count"] > 0
        assert "nausea" in result["extracted_adverse_events"]
        assert result["patient_age"] == 55.0
        assert result["patient_sex"] == "F"

    def test_preprocess_report_without_indication(self) -> None:
        """Test preprocessing when drug indication is None."""
        nlp = NLPPreprocessor()
        record = FAERSRecord(
            report_id="R001",
            case_id="C001",
            report_date="20240101",
            drug_name="Aspirin",
            reaction_description="Mild headache",
        )

        result = nlp.preprocess_report(record)
        assert result["drug_name"] == "Aspirin"
        assert "aspirin" in result["extracted_drugs"]
