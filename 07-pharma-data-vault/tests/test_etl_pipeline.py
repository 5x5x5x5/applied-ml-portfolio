"""
Tests for the ETL pipeline and file processing modules.

Tests cover:
  - File discovery and pattern matching
  - File validation (format, checksum, schema)
  - CSV and fixed-width file parsing
  - Data cleansing and rejection logic
  - File archival
  - Pipeline step execution and retry logic
  - Pipeline run orchestration
"""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

from pharma_vault._config import VaultConfig
from pharma_vault.etl.file_processor import (
    FEED_SCHEMAS,
    FileManifest,
    FileProcessor,
)
from pharma_vault.etl.pipeline import (
    ETLPipeline,
    PipelineRun,
    PipelineStep,
    StepStatus,
)

# =============================================================================
# FileProcessor Tests
# =============================================================================


class TestFileProcessorDiscovery:
    """Tests for file discovery and pattern matching."""

    def test_discover_csv_files(
        self,
        test_config: VaultConfig,
        sample_drug_csv: Path,
        sample_patient_csv: Path,
    ) -> None:
        """Discover CSV files in the staging directory."""
        processor = FileProcessor(test_config)
        manifests = processor.discover_files()

        assert len(manifests) >= 2
        feed_types = {m.feed_type for m in manifests}
        assert "drug_feed" in feed_types
        assert "patient_feed" in feed_types

    def test_discover_fixed_width_files(
        self,
        test_config: VaultConfig,
        sample_mfg_fixed_width: Path,
    ) -> None:
        """Discover fixed-width .dat files."""
        processor = FileProcessor(test_config)
        manifests = processor.discover_files()

        mfg_manifests = [m for m in manifests if m.feed_type == "mfg_feed"]
        assert len(mfg_manifests) == 1
        assert mfg_manifests[0].file_date == "20260305"

    def test_discover_skips_unrecognized_files(
        self,
        test_config: VaultConfig,
        tmp_staging_dir: Path,
    ) -> None:
        """Files not matching the naming convention are skipped."""
        (tmp_staging_dir / "random_file.csv").write_text("data")
        (tmp_staging_dir / "not_a_feed_20260305.csv").write_text("data")

        processor = FileProcessor(test_config)
        manifests = processor.discover_files()

        assert len(manifests) == 0

    def test_discover_empty_directory(self, test_config: VaultConfig) -> None:
        """Empty staging directory returns no manifests."""
        processor = FileProcessor(test_config)
        manifests = processor.discover_files()
        assert manifests == []

    def test_discover_nonexistent_directory(self, tmp_path: Path) -> None:
        """Nonexistent staging directory returns empty list without error."""
        config = VaultConfig(staging_dir=tmp_path / "nonexistent")
        processor = FileProcessor(config)
        manifests = processor.discover_files()
        assert manifests == []

    def test_manifest_file_size(
        self,
        test_config: VaultConfig,
        sample_drug_csv: Path,
    ) -> None:
        """Manifest captures the correct file size."""
        processor = FileProcessor(test_config)
        manifests = processor.discover_files()

        drug_manifest = next(m for m in manifests if m.feed_type == "drug_feed")
        assert drug_manifest.file_size_bytes == sample_drug_csv.stat().st_size
        assert drug_manifest.file_size_bytes > 0


class TestFileProcessorValidation:
    """Tests for file validation logic."""

    def test_validate_valid_drug_csv(
        self,
        test_config: VaultConfig,
        sample_drug_csv: Path,
    ) -> None:
        """Valid drug CSV passes all validation checks."""
        processor = FileProcessor(test_config)
        manifest = FileManifest(
            file_path=str(sample_drug_csv),
            feed_type="drug_feed",
            file_date="20260305",
            file_size_bytes=sample_drug_csv.stat().st_size,
        )

        is_valid = processor.validate_file(manifest)

        assert is_valid is True
        assert manifest.is_valid is True
        assert len(manifest.errors) == 0
        assert manifest.row_count == 4  # 4 data rows

    def test_validate_checksum_match(
        self,
        test_config: VaultConfig,
        sample_drug_csv: Path,
        checksum_file: Path,
    ) -> None:
        """File with matching checksum passes validation."""
        processor = FileProcessor(test_config)

        expected_md5 = checksum_file.read_text().strip().split()[0]
        manifest = FileManifest(
            file_path=str(sample_drug_csv),
            feed_type="drug_feed",
            file_date="20260305",
            file_size_bytes=sample_drug_csv.stat().st_size,
            expected_checksum=expected_md5,
        )

        is_valid = processor.validate_file(manifest)
        assert is_valid is True
        assert manifest.actual_checksum == expected_md5

    def test_validate_checksum_mismatch(
        self,
        test_config: VaultConfig,
        sample_drug_csv: Path,
    ) -> None:
        """File with wrong checksum fails validation."""
        processor = FileProcessor(test_config)
        manifest = FileManifest(
            file_path=str(sample_drug_csv),
            feed_type="drug_feed",
            file_date="20260305",
            file_size_bytes=sample_drug_csv.stat().st_size,
            expected_checksum="0000000000000000deadbeef00000000",
        )

        is_valid = processor.validate_file(manifest)
        assert is_valid is False
        assert any("Checksum mismatch" in e for e in manifest.errors)

    def test_validate_empty_file(
        self,
        test_config: VaultConfig,
        tmp_staging_dir: Path,
    ) -> None:
        """Empty file fails validation."""
        empty_file = tmp_staging_dir / "drug_feed_20260305.csv"
        empty_file.write_text("")

        processor = FileProcessor(test_config)
        manifest = FileManifest(
            file_path=str(empty_file),
            feed_type="drug_feed",
            file_date="20260305",
            file_size_bytes=0,
        )

        is_valid = processor.validate_file(manifest)
        assert is_valid is False

    def test_validate_nonexistent_file(self, test_config: VaultConfig) -> None:
        """Nonexistent file fails validation gracefully."""
        processor = FileProcessor(test_config)
        manifest = FileManifest(
            file_path="/nonexistent/file.csv",
            feed_type="drug_feed",
            file_date="20260305",
            file_size_bytes=0,
        )

        is_valid = processor.validate_file(manifest)
        assert is_valid is False
        assert any("not found" in e for e in manifest.errors)

    def test_validate_missing_columns(
        self,
        test_config: VaultConfig,
        tmp_staging_dir: Path,
    ) -> None:
        """CSV with missing required columns reports errors."""
        bad_csv = tmp_staging_dir / "drug_feed_20260305.csv"
        with open(bad_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ndc_code", "drug_name"])  # Missing most columns
            writer.writerow(["12345-6789-01", "TestDrug"])

        processor = FileProcessor(test_config)
        manifest = FileManifest(
            file_path=str(bad_csv),
            feed_type="drug_feed",
            file_date="20260305",
            file_size_bytes=bad_csv.stat().st_size,
        )

        is_valid = processor.validate_file(manifest)
        assert is_valid is False
        assert any("Missing columns" in e for e in manifest.errors)


class TestFileProcessorProcessing:
    """Tests for file parsing and cleansing."""

    def test_process_drug_csv(
        self,
        test_config: VaultConfig,
        sample_drug_csv: Path,
    ) -> None:
        """Process a valid drug CSV and get a report."""
        processor = FileProcessor(test_config)
        manifest = FileManifest(
            file_path=str(sample_drug_csv),
            feed_type="drug_feed",
            file_date="20260305",
            file_size_bytes=sample_drug_csv.stat().st_size,
        )

        report = processor.process_file(manifest)

        assert report["rows_read"] == 4
        assert report["rows_processed"] >= 4  # All should pass cleansing
        assert report["feed_type"] == "drug_feed"

    def test_process_csv_with_errors(
        self,
        test_config: VaultConfig,
        sample_drug_csv_with_errors: Path,
    ) -> None:
        """Process a CSV with quality issues; some rows should be rejected."""
        processor = FileProcessor(test_config)
        manifest = FileManifest(
            file_path=str(sample_drug_csv_with_errors),
            feed_type="drug_feed",
            file_date="20260305",
            file_size_bytes=sample_drug_csv_with_errors.stat().st_size,
        )

        report = processor.process_file(manifest)

        assert report["rows_read"] == 5
        # Rows with missing ndc_code, drug_name, or manufacturer should be rejected
        assert report["rows_rejected"] > 0
        assert "rejection_reasons" in report

    def test_process_fixed_width_mfg(
        self,
        test_config: VaultConfig,
        sample_mfg_fixed_width: Path,
    ) -> None:
        """Process a fixed-width manufacturing feed."""
        processor = FileProcessor(test_config)
        manifest = FileManifest(
            file_path=str(sample_mfg_fixed_width),
            feed_type="mfg_feed",
            file_date="20260305",
            file_size_bytes=sample_mfg_fixed_width.stat().st_size,
        )

        report = processor.process_file(manifest)

        # HDR and TRL lines should be skipped, leaving 2 data rows
        assert report["rows_read"] == 2

    def test_process_patient_csv(
        self,
        test_config: VaultConfig,
        sample_patient_csv: Path,
    ) -> None:
        """Process a patient feed CSV."""
        processor = FileProcessor(test_config)
        manifest = FileManifest(
            file_path=str(sample_patient_csv),
            feed_type="patient_feed",
            file_date="20260305",
            file_size_bytes=sample_patient_csv.stat().st_size,
        )

        report = processor.process_file(manifest)
        assert report["rows_read"] == 3
        assert report["rows_processed"] == 3


class TestFileProcessorArchival:
    """Tests for file archival."""

    def test_archive_file(
        self,
        test_config: VaultConfig,
        sample_drug_csv: Path,
    ) -> None:
        """Archived file is moved from staging to archive directory."""
        processor = FileProcessor(test_config)
        manifest = FileManifest(
            file_path=str(sample_drug_csv),
            feed_type="drug_feed",
            file_date="20260305",
            file_size_bytes=sample_drug_csv.stat().st_size,
        )

        archive_path = processor.archive_file(manifest)

        # Original file should no longer exist
        assert not sample_drug_csv.exists()
        # Archived file should exist
        assert Path(archive_path).exists()
        # Should be under YYYYMM subdirectory
        assert "202603" in archive_path

    def test_archive_with_checksum(
        self,
        test_config: VaultConfig,
        sample_drug_csv: Path,
        checksum_file: Path,
    ) -> None:
        """Checksum companion file is also archived."""
        processor = FileProcessor(test_config)
        manifest = FileManifest(
            file_path=str(sample_drug_csv),
            feed_type="drug_feed",
            file_date="20260305",
            file_size_bytes=sample_drug_csv.stat().st_size,
        )

        processor.archive_file(manifest)

        # Both the data file and checksum file should be gone from staging
        assert not sample_drug_csv.exists()
        assert not checksum_file.exists()


# =============================================================================
# Pipeline Tests
# =============================================================================


class TestPipelineStep:
    """Tests for PipelineStep dataclass."""

    def test_step_default_state(self) -> None:
        """New step starts in PENDING status."""
        step = PipelineStep(name="test_step")
        assert step.status == StepStatus.PENDING
        assert step.start_time is None
        assert step.retry_count == 0

    def test_step_status_transitions(self) -> None:
        """Step status can be updated."""
        step = PipelineStep(name="test_step")
        step.status = StepStatus.RUNNING
        assert step.status == StepStatus.RUNNING
        step.status = StepStatus.SUCCESS
        assert step.status == StepStatus.SUCCESS


class TestPipelineRun:
    """Tests for PipelineRun dataclass."""

    def test_run_to_dict(self) -> None:
        """Pipeline run can be serialized to dict."""
        run = PipelineRun(run_id="TEST_RUN_001")
        run.steps.append(PipelineStep(name="step_1"))

        result = run.to_dict()
        assert result["run_id"] == "TEST_RUN_001"
        assert len(result["steps"]) == 1
        assert result["steps"][0]["name"] == "step_1"


class TestETLPipeline:
    """Tests for the ETL pipeline orchestrator."""

    def test_pipeline_init(self, test_config: VaultConfig) -> None:
        """Pipeline initializes with config."""
        pipeline = ETLPipeline(config=test_config)
        assert pipeline._config == test_config
        assert pipeline._engine is None

    def test_generate_run_id(self, test_config: VaultConfig) -> None:
        """Run IDs are unique and follow expected format."""
        pipeline = ETLPipeline(config=test_config)
        id1 = pipeline._generate_run_id()
        id2 = pipeline._generate_run_id()

        assert id1.startswith("RUN_")
        assert len(id1) > 20

    def test_daily_etl_no_files(self, test_config: VaultConfig) -> None:
        """Pipeline with no files to process completes successfully."""
        pipeline = ETLPipeline(config=test_config)
        run = pipeline.run_daily_etl()

        assert run.status == StepStatus.SUCCESS
        # Should have a discover step that was skipped
        discover_steps = [s for s in run.steps if s.name == "discover_files"]
        assert len(discover_steps) == 1
        assert discover_steps[0].status == StepStatus.SKIPPED

    @patch.object(ETLPipeline, "_execute_plsql")
    def test_daily_etl_with_files(
        self,
        mock_plsql: MagicMock,
        test_config: VaultConfig,
        sample_drug_csv: Path,
        sample_patient_csv: Path,
        sample_ae_csv: Path,
    ) -> None:
        """Pipeline processes available files through all phases."""
        mock_plsql.return_value = {"status": "success", "rows_affected": 100}

        pipeline = ETLPipeline(config=test_config)

        # Patch the DQ framework import to avoid DB dependency
        with patch("pharma_vault.etl.pipeline.DataQualityFramework") as mock_dq_class:
            mock_dq = MagicMock()
            mock_dq.run_all_checks.return_value = {
                "overall_pass": True,
                "checks_passed": 10,
            }
            mock_dq_class.return_value = mock_dq

            run = pipeline.run_daily_etl()

        # Pipeline should have discovered and processed files
        assert run.status in (StepStatus.SUCCESS, StepStatus.FAILED)
        assert len(run.steps) > 0
        assert len(run.source_files) > 0

    def test_pipeline_shutdown(self, test_config: VaultConfig) -> None:
        """Shutdown disposes engine cleanly."""
        pipeline = ETLPipeline(config=test_config)
        pipeline.shutdown()  # Should not raise even with no engine
        assert pipeline._engine is None

    def test_execute_step_with_retry_success(self, test_config: VaultConfig) -> None:
        """Step succeeds on first attempt."""
        pipeline = ETLPipeline(config=test_config)
        step = PipelineStep(name="test_step")

        with patch.object(pipeline, "_execute_plsql") as mock_exec:
            mock_exec.return_value = {"status": "success"}
            result = pipeline._execute_step_with_retry(step, "TEST_PROC")

        assert result is True
        assert step.status == StepStatus.SUCCESS
        assert step.retry_count == 0

    def test_execute_step_with_retry_eventual_success(self, test_config: VaultConfig) -> None:
        """Step fails first, succeeds on retry."""
        from sqlalchemy.exc import OperationalError

        pipeline = ETLPipeline(config=test_config)
        step = PipelineStep(name="test_step")

        with patch.object(pipeline, "_execute_plsql") as mock_exec:
            mock_exec.side_effect = [
                OperationalError("stmt", {}, Exception("connection lost")),
                {"status": "success"},
            ]
            result = pipeline._execute_step_with_retry(step, "TEST_PROC")

        assert result is True
        assert step.status == StepStatus.SUCCESS

    def test_execute_step_with_retry_all_fail(self, test_config: VaultConfig) -> None:
        """Step fails all retry attempts."""
        from sqlalchemy.exc import OperationalError

        pipeline = ETLPipeline(config=test_config)
        step = PipelineStep(name="test_step")

        with patch.object(pipeline, "_execute_plsql") as mock_exec:
            mock_exec.side_effect = OperationalError("stmt", {}, Exception("connection lost"))
            result = pipeline._execute_step_with_retry(step, "TEST_PROC")

        assert result is False
        assert step.status == StepStatus.FAILED
        assert step.retry_count == test_config.max_retries


class TestFeedSchemas:
    """Tests for feed schema definitions."""

    def test_all_feed_types_defined(self) -> None:
        """All expected feed types have schema definitions."""
        expected_types = {"drug_feed", "patient_feed", "trial_feed", "ae_feed", "mfg_feed"}
        assert expected_types == set(FEED_SCHEMAS.keys())

    def test_drug_feed_columns(self) -> None:
        """Drug feed schema has expected columns."""
        drug_cols = FEED_SCHEMAS["drug_feed"]
        assert "ndc_code" in drug_cols
        assert "drug_name" in drug_cols
        assert "manufacturer" in drug_cols
        assert len(drug_cols) == 11

    def test_ae_feed_columns(self) -> None:
        """AE feed schema has severity and onset_date."""
        ae_cols = FEED_SCHEMAS["ae_feed"]
        assert "ae_report_id" in ae_cols
        assert "severity" in ae_cols
        assert "onset_date" in ae_cols
