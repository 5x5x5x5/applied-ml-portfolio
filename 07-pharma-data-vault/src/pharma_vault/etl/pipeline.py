"""
ETL Pipeline Orchestrator for PharmaDataVault.

Manages the end-to-end ETL workflow:
  1. Watch for incoming data files in the staging directory
  2. Validate and process files via file_processor module
  3. Call PL/SQL staging, loading, and mart procedures via SQLAlchemy
  4. Run data quality checks with Great Expectations
  5. Track pipeline state and handle retries for failed steps

The pipeline is designed to be called by Control-M or cron, and also
supports a file-watcher daemon mode for event-driven processing.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from pharma_vault._config import VaultConfig
from pharma_vault.etl.file_processor import FileProcessor

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of an individual pipeline step."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class PipelineStep:
    """Tracks the execution state of a single pipeline step."""

    name: str
    status: StepStatus = StepStatus.PENDING
    start_time: datetime | None = None
    end_time: datetime | None = None
    rows_affected: int = 0
    error_message: str | None = None
    retry_count: int = 0


@dataclass
class PipelineRun:
    """Represents a full pipeline execution run."""

    run_id: str
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    status: StepStatus = StepStatus.PENDING
    steps: list[PipelineStep] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    batch_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the pipeline run to a dictionary for reporting."""
        return {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status.value,
            "source_files": self.source_files,
            "steps": [
                {
                    "name": s.name,
                    "status": s.status.value,
                    "rows_affected": s.rows_affected,
                    "duration_seconds": (
                        (s.end_time - s.start_time).total_seconds()
                        if s.start_time and s.end_time
                        else None
                    ),
                    "error": s.error_message,
                    "retries": s.retry_count,
                }
                for s in self.steps
            ],
        }


class ETLPipeline:
    """
    Orchestrates the full ETL pipeline for pharmaceutical data.

    Manages file ingestion, PL/SQL procedure execution, data quality
    validation, and pipeline state tracking. Supports both batch mode
    (called from scheduler) and daemon mode (file watcher).
    """

    # PL/SQL procedures to call in order
    STAGING_PROCEDURES: list[tuple[str, str]] = [
        ("stage_drug", "SP_STAGE_DRUG_DATA"),
        ("cleanse_drug", "SP_CLEANSE_DRUG_DATA"),
        ("validate_drug", "SP_VALIDATE_DRUG_DATA"),
    ]

    HUB_LOAD_PROCEDURES: list[tuple[str, str]] = [
        ("load_hub_drug", "PKG_LOAD_HUB_DRUG.LOAD_FROM_BATCH"),
        ("verify_hub_drug", "PKG_LOAD_HUB_DRUG.VERIFY_HUB_INTEGRITY"),
    ]

    SATELLITE_PROCEDURES: list[tuple[str, str]] = [
        ("load_sat_drug_details", "PKG_LOAD_SATELLITES.LOAD_SAT_DRUG_DETAILS"),
        ("load_sat_patient_demo", "PKG_LOAD_SATELLITES.LOAD_SAT_PATIENT_DEMOGRAPHICS"),
        ("load_sat_trial_details", "PKG_LOAD_SATELLITES.LOAD_SAT_CLINICAL_TRIAL_DETAILS"),
        ("load_sat_ae_details", "PKG_LOAD_SATELLITES.LOAD_SAT_ADVERSE_EVENT_DETAILS"),
        ("load_sat_mfg", "PKG_LOAD_SATELLITES.LOAD_SAT_DRUG_MANUFACTURING"),
    ]

    MART_PROCEDURES: list[tuple[str, str]] = [
        ("refresh_dim_drug", "PKG_POPULATE_CLINICAL_MART.REFRESH_DIM_DRUG"),
        ("refresh_dim_patient", "PKG_POPULATE_CLINICAL_MART.REFRESH_DIM_PATIENT"),
        ("load_fact_enrollment", "PKG_POPULATE_CLINICAL_MART.LOAD_FACT_ENROLLMENT"),
        ("load_fact_ae", "PKG_POPULATE_CLINICAL_MART.LOAD_FACT_ADVERSE_EVENTS"),
        ("refresh_mvs", "PKG_POPULATE_CLINICAL_MART.REFRESH_ALL_MVS"),
    ]

    def __init__(self, config: VaultConfig | None = None) -> None:
        self._config = config or VaultConfig()
        self._engine: Engine | None = None
        self._file_processor = FileProcessor(self._config)
        self._current_run: PipelineRun | None = None

    @property
    def engine(self) -> Engine:
        """Lazy-initialize the SQLAlchemy engine."""
        if self._engine is None:
            self._engine = create_engine(
                self._config.connection_string,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                echo=False,
            )
            logger.info("Database engine created: %s", self._config.db_host)
        return self._engine

    def _generate_run_id(self) -> str:
        """Generate a unique run identifier."""
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        hash_suffix = hashlib.md5(ts.encode()).hexdigest()[:8]
        return f"RUN_{ts}_{hash_suffix}"

    def _execute_plsql(
        self,
        procedure_name: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a PL/SQL stored procedure via SQLAlchemy.

        Args:
            procedure_name: Fully qualified PL/SQL procedure name.
            params: Input/output parameters for the procedure.

        Returns:
            Dictionary of output parameter values.

        Raises:
            SQLAlchemyError: If the database call fails.
        """
        params = params or {}
        logger.info("Executing PL/SQL: %s with params: %s", procedure_name, list(params.keys()))

        with self.engine.connect() as conn:
            # Build the PL/SQL anonymous block
            param_declarations = []
            param_assignments = []
            out_params: list[str] = []

            for pname, pvalue in params.items():
                if isinstance(pvalue, str) and pvalue.startswith("OUT:"):
                    param_declarations.append(f"v_{pname} NUMBER;")
                    param_assignments.append(f"v_{pname}")
                    out_params.append(pname)
                else:
                    param_assignments.append(f":{pname}")

            call_params = ", ".join(param_assignments)

            plsql_block = f"""
            DECLARE
                {chr(10).join(param_declarations)}
            BEGIN
                {procedure_name}({call_params});
            END;
            """

            bind_params = {
                k: v for k, v in params.items() if not (isinstance(v, str) and v.startswith("OUT:"))
            }

            result = conn.execute(text(plsql_block), bind_params)
            conn.commit()

            return {"procedure": procedure_name, "status": "success"}

    def _execute_step_with_retry(
        self,
        step: PipelineStep,
        procedure_name: str,
        params: dict[str, Any] | None = None,
    ) -> bool:
        """
        Execute a pipeline step with retry logic.

        Args:
            step: The PipelineStep to execute.
            procedure_name: PL/SQL procedure to call.
            params: Procedure parameters.

        Returns:
            True if step succeeded, False otherwise.
        """
        step.status = StepStatus.RUNNING
        step.start_time = datetime.now(UTC)

        for attempt in range(1, self._config.max_retries + 1):
            try:
                logger.info(
                    "Step '%s' attempt %d/%d",
                    step.name,
                    attempt,
                    self._config.max_retries,
                )
                result = self._execute_plsql(procedure_name, params)

                step.status = StepStatus.SUCCESS
                step.end_time = datetime.now(UTC)
                step.rows_affected = result.get("rows_affected", 0)
                logger.info("Step '%s' completed successfully", step.name)
                return True

            except SQLAlchemyError as exc:
                step.retry_count = attempt
                step.error_message = str(exc)
                logger.warning(
                    "Step '%s' failed (attempt %d): %s",
                    step.name,
                    attempt,
                    exc,
                )

                if attempt < self._config.max_retries:
                    step.status = StepStatus.RETRYING
                    delay = self._config.retry_delay_seconds * attempt
                    logger.info("Retrying in %d seconds...", delay)
                    time.sleep(delay)
                else:
                    step.status = StepStatus.FAILED
                    step.end_time = datetime.now(UTC)
                    logger.error(
                        "Step '%s' failed after %d attempts",
                        step.name,
                        self._config.max_retries,
                    )
                    return False

        return False

    def run_daily_etl(self, file_pattern: str = "*.csv") -> PipelineRun:
        """
        Execute the daily ETL pipeline.

        This is the main entry point called by Control-M or cron.
        Processes all matching files in the staging directory through
        the complete ETL pipeline.

        Args:
            file_pattern: Glob pattern for files to process.

        Returns:
            PipelineRun with execution details and status.
        """
        run = PipelineRun(run_id=self._generate_run_id())
        self._current_run = run
        logger.info("Starting daily ETL pipeline: %s", run.run_id)

        try:
            # ---------------------------------------------------------------
            # Phase 1: File Discovery and Processing
            # ---------------------------------------------------------------
            step_discover = PipelineStep(name="discover_files")
            run.steps.append(step_discover)
            step_discover.status = StepStatus.RUNNING
            step_discover.start_time = datetime.now(UTC)

            manifests = self._file_processor.discover_files(file_pattern)
            if not manifests:
                logger.info("No files to process. Pipeline complete.")
                step_discover.status = StepStatus.SKIPPED
                step_discover.end_time = datetime.now(UTC)
                run.status = StepStatus.SUCCESS
                run.end_time = datetime.now(UTC)
                return run

            run.source_files = [m.file_path for m in manifests]
            step_discover.rows_affected = len(manifests)
            step_discover.status = StepStatus.SUCCESS
            step_discover.end_time = datetime.now(UTC)

            # ---------------------------------------------------------------
            # Phase 2: File Validation and Staging
            # ---------------------------------------------------------------
            for manifest in manifests:
                step_validate = PipelineStep(name=f"validate_file_{Path(manifest.file_path).name}")
                run.steps.append(step_validate)

                if not self._file_processor.validate_file(manifest):
                    step_validate.status = StepStatus.FAILED
                    step_validate.error_message = "File validation failed"
                    logger.error("File validation failed: %s", manifest.file_path)
                    continue

                step_validate.status = StepStatus.SUCCESS
                step_validate.end_time = datetime.now(UTC)

                # Process the file into staging
                step_process = PipelineStep(name=f"process_file_{Path(manifest.file_path).name}")
                run.steps.append(step_process)

                report = self._file_processor.process_file(manifest)
                step_process.rows_affected = report.get("rows_processed", 0)
                step_process.status = StepStatus.SUCCESS
                step_process.end_time = datetime.now(UTC)

            # ---------------------------------------------------------------
            # Phase 3: PL/SQL Staging Procedures
            # ---------------------------------------------------------------
            for step_name, proc_name in self.STAGING_PROCEDURES:
                step = PipelineStep(name=step_name)
                run.steps.append(step)
                success = self._execute_step_with_retry(step, proc_name)
                if not success:
                    logger.error("Staging step '%s' failed. Aborting pipeline.", step_name)
                    run.status = StepStatus.FAILED
                    run.end_time = datetime.now(UTC)
                    return run

            # ---------------------------------------------------------------
            # Phase 4: Hub Loading
            # ---------------------------------------------------------------
            for step_name, proc_name in self.HUB_LOAD_PROCEDURES:
                step = PipelineStep(name=step_name)
                run.steps.append(step)
                success = self._execute_step_with_retry(step, proc_name)
                if not success:
                    logger.error("Hub loading step '%s' failed. Aborting.", step_name)
                    run.status = StepStatus.FAILED
                    run.end_time = datetime.now(UTC)
                    return run

            # ---------------------------------------------------------------
            # Phase 5: Satellite Loading
            # ---------------------------------------------------------------
            for step_name, proc_name in self.SATELLITE_PROCEDURES:
                step = PipelineStep(name=step_name)
                run.steps.append(step)
                self._execute_step_with_retry(step, proc_name)
                # Satellite failures are non-fatal; continue with remaining

            # ---------------------------------------------------------------
            # Phase 6: Data Quality Validation
            # ---------------------------------------------------------------
            step_dq = PipelineStep(name="data_quality_checks")
            run.steps.append(step_dq)
            step_dq.status = StepStatus.RUNNING
            step_dq.start_time = datetime.now(UTC)

            try:
                from pharma_vault.quality.data_quality import DataQualityFramework

                dq = DataQualityFramework(self._config)
                dq_results = dq.run_all_checks()
                step_dq.status = StepStatus.SUCCESS
                step_dq.end_time = datetime.now(UTC)
                step_dq.rows_affected = dq_results.get("checks_passed", 0)

                if not dq_results.get("overall_pass", False):
                    logger.warning(
                        "Data quality checks reported warnings: %s",
                        dq_results.get("failures", []),
                    )
            except Exception as exc:
                step_dq.status = StepStatus.FAILED
                step_dq.error_message = str(exc)
                step_dq.end_time = datetime.now(UTC)
                logger.warning("Data quality checks failed (non-fatal): %s", exc)

            # ---------------------------------------------------------------
            # Phase 7: Archive Processed Files
            # ---------------------------------------------------------------
            step_archive = PipelineStep(name="archive_files")
            run.steps.append(step_archive)
            step_archive.status = StepStatus.RUNNING
            step_archive.start_time = datetime.now(UTC)

            for manifest in manifests:
                self._file_processor.archive_file(manifest)

            step_archive.status = StepStatus.SUCCESS
            step_archive.end_time = datetime.now(UTC)

            # ---------------------------------------------------------------
            # Pipeline Complete
            # ---------------------------------------------------------------
            failed_steps = [s for s in run.steps if s.status == StepStatus.FAILED]
            run.status = StepStatus.FAILED if failed_steps else StepStatus.SUCCESS
            run.end_time = datetime.now(UTC)

            logger.info(
                "Pipeline %s completed with status: %s (%d steps, %d failed)",
                run.run_id,
                run.status.value,
                len(run.steps),
                len(failed_steps),
            )

            return run

        except Exception as exc:
            run.status = StepStatus.FAILED
            run.end_time = datetime.now(UTC)
            logger.exception("Pipeline %s failed with unexpected error: %s", run.run_id, exc)
            return run

    def run_mart_refresh(self) -> PipelineRun:
        """
        Execute the data mart refresh pipeline.

        Called weekly (or on-demand) to refresh dimensions, facts,
        and materialized views in the clinical data mart.
        """
        run = PipelineRun(run_id=self._generate_run_id())
        self._current_run = run
        logger.info("Starting mart refresh pipeline: %s", run.run_id)

        for step_name, proc_name in self.MART_PROCEDURES:
            step = PipelineStep(name=step_name)
            run.steps.append(step)
            self._execute_step_with_retry(step, proc_name)

        failed_steps = [s for s in run.steps if s.status == StepStatus.FAILED]
        run.status = StepStatus.FAILED if failed_steps else StepStatus.SUCCESS
        run.end_time = datetime.now(UTC)

        logger.info(
            "Mart refresh %s completed: %s",
            run.run_id,
            run.status.value,
        )
        return run

    def watch_directory(self, poll_interval_seconds: int = 60) -> None:
        """
        Daemon mode: continuously watch the staging directory for new files.

        This method blocks and runs indefinitely, polling the staging
        directory for new files and triggering the ETL pipeline when
        files are detected.

        Args:
            poll_interval_seconds: How often to check for new files.
        """
        logger.info(
            "Starting file watcher on %s (poll every %ds)",
            self._config.staging_dir,
            poll_interval_seconds,
        )
        processed_files: set[str] = set()

        while True:
            try:
                staging_dir = self._config.staging_dir
                if not staging_dir.exists():
                    logger.warning("Staging directory does not exist: %s", staging_dir)
                    time.sleep(poll_interval_seconds)
                    continue

                current_files = {str(f) for f in staging_dir.glob("*.csv") if f.is_file()}
                new_files = current_files - processed_files

                if new_files:
                    logger.info("Detected %d new files", len(new_files))
                    run = self.run_daily_etl()

                    if run.status == StepStatus.SUCCESS:
                        processed_files.update(new_files)
                    else:
                        logger.error("Pipeline failed. Files will be retried next cycle.")

                time.sleep(poll_interval_seconds)

            except KeyboardInterrupt:
                logger.info("File watcher stopped by user")
                break
            except Exception as exc:
                logger.exception("File watcher error: %s", exc)
                time.sleep(poll_interval_seconds)

    def shutdown(self) -> None:
        """Clean up resources."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            logger.info("Database engine disposed")
