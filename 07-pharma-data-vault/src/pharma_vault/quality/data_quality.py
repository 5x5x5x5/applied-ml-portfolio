"""
Data Quality Framework for PharmaDataVault.

Provides comprehensive data quality validation at multiple layers:
  - Completeness: checks for NULL/missing values in required fields
  - Referential Integrity: validates foreign key relationships across vault
  - Business Rules: domain-specific validations for pharma data
  - Quality Scoring: generates a composite DQ score per entity and table

Designed to run after ETL loading and before mart population. Results
are logged to ETL_ERROR_LOG and returned as structured reports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from pharma_vault._config import VaultConfig

logger = logging.getLogger(__name__)


class CheckSeverity(Enum):
    """Severity level for data quality check results."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class CheckCategory(Enum):
    """Category of data quality check."""

    COMPLETENESS = "completeness"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    BUSINESS_RULE = "business_rule"
    TIMELINESS = "timeliness"
    UNIQUENESS = "uniqueness"
    CONSISTENCY = "consistency"


@dataclass
class DQCheckResult:
    """Result of a single data quality check."""

    check_name: str
    category: CheckCategory
    table_name: str
    severity: CheckSeverity
    passed: bool
    metric_value: float | int
    threshold: float | int
    description: str
    details: str | None = None
    check_time: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "check_name": self.check_name,
            "category": self.category.value,
            "table": self.table_name,
            "severity": self.severity.value,
            "passed": self.passed,
            "metric": self.metric_value,
            "threshold": self.threshold,
            "description": self.description,
            "details": self.details,
        }


@dataclass
class DQReport:
    """Aggregate data quality report."""

    report_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    checks_run: int = 0
    checks_passed: int = 0
    checks_failed: int = 0
    checks_warned: int = 0
    overall_score: float = 0.0
    overall_pass: bool = True
    results: list[DQCheckResult] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "report_time": self.report_time.isoformat(),
            "checks_run": self.checks_run,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "checks_warned": self.checks_warned,
            "overall_score": round(self.overall_score, 2),
            "overall_pass": self.overall_pass,
            "results": [r.to_dict() for r in self.results],
            "failures": self.failures,
        }


class DataQualityFramework:
    """
    Comprehensive data quality validation for the pharmaceutical data vault.

    Executes a battery of checks across raw vault, business vault, and
    staging tables. Each check returns a pass/fail result with metrics.
    """

    def __init__(self, config: VaultConfig | None = None) -> None:
        self._config = config or VaultConfig()
        self._engine: Engine | None = None

    @property
    def engine(self) -> Engine:
        """Lazy-initialize database engine."""
        if self._engine is None:
            self._engine = create_engine(
                self._config.connection_string,
                pool_size=3,
                pool_pre_ping=True,
            )
        return self._engine

    def _execute_scalar(self, query: str) -> Any:
        """Execute a query and return a single scalar value."""
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            row = result.fetchone()
            return row[0] if row else None

    def _add_result(
        self,
        report: DQReport,
        check_name: str,
        category: CheckCategory,
        table_name: str,
        metric_value: float | int,
        threshold: float | int,
        description: str,
        higher_is_better: bool = True,
        severity_on_fail: CheckSeverity = CheckSeverity.ERROR,
        details: str | None = None,
    ) -> None:
        """Create and add a check result to the report."""
        if higher_is_better:
            passed = metric_value >= threshold
        else:
            passed = metric_value <= threshold

        result = DQCheckResult(
            check_name=check_name,
            category=category,
            table_name=table_name,
            severity=CheckSeverity.INFO if passed else severity_on_fail,
            passed=passed,
            metric_value=metric_value,
            threshold=threshold,
            description=description,
            details=details,
        )

        report.results.append(result)
        report.checks_run += 1

        if passed:
            report.checks_passed += 1
        elif severity_on_fail == CheckSeverity.WARNING:
            report.checks_warned += 1
        else:
            report.checks_failed += 1
            report.failures.append(f"{check_name}: {description} (value={metric_value})")

    # =========================================================================
    # Completeness Checks
    # =========================================================================

    def check_completeness(self, report: DQReport) -> None:
        """Run completeness checks on required fields across vault tables."""
        logger.info("Running completeness checks...")

        # Hub tables: business keys must never be NULL
        hub_checks: list[tuple[str, str]] = [
            ("HUB_DRUG", "DRUG_NDC"),
            ("HUB_PATIENT", "PATIENT_MRN"),
            ("HUB_CLINICAL_TRIAL", "TRIAL_NCT_ID"),
            ("HUB_FACILITY", "FACILITY_ID"),
            ("HUB_ADVERSE_EVENT", "AE_REPORT_ID"),
        ]

        for table, column in hub_checks:
            try:
                null_count = self._execute_scalar(
                    f"SELECT COUNT(*) FROM {table} WHERE {column} IS NULL"
                )
                total_count = self._execute_scalar(f"SELECT COUNT(*) FROM {table}")
                completeness_pct = 100.0 * (total_count - (null_count or 0)) / max(total_count, 1)

                self._add_result(
                    report,
                    check_name=f"completeness_{table}_{column}",
                    category=CheckCategory.COMPLETENESS,
                    table_name=table,
                    metric_value=completeness_pct,
                    threshold=100.0,
                    description=f"{column} completeness in {table}",
                    higher_is_better=True,
                    severity_on_fail=CheckSeverity.CRITICAL,
                    details=f"{null_count} NULL values out of {total_count} rows",
                )
            except Exception as exc:
                logger.warning("Completeness check failed for %s.%s: %s", table, column, exc)

        # Satellite required fields
        sat_checks: list[tuple[str, list[str]]] = [
            ("SAT_DRUG_DETAILS", ["DRUG_NAME", "MANUFACTURER", "DRUG_FORM"]),
            ("SAT_PATIENT_DEMOGRAPHICS", ["SEX"]),
            ("SAT_CLINICAL_TRIAL_DETAILS", ["TRIAL_PHASE", "TRIAL_STATUS", "SPONSOR"]),
            ("SAT_ADVERSE_EVENT_DETAILS", ["AE_TERM", "SEVERITY", "ONSET_DATE"]),
        ]

        for table, columns in sat_checks:
            for column in columns:
                try:
                    null_count = self._execute_scalar(
                        f"SELECT COUNT(*) FROM {table} WHERE {column} IS NULL"
                    )
                    total = self._execute_scalar(f"SELECT COUNT(*) FROM {table}")
                    pct = 100.0 * (total - (null_count or 0)) / max(total, 1)

                    self._add_result(
                        report,
                        check_name=f"completeness_{table}_{column}",
                        category=CheckCategory.COMPLETENESS,
                        table_name=table,
                        metric_value=pct,
                        threshold=95.0,
                        description=f"{column} completeness in {table}",
                        higher_is_better=True,
                        severity_on_fail=CheckSeverity.WARNING,
                    )
                except Exception as exc:
                    logger.warning("Satellite check failed for %s.%s: %s", table, column, exc)

    # =========================================================================
    # Referential Integrity Checks
    # =========================================================================

    def check_referential_integrity(self, report: DQReport) -> None:
        """Validate foreign key relationships in link and satellite tables."""
        logger.info("Running referential integrity checks...")

        # Link -> Hub FK checks
        fk_checks: list[tuple[str, str, str, str]] = [
            ("LNK_PATIENT_TRIAL", "PATIENT_KEY", "HUB_PATIENT", "PATIENT_KEY"),
            ("LNK_PATIENT_TRIAL", "TRIAL_KEY", "HUB_CLINICAL_TRIAL", "TRIAL_KEY"),
            ("LNK_TRIAL_DRUG", "TRIAL_KEY", "HUB_CLINICAL_TRIAL", "TRIAL_KEY"),
            ("LNK_TRIAL_DRUG", "DRUG_KEY", "HUB_DRUG", "DRUG_KEY"),
            ("LNK_PATIENT_DRUG", "PATIENT_KEY", "HUB_PATIENT", "PATIENT_KEY"),
            ("LNK_PATIENT_DRUG", "DRUG_KEY", "HUB_DRUG", "DRUG_KEY"),
            ("LNK_DRUG_ADVERSE_EVENT", "DRUG_KEY", "HUB_DRUG", "DRUG_KEY"),
            ("LNK_DRUG_ADVERSE_EVENT", "AE_KEY", "HUB_ADVERSE_EVENT", "AE_KEY"),
            ("LNK_TRIAL_FACILITY", "TRIAL_KEY", "HUB_CLINICAL_TRIAL", "TRIAL_KEY"),
            ("LNK_TRIAL_FACILITY", "FACILITY_KEY", "HUB_FACILITY", "FACILITY_KEY"),
        ]

        for child_table, child_col, parent_table, parent_col in fk_checks:
            try:
                orphan_count = self._execute_scalar(f"""
                    SELECT COUNT(*)
                    FROM {child_table} c
                    WHERE NOT EXISTS (
                        SELECT 1 FROM {parent_table} p
                        WHERE p.{parent_col} = c.{child_col}
                    )
                """)

                self._add_result(
                    report,
                    check_name=f"ri_{child_table}_{child_col}",
                    category=CheckCategory.REFERENTIAL_INTEGRITY,
                    table_name=child_table,
                    metric_value=orphan_count or 0,
                    threshold=0,
                    description=(
                        f"Orphan {child_col} in {child_table} not found in {parent_table}"
                    ),
                    higher_is_better=False,
                    severity_on_fail=CheckSeverity.CRITICAL,
                    details=f"{orphan_count} orphan records",
                )
            except Exception as exc:
                logger.warning("RI check failed for %s->%s: %s", child_table, parent_table, exc)

        # Satellite -> Hub FK checks
        sat_hub_checks: list[tuple[str, str, str]] = [
            ("SAT_DRUG_DETAILS", "DRUG_KEY", "HUB_DRUG"),
            ("SAT_PATIENT_DEMOGRAPHICS", "PATIENT_KEY", "HUB_PATIENT"),
            ("SAT_CLINICAL_TRIAL_DETAILS", "TRIAL_KEY", "HUB_CLINICAL_TRIAL"),
            ("SAT_ADVERSE_EVENT_DETAILS", "AE_KEY", "HUB_ADVERSE_EVENT"),
            ("SAT_DRUG_MANUFACTURING", "DRUG_KEY", "HUB_DRUG"),
        ]

        for sat_table, key_col, hub_table in sat_hub_checks:
            try:
                orphan_count = self._execute_scalar(f"""
                    SELECT COUNT(DISTINCT s.{key_col})
                    FROM {sat_table} s
                    WHERE NOT EXISTS (
                        SELECT 1 FROM {hub_table} h
                        WHERE h.{key_col} = s.{key_col}
                    )
                """)

                self._add_result(
                    report,
                    check_name=f"ri_{sat_table}_{hub_table}",
                    category=CheckCategory.REFERENTIAL_INTEGRITY,
                    table_name=sat_table,
                    metric_value=orphan_count or 0,
                    threshold=0,
                    description=f"Satellite {sat_table} references missing hub {hub_table}",
                    higher_is_better=False,
                    severity_on_fail=CheckSeverity.CRITICAL,
                )
            except Exception as exc:
                logger.warning("Satellite RI check failed for %s: %s", sat_table, exc)

    # =========================================================================
    # Business Rule Checks
    # =========================================================================

    def check_business_rules(self, report: DQReport) -> None:
        """Validate pharmaceutical domain-specific business rules."""
        logger.info("Running business rule checks...")

        # Rule 1: NDC format validation (should be XX-XXXX-XX pattern)
        try:
            invalid_ndc = self._execute_scalar("""
                SELECT COUNT(*)
                FROM HUB_DRUG
                WHERE NOT REGEXP_LIKE(DRUG_NDC, '^[0-9]{5}-[0-9]{4}-[0-9]{2}$')
            """)

            self._add_result(
                report,
                check_name="br_ndc_format",
                category=CheckCategory.BUSINESS_RULE,
                table_name="HUB_DRUG",
                metric_value=invalid_ndc or 0,
                threshold=0,
                description="NDC codes not in standard 5-4-2 format",
                higher_is_better=False,
                severity_on_fail=CheckSeverity.WARNING,
            )
        except Exception as exc:
            logger.warning("NDC format check failed: %s", exc)

        # Rule 2: Clinical trial phase validation
        try:
            invalid_phase = self._execute_scalar("""
                SELECT COUNT(*)
                FROM SAT_CLINICAL_TRIAL_DETAILS
                WHERE LOAD_END_DATE = TO_TIMESTAMP('9999-12-31 23:59:59', 'YYYY-MM-DD HH24:MI:SS')
                  AND TRIAL_PHASE NOT IN (
                      'Phase I', 'Phase I/II', 'Phase II', 'Phase II/III',
                      'Phase III', 'Phase IV', 'Pre-clinical'
                  )
            """)

            self._add_result(
                report,
                check_name="br_trial_phase",
                category=CheckCategory.BUSINESS_RULE,
                table_name="SAT_CLINICAL_TRIAL_DETAILS",
                metric_value=invalid_phase or 0,
                threshold=0,
                description="Clinical trials with invalid phase designation",
                higher_is_better=False,
                severity_on_fail=CheckSeverity.ERROR,
            )
        except Exception as exc:
            logger.warning("Trial phase check failed: %s", exc)

        # Rule 3: AE severity vs outcome consistency
        # Fatal severity must have fatal outcome
        try:
            inconsistent_fatal = self._execute_scalar("""
                SELECT COUNT(*)
                FROM SAT_ADVERSE_EVENT_DETAILS
                WHERE LOAD_END_DATE = TO_TIMESTAMP('9999-12-31 23:59:59', 'YYYY-MM-DD HH24:MI:SS')
                  AND SEVERITY = 'fatal'
                  AND OUTCOME != 'fatal'
            """)

            self._add_result(
                report,
                check_name="br_ae_fatal_consistency",
                category=CheckCategory.BUSINESS_RULE,
                table_name="SAT_ADVERSE_EVENT_DETAILS",
                metric_value=inconsistent_fatal or 0,
                threshold=0,
                description="AEs with fatal severity but non-fatal outcome",
                higher_is_better=False,
                severity_on_fail=CheckSeverity.ERROR,
            )
        except Exception as exc:
            logger.warning("AE consistency check failed: %s", exc)

        # Rule 4: Manufacturing expiry date must be after mfg date
        try:
            invalid_expiry = self._execute_scalar("""
                SELECT COUNT(*)
                FROM SAT_DRUG_MANUFACTURING
                WHERE LOAD_END_DATE = TO_TIMESTAMP('9999-12-31 23:59:59', 'YYYY-MM-DD HH24:MI:SS')
                  AND EXPIRY_DATE <= MFG_DATE
            """)

            self._add_result(
                report,
                check_name="br_mfg_expiry_after_mfg",
                category=CheckCategory.BUSINESS_RULE,
                table_name="SAT_DRUG_MANUFACTURING",
                metric_value=invalid_expiry or 0,
                threshold=0,
                description="Manufacturing lots where expiry <= mfg date",
                higher_is_better=False,
                severity_on_fail=CheckSeverity.CRITICAL,
            )
        except Exception as exc:
            logger.warning("Mfg expiry check failed: %s", exc)

        # Rule 5: AE onset date must be on or before report date
        try:
            invalid_onset = self._execute_scalar("""
                SELECT COUNT(*)
                FROM SAT_ADVERSE_EVENT_DETAILS
                WHERE LOAD_END_DATE = TO_TIMESTAMP('9999-12-31 23:59:59', 'YYYY-MM-DD HH24:MI:SS')
                  AND ONSET_DATE > REPORT_DATE
            """)

            self._add_result(
                report,
                check_name="br_ae_onset_before_report",
                category=CheckCategory.BUSINESS_RULE,
                table_name="SAT_ADVERSE_EVENT_DETAILS",
                metric_value=invalid_onset or 0,
                threshold=0,
                description="AEs where onset date is after report date",
                higher_is_better=False,
                severity_on_fail=CheckSeverity.WARNING,
            )
        except Exception as exc:
            logger.warning("AE onset check failed: %s", exc)

    # =========================================================================
    # Uniqueness Checks
    # =========================================================================

    def check_uniqueness(self, report: DQReport) -> None:
        """Validate uniqueness constraints on business keys."""
        logger.info("Running uniqueness checks...")

        hub_bk_checks: list[tuple[str, str]] = [
            ("HUB_DRUG", "DRUG_NDC"),
            ("HUB_PATIENT", "PATIENT_MRN"),
            ("HUB_CLINICAL_TRIAL", "TRIAL_NCT_ID"),
            ("HUB_FACILITY", "FACILITY_ID"),
            ("HUB_ADVERSE_EVENT", "AE_REPORT_ID"),
        ]

        for table, bk_column in hub_bk_checks:
            try:
                dup_count = self._execute_scalar(f"""
                    SELECT COUNT(*) FROM (
                        SELECT {bk_column}, COUNT(*) AS cnt
                        FROM {table}
                        GROUP BY {bk_column}
                        HAVING COUNT(*) > 1
                    )
                """)

                self._add_result(
                    report,
                    check_name=f"unique_{table}_{bk_column}",
                    category=CheckCategory.UNIQUENESS,
                    table_name=table,
                    metric_value=dup_count or 0,
                    threshold=0,
                    description=f"Duplicate {bk_column} values in {table}",
                    higher_is_better=False,
                    severity_on_fail=CheckSeverity.CRITICAL,
                )
            except Exception as exc:
                logger.warning("Uniqueness check failed for %s: %s", table, exc)

        # Satellite: check for multiple current records per hub key
        sat_current_checks: list[tuple[str, str]] = [
            ("SAT_DRUG_DETAILS", "DRUG_KEY"),
            ("SAT_PATIENT_DEMOGRAPHICS", "PATIENT_KEY"),
            ("SAT_CLINICAL_TRIAL_DETAILS", "TRIAL_KEY"),
            ("SAT_ADVERSE_EVENT_DETAILS", "AE_KEY"),
        ]

        for table, key_col in sat_current_checks:
            try:
                multi_current = self._execute_scalar(f"""
                    SELECT COUNT(*) FROM (
                        SELECT {key_col}, COUNT(*) AS cnt
                        FROM {table}
                        WHERE LOAD_END_DATE = TO_TIMESTAMP('9999-12-31 23:59:59',
                              'YYYY-MM-DD HH24:MI:SS')
                        GROUP BY {key_col}
                        HAVING COUNT(*) > 1
                    )
                """)

                self._add_result(
                    report,
                    check_name=f"unique_current_{table}",
                    category=CheckCategory.UNIQUENESS,
                    table_name=table,
                    metric_value=multi_current or 0,
                    threshold=0,
                    description=f"Hub keys with multiple current records in {table}",
                    higher_is_better=False,
                    severity_on_fail=CheckSeverity.ERROR,
                )
            except Exception as exc:
                logger.warning("Current-record uniqueness check failed for %s: %s", table, exc)

    # =========================================================================
    # Run All Checks
    # =========================================================================

    def run_all_checks(self) -> dict[str, Any]:
        """
        Execute all data quality checks and return a comprehensive report.

        Returns:
            Dictionary containing the full DQ report.
        """
        logger.info("Starting comprehensive data quality validation...")
        report = DQReport()

        try:
            self.check_completeness(report)
            self.check_referential_integrity(report)
            self.check_business_rules(report)
            self.check_uniqueness(report)
        except Exception as exc:
            logger.exception("Data quality framework error: %s", exc)
            report.failures.append(f"Framework error: {exc}")

        # Calculate overall score
        if report.checks_run > 0:
            report.overall_score = (report.checks_passed / report.checks_run) * 100
        else:
            report.overall_score = 0.0

        report.overall_pass = report.checks_failed == 0
        report.report_time = datetime.now(UTC)

        logger.info(
            "Data quality validation complete: %d checks, %d passed, %d failed, %d warnings. "
            "Score: %.1f%%",
            report.checks_run,
            report.checks_passed,
            report.checks_failed,
            report.checks_warned,
            report.overall_score,
        )

        return report.to_dict()

    def shutdown(self) -> None:
        """Dispose of database connections."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
