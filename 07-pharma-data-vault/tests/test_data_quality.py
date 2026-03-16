"""
Tests for the Data Quality Framework.

Tests cover:
  - DQ check result construction and serialization
  - DQ report aggregation and scoring
  - Check result classification (pass/fail/warn)
  - Framework initialization and configuration
  - Individual check category logic (without DB)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pharma_vault._config import VaultConfig
from pharma_vault.quality.data_quality import (
    CheckCategory,
    CheckSeverity,
    DataQualityFramework,
    DQCheckResult,
    DQReport,
)

# =============================================================================
# DQCheckResult Tests
# =============================================================================


class TestDQCheckResult:
    """Tests for individual data quality check results."""

    def test_passing_check(self) -> None:
        """Check that meets threshold is marked as passed."""
        result = DQCheckResult(
            check_name="test_completeness",
            category=CheckCategory.COMPLETENESS,
            table_name="HUB_DRUG",
            severity=CheckSeverity.INFO,
            passed=True,
            metric_value=100.0,
            threshold=95.0,
            description="Drug NDC completeness",
        )
        assert result.passed is True
        assert result.severity == CheckSeverity.INFO

    def test_failing_check(self) -> None:
        """Check below threshold is marked as failed."""
        result = DQCheckResult(
            check_name="test_completeness",
            category=CheckCategory.COMPLETENESS,
            table_name="HUB_DRUG",
            severity=CheckSeverity.ERROR,
            passed=False,
            metric_value=85.0,
            threshold=95.0,
            description="Drug NDC completeness",
        )
        assert result.passed is False
        assert result.severity == CheckSeverity.ERROR

    def test_to_dict(self) -> None:
        """Check result serializes to dictionary correctly."""
        result = DQCheckResult(
            check_name="test_check",
            category=CheckCategory.BUSINESS_RULE,
            table_name="SAT_DRUG_DETAILS",
            severity=CheckSeverity.WARNING,
            passed=False,
            metric_value=5,
            threshold=0,
            description="Invalid NDC format",
            details="5 records with bad NDC",
        )
        d = result.to_dict()

        assert d["check_name"] == "test_check"
        assert d["category"] == "business_rule"
        assert d["table"] == "SAT_DRUG_DETAILS"
        assert d["severity"] == "warning"
        assert d["passed"] is False
        assert d["metric"] == 5
        assert d["threshold"] == 0
        assert d["details"] == "5 records with bad NDC"


# =============================================================================
# DQReport Tests
# =============================================================================


class TestDQReport:
    """Tests for aggregate DQ report."""

    def test_empty_report(self) -> None:
        """Empty report has zero counts."""
        report = DQReport()
        assert report.checks_run == 0
        assert report.checks_passed == 0
        assert report.checks_failed == 0
        assert report.overall_score == 0.0
        assert report.overall_pass is True

    def test_report_with_results(self) -> None:
        """Report aggregates results correctly."""
        report = DQReport()
        report.checks_run = 10
        report.checks_passed = 8
        report.checks_failed = 1
        report.checks_warned = 1
        report.overall_score = 80.0
        report.overall_pass = False
        report.failures = ["check_1: Failed"]

        d = report.to_dict()
        assert d["checks_run"] == 10
        assert d["checks_passed"] == 8
        assert d["checks_failed"] == 1
        assert d["overall_score"] == 80.0
        assert d["overall_pass"] is False
        assert len(d["failures"]) == 1

    def test_report_score_calculation(self) -> None:
        """Overall score correctly reflects pass rate."""
        report = DQReport()
        report.checks_run = 20
        report.checks_passed = 18
        report.checks_failed = 2
        report.overall_score = (18 / 20) * 100

        assert report.overall_score == 90.0


# =============================================================================
# DataQualityFramework Tests
# =============================================================================


class TestDataQualityFramework:
    """Tests for the DQ framework without actual database."""

    def test_framework_init(self, test_config: VaultConfig) -> None:
        """Framework initializes with config."""
        dq = DataQualityFramework(config=test_config)
        assert dq._config == test_config
        assert dq._engine is None

    def test_framework_default_config(self) -> None:
        """Framework works with default config."""
        dq = DataQualityFramework()
        assert dq._config is not None

    def test_add_result_passing(self, test_config: VaultConfig) -> None:
        """_add_result correctly adds a passing check."""
        dq = DataQualityFramework(config=test_config)
        report = DQReport()

        dq._add_result(
            report,
            check_name="test_pass",
            category=CheckCategory.COMPLETENESS,
            table_name="HUB_DRUG",
            metric_value=100.0,
            threshold=95.0,
            description="All good",
            higher_is_better=True,
        )

        assert report.checks_run == 1
        assert report.checks_passed == 1
        assert report.checks_failed == 0
        assert report.results[0].passed is True
        assert report.results[0].severity == CheckSeverity.INFO

    def test_add_result_failing(self, test_config: VaultConfig) -> None:
        """_add_result correctly adds a failing check."""
        dq = DataQualityFramework(config=test_config)
        report = DQReport()

        dq._add_result(
            report,
            check_name="test_fail",
            category=CheckCategory.REFERENTIAL_INTEGRITY,
            table_name="LNK_PATIENT_TRIAL",
            metric_value=50,
            threshold=0,
            description="Orphan records found",
            higher_is_better=False,
            severity_on_fail=CheckSeverity.CRITICAL,
        )

        assert report.checks_run == 1
        assert report.checks_passed == 0
        assert report.checks_failed == 1
        assert report.results[0].passed is False
        assert report.results[0].severity == CheckSeverity.CRITICAL
        assert len(report.failures) == 1

    def test_add_result_warning(self, test_config: VaultConfig) -> None:
        """_add_result with WARNING severity increments warns not fails."""
        dq = DataQualityFramework(config=test_config)
        report = DQReport()

        dq._add_result(
            report,
            check_name="test_warn",
            category=CheckCategory.BUSINESS_RULE,
            table_name="HUB_DRUG",
            metric_value=3,
            threshold=0,
            description="Minor issues",
            higher_is_better=False,
            severity_on_fail=CheckSeverity.WARNING,
        )

        assert report.checks_run == 1
        assert report.checks_passed == 0
        assert report.checks_warned == 1
        assert report.checks_failed == 0

    def test_add_result_lower_is_better(self, test_config: VaultConfig) -> None:
        """higher_is_better=False: metric at threshold passes."""
        dq = DataQualityFramework(config=test_config)
        report = DQReport()

        dq._add_result(
            report,
            check_name="test_zero",
            category=CheckCategory.UNIQUENESS,
            table_name="HUB_DRUG",
            metric_value=0,
            threshold=0,
            description="No duplicates",
            higher_is_better=False,
        )

        assert report.results[0].passed is True

    @patch.object(DataQualityFramework, "_execute_scalar")
    def test_check_completeness(
        self,
        mock_scalar: MagicMock,
        test_config: VaultConfig,
    ) -> None:
        """Completeness checks execute and aggregate correctly."""
        # Mock: no nulls, 100 total rows for each table
        mock_scalar.side_effect = [
            0,
            100,  # HUB_DRUG.DRUG_NDC: 0 nulls, 100 total
            0,
            100,  # HUB_PATIENT.PATIENT_MRN
            0,
            100,  # HUB_CLINICAL_TRIAL.TRIAL_NCT_ID
            0,
            100,  # HUB_FACILITY.FACILITY_ID
            0,
            100,  # HUB_ADVERSE_EVENT.AE_REPORT_ID
            0,
            100,  # SAT_DRUG_DETAILS.DRUG_NAME
            0,
            100,  # SAT_DRUG_DETAILS.MANUFACTURER
            0,
            100,  # SAT_DRUG_DETAILS.DRUG_FORM
            0,
            100,  # SAT_PATIENT_DEMOGRAPHICS.SEX
            0,
            100,  # SAT_CLINICAL_TRIAL_DETAILS.TRIAL_PHASE
            0,
            100,  # SAT_CLINICAL_TRIAL_DETAILS.TRIAL_STATUS
            0,
            100,  # SAT_CLINICAL_TRIAL_DETAILS.SPONSOR
            0,
            100,  # SAT_ADVERSE_EVENT_DETAILS.AE_TERM
            0,
            100,  # SAT_ADVERSE_EVENT_DETAILS.SEVERITY
            0,
            100,  # SAT_ADVERSE_EVENT_DETAILS.ONSET_DATE
        ]

        dq = DataQualityFramework(config=test_config)
        report = DQReport()
        dq.check_completeness(report)

        # All checks should pass (100% completeness)
        assert report.checks_run > 0
        assert report.checks_passed == report.checks_run
        assert report.checks_failed == 0

    @patch.object(DataQualityFramework, "_execute_scalar")
    def test_check_referential_integrity(
        self,
        mock_scalar: MagicMock,
        test_config: VaultConfig,
    ) -> None:
        """RI checks with zero orphans pass."""
        # 10 link checks + 5 satellite checks = 15 checks, all returning 0
        mock_scalar.return_value = 0

        dq = DataQualityFramework(config=test_config)
        report = DQReport()
        dq.check_referential_integrity(report)

        assert report.checks_run > 0
        assert report.checks_failed == 0

    @patch.object(DataQualityFramework, "_execute_scalar")
    def test_check_referential_integrity_with_orphans(
        self,
        mock_scalar: MagicMock,
        test_config: VaultConfig,
    ) -> None:
        """RI checks with orphan records fail."""
        # First check returns 5 orphans, rest return 0
        mock_scalar.side_effect = [5] + [0] * 20

        dq = DataQualityFramework(config=test_config)
        report = DQReport()
        dq.check_referential_integrity(report)

        assert report.checks_failed >= 1
        assert any("orphan" in f.lower() for f in report.failures)

    @patch.object(DataQualityFramework, "_execute_scalar")
    def test_check_business_rules(
        self,
        mock_scalar: MagicMock,
        test_config: VaultConfig,
    ) -> None:
        """Business rule checks all passing."""
        mock_scalar.return_value = 0

        dq = DataQualityFramework(config=test_config)
        report = DQReport()
        dq.check_business_rules(report)

        assert report.checks_run >= 4  # At least 4 business rule checks
        assert report.checks_failed == 0

    @patch.object(DataQualityFramework, "_execute_scalar")
    def test_check_uniqueness(
        self,
        mock_scalar: MagicMock,
        test_config: VaultConfig,
    ) -> None:
        """Uniqueness checks with no duplicates pass."""
        mock_scalar.return_value = 0

        dq = DataQualityFramework(config=test_config)
        report = DQReport()
        dq.check_uniqueness(report)

        assert report.checks_run >= 5  # Hub BK checks + satellite current checks
        assert report.checks_failed == 0

    @patch.object(DataQualityFramework, "_execute_scalar")
    def test_run_all_checks(
        self,
        mock_scalar: MagicMock,
        test_config: VaultConfig,
    ) -> None:
        """Full DQ run returns structured report."""
        mock_scalar.return_value = 0

        dq = DataQualityFramework(config=test_config)
        result = dq.run_all_checks()

        assert "checks_run" in result
        assert "checks_passed" in result
        assert "overall_score" in result
        assert "overall_pass" in result
        assert result["checks_run"] > 0
        # With all mocks returning 0, everything should pass
        assert result["overall_pass"] is True

    @patch.object(DataQualityFramework, "_execute_scalar")
    def test_run_all_checks_with_failures(
        self,
        mock_scalar: MagicMock,
        test_config: VaultConfig,
    ) -> None:
        """Full DQ run with some failures sets overall_pass=False."""
        # Return non-zero for some checks to trigger failures
        call_count = 0

        def side_effect_fn(*_args: object, **_kwargs: object) -> int:
            nonlocal call_count
            call_count += 1
            # Every 5th call returns a failure value
            if call_count % 5 == 0:
                return 10  # 10 orphans / duplicates / etc.
            return 0

        mock_scalar.side_effect = side_effect_fn

        dq = DataQualityFramework(config=test_config)
        result = dq.run_all_checks()

        assert result["checks_run"] > 0
        # Score should be less than 100
        assert result["overall_score"] < 100.0

    def test_shutdown(self, test_config: VaultConfig) -> None:
        """Shutdown without engine doesn't raise."""
        dq = DataQualityFramework(config=test_config)
        dq.shutdown()
        assert dq._engine is None


# =============================================================================
# Check Category and Severity Enum Tests
# =============================================================================


class TestEnums:
    """Tests for DQ enums."""

    def test_check_severity_values(self) -> None:
        """All severity levels are defined."""
        assert CheckSeverity.INFO.value == "info"
        assert CheckSeverity.WARNING.value == "warning"
        assert CheckSeverity.ERROR.value == "error"
        assert CheckSeverity.CRITICAL.value == "critical"

    def test_check_category_values(self) -> None:
        """All check categories are defined."""
        assert CheckCategory.COMPLETENESS.value == "completeness"
        assert CheckCategory.REFERENTIAL_INTEGRITY.value == "referential_integrity"
        assert CheckCategory.BUSINESS_RULE.value == "business_rule"
        assert CheckCategory.TIMELINESS.value == "timeliness"
        assert CheckCategory.UNIQUENESS.value == "uniqueness"
        assert CheckCategory.CONSISTENCY.value == "consistency"
