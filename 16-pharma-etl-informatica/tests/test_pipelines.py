"""
Integration tests for PharmaFlow ETL pipelines.

Tests the complete pipeline flow from source to target using
sample data fixtures. These tests validate that all transformations
work together correctly in a real mapping.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pharma_flow.framework.session import Session, SessionStatus
from pharma_flow.framework.workflow import (
    AssignmentTask,
    DecisionTask,
    LinkCondition,
    SessionTask,
    TaskStatus,
    Workflow,
    WorkflowContext,
)

# ---------------------------------------------------------------------------
# Drug Master Pipeline Tests
# ---------------------------------------------------------------------------


class TestDrugMasterPipeline:
    def test_build_mapping(self, sample_drug_csv: Path) -> None:
        """Verify the mapping is correctly configured."""
        from pharma_flow.pipelines.drug_master_etl import build_drug_master_mapping

        mapping = build_drug_master_mapping([str(sample_drug_csv)])

        assert mapping.name == "m_drug_master_scd2"
        assert len(mapping.sources) == 1
        assert len(mapping.targets) == 1
        assert len(mapping.transformations) >= 8
        assert len(mapping.parameters) == 2

        # Validate lineage
        lineage = mapping.get_lineage()
        assert "SRC_DRUG_SUPPLIER_1" in lineage["sources"]
        assert "TGT_DIM_DRUG" in lineage["targets"]

    def test_execute_pipeline(self, sample_drug_csv: Path) -> None:
        """Run the full drug master pipeline end-to-end."""
        from pharma_flow.pipelines.drug_master_etl import build_drug_master_mapping

        mapping = build_drug_master_mapping([str(sample_drug_csv)])
        session = Session(
            name="s_drug_test",
            mapping=mapping,
            runtime_params={"$$PROCESS_DATE": "2026-03-05", "$$SUPPLIER_PRIORITY": "1"},
        )

        stats = session.execute()

        assert stats.status == SessionStatus.SUCCEEDED
        assert stats.source_rows_read == 8  # All rows read
        # Some filtered out (invalid NDC), some deduped
        assert stats.target_rows_written > 0

    def test_pipeline_with_existing_dimension(
        self,
        sample_drug_csv: Path,
        existing_dim_drug_df: pd.DataFrame,
    ) -> None:
        """Test SCD logic with pre-existing dimension records."""
        from pharma_flow.pipelines.drug_master_etl import build_drug_master_mapping

        mapping = build_drug_master_mapping(
            [str(sample_drug_csv)],
            existing_dim_data=existing_dim_drug_df,
        )
        session = Session(
            name="s_drug_scd",
            mapping=mapping,
            runtime_params={"$$PROCESS_DATE": "2026-03-05", "$$SUPPLIER_PRIORITY": "1"},
        )

        stats = session.execute()
        assert stats.status == SessionStatus.SUCCEEDED


class TestDrugMasterHelpers:
    def test_validate_ndc_valid(self) -> None:
        from pharma_flow.pipelines.drug_master_etl import validate_ndc

        assert validate_ndc("00071-0155-23") is True
        assert validate_ndc("0378-0123-01") is True

    def test_validate_ndc_invalid(self) -> None:
        from pharma_flow.pipelines.drug_master_etl import validate_ndc

        assert validate_ndc("INVALID") is False
        assert validate_ndc("") is False
        assert validate_ndc(None) is False  # type: ignore[arg-type]

    def test_standardize_drug_name(self) -> None:
        from pharma_flow.pipelines.drug_master_etl import standardize_drug_name

        assert standardize_drug_name("  lipitor  ") == "LIPITOR"
        assert standardize_drug_name("Atorvastatin   Calcium") == "ATORVASTATIN CALCIUM"
        assert standardize_drug_name(None) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Clinical Trial Pipeline Tests
# ---------------------------------------------------------------------------


class TestClinicalTrialPipeline:
    def test_build_mapping(self, sample_trial_xml: Path) -> None:
        from pharma_flow.pipelines.clinical_trial_etl import build_clinical_trial_mapping

        mapping = build_clinical_trial_mapping(str(sample_trial_xml))
        assert mapping.name == "m_clinical_trial_load"
        assert len(mapping.sources) >= 1
        assert len(mapping.targets) >= 1
        assert len(mapping.transformations) >= 4

    def test_execute_pipeline(self, sample_trial_xml: Path) -> None:
        from pharma_flow.pipelines.clinical_trial_etl import build_clinical_trial_mapping

        mapping = build_clinical_trial_mapping(str(sample_trial_xml))
        session = Session(
            name="s_trial_test",
            mapping=mapping,
            runtime_params={
                "$$LOAD_DATE": "2026-03-05",
                "$$DATA_SOURCE": "TEST",
            },
        )

        stats = session.execute()
        assert stats.status == SessionStatus.SUCCEEDED
        assert stats.source_rows_read > 0

    def test_phase_classification(self) -> None:
        from pharma_flow.pipelines.clinical_trial_etl import classify_phase

        assert classify_phase("Phase 3") == "Phase III"
        assert classify_phase("phase 1/phase 2") == "Phase I/II"
        assert classify_phase("Not Applicable") == "N/A"
        assert classify_phase(None) == "Unknown"  # type: ignore[arg-type]

    def test_trial_duration_calculation(self) -> None:
        from pharma_flow.pipelines.clinical_trial_etl import calculate_trial_duration_days

        days = calculate_trial_duration_days("2020-01-01", "2020-12-31")
        assert days == 365

        assert calculate_trial_duration_days("", "") == 0
        assert calculate_trial_duration_days(None, None) == 0  # type: ignore[arg-type]

    def test_parse_enrollment(self) -> None:
        from pharma_flow.pipelines.clinical_trial_etl import parse_enrollment

        assert parse_enrollment("1500") == 1500
        assert parse_enrollment("1,500") == 1500
        assert parse_enrollment("") == 0
        assert parse_enrollment(None) == 0  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FAERS Pipeline Tests
# ---------------------------------------------------------------------------


class TestFAERSPipeline:
    def test_build_mapping(self, sample_faers_files: dict[str, Path]) -> None:
        from pharma_flow.pipelines.adverse_event_etl import build_faers_mapping

        mapping = build_faers_mapping(
            demo_file=str(sample_faers_files["demo"]),
            drug_file=str(sample_faers_files["drug"]),
            reac_file=str(sample_faers_files["reac"]),
            outc_file=str(sample_faers_files["outc"]),
        )
        assert mapping.name == "m_faers_adverse_event_load"
        assert len(mapping.sources) == 4
        assert len(mapping.targets) == 2

    def test_prr_calculation(self) -> None:
        from pharma_flow.pipelines.adverse_event_etl import compute_prr

        prr = compute_prr(a=10, b=90, c=20, d=880)
        assert prr > 0
        # PRR = (10/100) / (20/900) = 0.1 / 0.0222 = 4.5
        assert abs(prr - 4.5) < 0.1

    def test_ror_calculation(self) -> None:
        from pharma_flow.pipelines.adverse_event_etl import compute_ror

        ror = compute_ror(a=10, b=90, c=20, d=880)
        # ROR = (10*880) / (90*20) = 8800/1800 = 4.89
        assert ror > 0

    def test_chi_square_calculation(self) -> None:
        from pharma_flow.pipelines.adverse_event_etl import compute_chi_square

        chi2 = compute_chi_square(a=10, b=90, c=20, d=880)
        assert chi2 >= 0

    def test_age_classification(self) -> None:
        from pharma_flow.pipelines.adverse_event_etl import classify_age_group

        assert classify_age_group(1.5, "YR") == "Infant"
        assert classify_age_group(8, "YR") == "Child"
        assert classify_age_group(16, "YR") == "Adolescent"
        assert classify_age_group(45, "YR") == "Adult"
        assert classify_age_group(70, "YR") == "Elderly"
        assert classify_age_group(0, "YR") == "Unknown"
        assert classify_age_group(18, "MON") == "Infant"  # 1.5 years

    def test_meddra_standardization(self) -> None:
        from pharma_flow.pipelines.adverse_event_etl import standardize_meddra_term

        assert standardize_meddra_term("nausea") == "Nausea"
        assert standardize_meddra_term("  HEADACHE  ") == "Headache"
        assert standardize_meddra_term("Diarrhea") == "Diarrhoea"
        assert standardize_meddra_term(None) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Workflow Orchestration Tests
# ---------------------------------------------------------------------------


class TestWorkflowOrchestration:
    def test_simple_sequential_workflow(self) -> None:
        """Test a workflow with sequential tasks."""
        wf = Workflow(name="wf_test_sequential")

        task1 = AssignmentTask(
            name="T_INIT",
            assignments={"status": "initialized"},
        )
        task2 = AssignmentTask(
            name="T_SET_DATE",
            assignments={"process_date": "2026-03-05"},
        )

        wf.add_task(task1)
        wf.add_task(task2)
        wf.start_task = "T_INIT"
        wf.add_link("T_INIT", "T_SET_DATE", LinkCondition.ON_SUCCESS)

        results = wf.execute()

        assert results["T_INIT"] == TaskStatus.SUCCEEDED
        assert results["T_SET_DATE"] == TaskStatus.SUCCEEDED
        assert wf.context.get_variable("status") == "initialized"
        assert wf.context.get_variable("process_date") == "2026-03-05"

    def test_decision_branching(self) -> None:
        """Test workflow with conditional decision task."""
        wf = Workflow(name="wf_test_decision")

        init = AssignmentTask(
            name="T_INIT",
            assignments={"row_count": 100},
        )
        decision = DecisionTask(
            name="T_DECIDE",
            condition_expression="row_count > 50",
        )
        success_path = AssignmentTask(
            name="T_PROCEED",
            assignments={"path": "success"},
        )
        fail_path = AssignmentTask(
            name="T_SKIP",
            assignments={"path": "skipped"},
        )

        wf.add_task(init)
        wf.add_task(decision)
        wf.add_task(success_path)
        wf.add_task(fail_path)
        wf.start_task = "T_INIT"
        wf.add_link("T_INIT", "T_DECIDE", LinkCondition.ON_SUCCESS)
        wf.add_link("T_DECIDE", "T_PROCEED", LinkCondition.ON_SUCCESS)
        wf.add_link("T_DECIDE", "T_SKIP", LinkCondition.ON_FAILURE)

        results = wf.execute()

        assert results["T_DECIDE"] == TaskStatus.SUCCEEDED
        assert results["T_PROCEED"] == TaskStatus.SUCCEEDED
        assert wf.context.get_variable("path") == "success"

    def test_disabled_task_skipped(self) -> None:
        """Test that disabled tasks are properly skipped."""
        task = AssignmentTask(
            name="T_DISABLED",
            assignments={"x": 1},
            enabled=False,
        )
        ctx = WorkflowContext()
        status = task.run(ctx)
        assert status == TaskStatus.DISABLED
        assert ctx.get_variable("x") is None

    def test_workflow_status_report(self) -> None:
        wf = Workflow(name="wf_report_test")
        wf.add_task(AssignmentTask(name="T1", assignments={"a": 1}))
        wf.start_task = "T1"
        wf.execute()

        report = wf.get_status_report()
        assert report["workflow"] == "wf_report_test"
        assert "T1" in report["tasks"]
        assert report["tasks"]["T1"]["status"] == "succeeded"

    def test_etl_session_in_workflow(self, sample_drug_csv: Path, tmp_path: Path) -> None:
        """Test running an ETL session within a workflow."""
        from pharma_flow.framework.mapping import (
            Mapping,
            SourceDefinition,
            SourceType,
            TargetDefinition,
            TargetType,
        )
        from pharma_flow.framework.transformations import SourceQualifier

        output = tmp_path / "wf_output.csv"
        source = SourceDefinition(
            name="SRC",
            source_type=SourceType.FLAT_FILE,
            file_path=str(sample_drug_csv),
        )
        target = TargetDefinition(
            name="TGT",
            target_type=TargetType.FLAT_FILE,
            file_path=str(output),
        )
        mapping = Mapping(
            name="m_wf_test",
            sources=[source],
            targets=[target],
        )
        mapping.add_transformation(SourceQualifier(name="SQ"))

        session = Session(name="s_wf", mapping=mapping)

        wf = Workflow(name="wf_etl_test")
        wf.add_task(SessionTask(name="T_ETL", session=session))
        wf.start_task = "T_ETL"

        results = wf.execute()
        assert results["T_ETL"] == TaskStatus.SUCCEEDED
        assert output.exists()
