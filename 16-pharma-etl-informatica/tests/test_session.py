"""
Tests for Session execution, error handling, and recovery.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pharma_flow.framework.mapping import (
    Mapping,
    MappingParameter,
    SessionConfig,
    SourceDefinition,
    SourceType,
    TargetDefinition,
    TargetType,
)
from pharma_flow.framework.session import (
    PerformanceStats,
    Session,
    SessionCheckpoint,
    SessionStatus,
)
from pharma_flow.framework.transformations import (
    Expression,
    Filter,
    RowDisposition,
    SequenceGenerator,
    SourceQualifier,
    UpdateStrategy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_simple_mapping(csv_path: str, output_path: str) -> Mapping:
    """Build a minimal mapping for testing."""
    source = SourceDefinition(
        name="SRC_TEST",
        source_type=SourceType.FLAT_FILE,
        file_path=csv_path,
    )
    target = TargetDefinition(
        name="TGT_TEST",
        target_type=TargetType.FLAT_FILE,
        file_path=output_path,
    )
    sq = SourceQualifier(name="SQ_TEST")
    exp = Expression(name="EXP_UPPER")
    exp.add_expression("drug_upper", lambda d: d["drug_name"].str.upper().str.strip())

    mapping = Mapping(
        name="m_test_simple",
        sources=[source],
        targets=[target],
    )
    mapping.add_transformation(sq)
    mapping.add_transformation(exp)
    return mapping


# ---------------------------------------------------------------------------
# Session Execution Tests
# ---------------------------------------------------------------------------


class TestSessionExecution:
    def test_successful_session(self, sample_drug_csv: Path, tmp_path: Path) -> None:
        output = tmp_path / "output.csv"
        mapping = _build_simple_mapping(str(sample_drug_csv), str(output))

        session = Session(name="s_test", mapping=mapping)
        stats = session.execute()

        assert stats.status == SessionStatus.SUCCEEDED
        assert stats.source_rows_read > 0
        assert stats.target_rows_written > 0
        assert output.exists()
        assert stats.throughput_rows_per_sec > 0

    def test_session_with_parameters(self, sample_drug_csv: Path, tmp_path: Path) -> None:
        output = tmp_path / "output.csv"
        mapping = _build_simple_mapping(str(sample_drug_csv), str(output))
        mapping.add_parameter(MappingParameter("$$PROCESS_DATE", "2026-03-05"))

        session = Session(
            name="s_test_params",
            mapping=mapping,
            runtime_params={"$$PROCESS_DATE": "2026-03-01"},
        )
        stats = session.execute()
        assert stats.status == SessionStatus.SUCCEEDED

    def test_session_with_filter(self, sample_drug_csv: Path, tmp_path: Path) -> None:
        output = tmp_path / "output.csv"
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
        sq = SourceQualifier(name="SQ")
        fil = Filter(name="FIL", condition="supplier_code == 'SUP01'")

        mapping = Mapping(
            name="m_test_filter",
            sources=[source],
            targets=[target],
        )
        mapping.add_transformation(sq)
        mapping.add_transformation(fil)

        session = Session(name="s_filter", mapping=mapping)
        stats = session.execute()

        assert stats.status == SessionStatus.SUCCEEDED
        # Read output and verify filter applied
        result = pd.read_csv(output)
        assert all(result["supplier_code"] == "SUP01")

    def test_session_with_sequence_generator(self, sample_drug_csv: Path, tmp_path: Path) -> None:
        output = tmp_path / "output.csv"
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
        sq = SourceQualifier(name="SQ")
        seq = SequenceGenerator(
            name="SEQ",
            output_column="drug_key",
            start_value=1000,
        )

        mapping = Mapping(
            name="m_test_seq",
            sources=[source],
            targets=[target],
        )
        mapping.add_transformation(sq)
        mapping.add_transformation(seq)

        session = Session(name="s_seq", mapping=mapping)
        stats = session.execute()

        assert stats.status == SessionStatus.SUCCEEDED
        result = pd.read_csv(output)
        assert "drug_key" in result.columns
        assert result["drug_key"].iloc[0] == 1000

    def test_session_update_strategy(self, sample_drug_csv: Path, tmp_path: Path) -> None:
        output = tmp_path / "output.csv"
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
        sq = SourceQualifier(name="SQ")
        upd = UpdateStrategy(
            name="UPD",
            strategy_func=lambda d: d["supplier_code"].apply(
                lambda x: RowDisposition.DD_INSERT if x == "SUP01" else RowDisposition.DD_REJECT
            ),
        )

        mapping = Mapping(
            name="m_test_upd",
            sources=[source],
            targets=[target],
            session_config=SessionConfig(bad_file_path=str(tmp_path / "bad.csv")),
        )
        mapping.add_transformation(sq)
        mapping.add_transformation(upd)

        session = Session(name="s_upd", mapping=mapping)
        stats = session.execute()

        assert stats.status == SessionStatus.SUCCEEDED
        assert stats.target_rows_rejected > 0


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


class TestSessionErrorHandling:
    def test_missing_source_file_fails(self, tmp_path: Path) -> None:
        mapping = _build_simple_mapping(
            "/nonexistent/path.csv",
            str(tmp_path / "out.csv"),
        )
        session = Session(name="s_missing", mapping=mapping)

        with pytest.raises(Exception):
            session.execute()

        assert session.status == SessionStatus.FAILED

    def test_validation_failure(self, tmp_path: Path) -> None:
        # Empty mapping with no sources
        mapping = Mapping(name="m_empty")
        session = Session(name="s_invalid", mapping=mapping)

        with pytest.raises(ValueError, match="at least one source"):
            session.execute()

    def test_error_threshold(self, sample_drug_csv: Path, tmp_path: Path) -> None:
        """Verify that error threshold of 0 stops on first error."""
        output = tmp_path / "output.csv"
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
        # This expression will fail because column doesn't exist
        bad_exp = Expression(name="EXP_BAD")
        bad_exp.add_expression("fail", lambda d: d["totally_missing_column"])

        mapping = Mapping(
            name="m_error",
            sources=[source],
            targets=[target],
            session_config=SessionConfig(error_threshold=0),
        )
        mapping.add_transformation(bad_exp)

        session = Session(name="s_error", mapping=mapping)
        with pytest.raises(KeyError):
            session.execute()

        assert session.status == SessionStatus.FAILED


# ---------------------------------------------------------------------------
# Performance Stats Tests
# ---------------------------------------------------------------------------


class TestPerformanceStats:
    def test_stats_summary(self) -> None:
        stats = PerformanceStats(
            session_name="s_test",
            mapping_name="m_test",
            source_rows_read=1000,
            target_rows_written=950,
            target_rows_rejected=50,
        )
        summary = stats.summary()
        assert summary["source_rows"] == 1000
        assert summary["target_written"] == 950
        assert summary["rejected"] == 50


# ---------------------------------------------------------------------------
# Checkpoint Tests
# ---------------------------------------------------------------------------


class TestSessionCheckpoint:
    def test_save_and_load(self, tmp_path: Path) -> None:
        checkpoint = SessionCheckpoint(
            last_committed_row=5000,
            last_committed_target="dim_drug",
            checkpoint_time="2026-03-05T10:00:00Z",
            params_snapshot={"$$DATE": "2026-03-05"},
        )
        path = tmp_path / "checkpoint.json"
        checkpoint.save(path)

        loaded = SessionCheckpoint.load(path)
        assert loaded.last_committed_row == 5000
        assert loaded.last_committed_target == "dim_drug"
        assert loaded.params_snapshot["$$DATE"] == "2026-03-05"
