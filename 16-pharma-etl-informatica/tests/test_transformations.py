"""
Tests for Informatica-style transformation components.

Each test validates a specific transformation type against sample data,
verifying both the transformation logic and edge-case behavior.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pharma_flow.framework.transformations import (
    Aggregator,
    AggregatorMode,
    Expression,
    Filter,
    Joiner,
    JoinType,
    Lookup,
    LookupMode,
    Normalizer,
    Rank,
    Router,
    RowDisposition,
    SequenceGenerator,
    Sorter,
    SourceQualifier,
    UpdateStrategy,
    compute_row_hash,
)

# ---------------------------------------------------------------------------
# SourceQualifier
# ---------------------------------------------------------------------------


class TestSourceQualifier:
    def test_basic_passthrough(self, sample_drug_df: pd.DataFrame) -> None:
        sq = SourceQualifier(name="SQ_TEST")
        result = sq.execute(sample_drug_df)
        assert len(result) == len(sample_drug_df)

    def test_source_filter(self, sample_drug_df: pd.DataFrame) -> None:
        sq = SourceQualifier(
            name="SQ_FILTER",
            source_filter="supplier_code == 'SUP01'",
        )
        result = sq.execute(sample_drug_df)
        assert all(result["supplier_code"] == "SUP01")
        assert len(result) < len(sample_drug_df)

    def test_column_projection(self, sample_drug_df: pd.DataFrame) -> None:
        sq = SourceQualifier(
            name="SQ_PROJECT",
            select_columns=["drug_name", "ndc_code"],
        )
        result = sq.execute(sample_drug_df)
        assert list(result.columns) == ["drug_name", "ndc_code"]

    def test_distinct(self, sample_drug_df: pd.DataFrame) -> None:
        # Add duplicate row
        df = pd.concat([sample_drug_df, sample_drug_df.iloc[[0]]], ignore_index=True)
        sq = SourceQualifier(name="SQ_DISTINCT", distinct=True)
        result = sq.execute(df)
        assert len(result) == len(sample_drug_df)

    def test_join_additional_source(self) -> None:
        master = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
        detail = pd.DataFrame({"id": [1, 2], "value": [100, 200]})
        sq = SourceQualifier(name="SQ_JOIN", join_condition="id = id")
        sq.add_source(detail)
        result = sq.execute(master)
        assert len(result) == 2  # Inner join


# ---------------------------------------------------------------------------
# Expression
# ---------------------------------------------------------------------------


class TestExpression:
    def test_derived_column(self) -> None:
        df = pd.DataFrame({"price": [10.0, 20.0, 30.0], "qty": [2, 3, 1]})
        exp = Expression(name="EXP_TEST")
        exp.add_expression("total", lambda d: d["price"] * d["qty"])
        result = exp.execute(df)
        assert "total" in result.columns
        assert list(result["total"]) == [20.0, 60.0, 30.0]

    def test_string_transform(self, sample_drug_df: pd.DataFrame) -> None:
        exp = Expression(name="EXP_UPPER")
        exp.add_expression("drug_upper", lambda d: d["drug_name"].str.upper().str.strip())
        result = exp.execute(sample_drug_df)
        assert result["drug_upper"].iloc[0] == "LIPITOR"
        assert result["drug_upper"].iloc[1] == "ATORVASTATIN CALCIUM"

    def test_multiple_expressions(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        exp = Expression(name="EXP_MULTI")
        exp.add_expression("sum_ab", lambda d: d["a"] + d["b"])
        exp.add_expression("prod_ab", lambda d: d["a"] * d["b"])
        result = exp.execute(df)
        assert list(result["sum_ab"]) == [4, 6]
        assert list(result["prod_ab"]) == [3, 8]

    def test_expression_error_raises(self) -> None:
        df = pd.DataFrame({"a": [1, 2]})
        exp = Expression(name="EXP_ERR")
        exp.add_expression("bad", lambda d: d["nonexistent_col"])
        with pytest.raises(KeyError):
            exp.execute(df)


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


class TestFilter:
    def test_string_condition(self) -> None:
        df = pd.DataFrame({"value": [10, 20, 30, 40]})
        fil = Filter(name="FIL_TEST", condition="value > 20")
        result = fil.execute(df)
        assert len(result) == 2
        assert list(result["value"]) == [30, 40]

    def test_callable_condition(self) -> None:
        df = pd.DataFrame({"status": ["active", "inactive", "active"]})
        fil = Filter(
            name="FIL_FUNC",
            condition_func=lambda d: d["status"] == "active",
        )
        result = fil.execute(df)
        assert len(result) == 2

    def test_no_condition_passthrough(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        fil = Filter(name="FIL_NONE")
        result = fil.execute(df)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class TestAggregator:
    def test_group_by_sum(self) -> None:
        df = pd.DataFrame(
            {
                "category": ["A", "A", "B", "B", "B"],
                "amount": [10, 20, 30, 40, 50],
            }
        )
        agg = Aggregator(
            name="AGG_TEST",
            group_by=["category"],
            aggregations={"amount": "sum"},
        )
        result = agg.execute(df)
        assert len(result) == 2
        a_sum = result[result["category"] == "A"]["amount"].iloc[0]
        b_sum = result[result["category"] == "B"]["amount"].iloc[0]
        assert a_sum == 30
        assert b_sum == 120

    def test_multiple_aggregations(self) -> None:
        df = pd.DataFrame(
            {
                "grp": ["X", "X", "Y"],
                "val": [10, 20, 30],
            }
        )
        agg = Aggregator(
            name="AGG_MULTI",
            group_by=["grp"],
            aggregations={"val": ["sum", "mean"]},
        )
        result = agg.execute(df)
        assert len(result) == 2

    def test_sorted_mode_warning(self) -> None:
        df = pd.DataFrame({"grp": ["B", "A", "B"], "val": [1, 2, 3]})
        agg = Aggregator(
            name="AGG_SORTED",
            group_by=["grp"],
            aggregations={"val": "sum"},
            mode=AggregatorMode.SORTED,
        )
        # Should still work (just logs a warning)
        result = agg.execute(df)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Joiner
# ---------------------------------------------------------------------------


class TestJoiner:
    def test_inner_join(self) -> None:
        master = pd.DataFrame({"key": [1, 2, 3], "name": ["A", "B", "C"]})
        detail = pd.DataFrame({"key": [2, 3, 4], "desc": ["D2", "D3", "D4"]})
        jnr = Joiner(
            name="JNR_INNER",
            detail_source=detail,
            join_type=JoinType.INNER,
            master_keys=["key"],
            detail_keys=["key"],
        )
        result = jnr.execute(master)
        assert len(result) == 2
        assert set(result["name"]) == {"B", "C"}

    def test_left_outer_join(self) -> None:
        master = pd.DataFrame({"key": [1, 2, 3], "name": ["A", "B", "C"]})
        detail = pd.DataFrame({"key": [2], "desc": ["D2"]})
        jnr = Joiner(
            name="JNR_LEFT",
            detail_source=detail,
            join_type=JoinType.LEFT_OUTER,
            master_keys=["key"],
            detail_keys=["key"],
        )
        result = jnr.execute(master)
        assert len(result) == 3

    def test_master_only_anti_join(self) -> None:
        master = pd.DataFrame({"key": [1, 2, 3], "name": ["A", "B", "C"]})
        detail = pd.DataFrame({"key": [2, 3]})
        jnr = Joiner(
            name="JNR_MASTER_ONLY",
            detail_source=detail,
            join_type=JoinType.MASTER_ONLY,
            master_keys=["key"],
            detail_keys=["key"],
        )
        result = jnr.execute(master)
        assert len(result) == 1
        assert result["name"].iloc[0] == "A"

    def test_no_detail_raises(self) -> None:
        master = pd.DataFrame({"key": [1]})
        jnr = Joiner(name="JNR_NONE", master_keys=["key"], detail_keys=["key"])
        with pytest.raises(ValueError, match="no detail source"):
            jnr.execute(master)


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


class TestLookup:
    def test_connected_lookup(self) -> None:
        pipeline = pd.DataFrame({"code": ["A", "B", "C"]})
        ref = pd.DataFrame(
            {
                "code": ["A", "B"],
                "description": ["Alpha", "Beta"],
            }
        )
        lkp = Lookup(
            name="LKP_TEST",
            lookup_source=ref,
            lookup_keys=["code"],
            return_columns=["description"],
            mode=LookupMode.CONNECTED,
            default_values={"description": "Unknown"},
        )
        result = lkp.execute(pipeline)
        assert result["description"].iloc[0] == "Alpha"
        assert result["description"].iloc[1] == "Beta"
        assert result["description"].iloc[2] == "Unknown"

    def test_unconnected_lookup(self) -> None:
        ref = pd.DataFrame({"id": [1, 2], "name": ["One", "Two"]})
        lkp = Lookup(
            name="LKP_UNCONNECTED",
            lookup_source=ref,
            lookup_keys=["id"],
            return_columns=["name"],
            mode=LookupMode.UNCONNECTED,
        )
        lkp.build_cache()
        result = lkp.lookup_value((1,))
        assert result == {"name": "One"}

    def test_cache_miss_returns_defaults(self) -> None:
        ref = pd.DataFrame({"id": [1], "name": ["One"]})
        lkp = Lookup(
            name="LKP_MISS",
            lookup_source=ref,
            lookup_keys=["id"],
            return_columns=["name"],
            default_values={"name": "N/A"},
        )
        lkp.build_cache()
        result = lkp.lookup_value((999,))
        assert result == {"name": "N/A"}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class TestRouter:
    def test_basic_routing(self) -> None:
        df = pd.DataFrame({"status": ["new", "changed", "new", "unchanged"], "id": [1, 2, 3, 4]})
        rtr = Router(name="RTR_TEST")
        rtr.add_group("NEW", condition_func=lambda d: d["status"] == "new")
        rtr.add_group("CHANGED", condition_func=lambda d: d["status"] == "changed")

        result = rtr.execute(df)
        # First group (NEW) is returned as pipeline continuation
        assert len(result) == 2

        new_group = rtr.get_group("NEW")
        assert len(new_group) == 2

        changed_group = rtr.get_group("CHANGED")
        assert len(changed_group) == 1

        default_group = rtr.get_group("DEFAULT")
        assert len(default_group) == 1  # "unchanged" row

    def test_missing_group_raises(self) -> None:
        rtr = Router(name="RTR_EMPTY")
        rtr.execute(pd.DataFrame({"a": [1]}))
        with pytest.raises(KeyError):
            rtr.get_group("NONEXISTENT")


# ---------------------------------------------------------------------------
# Sorter
# ---------------------------------------------------------------------------


class TestSorter:
    def test_ascending_sort(self) -> None:
        df = pd.DataFrame({"name": ["Charlie", "Alice", "Bob"]})
        srt = Sorter(name="SRT_TEST", sort_keys=["name"], ascending=[True])
        result = srt.execute(df)
        assert list(result["name"]) == ["Alice", "Bob", "Charlie"]

    def test_descending_sort(self) -> None:
        df = pd.DataFrame({"val": [1, 3, 2]})
        srt = Sorter(name="SRT_DESC", sort_keys=["val"], ascending=[False])
        result = srt.execute(df)
        assert list(result["val"]) == [3, 2, 1]

    def test_distinct_sort(self) -> None:
        df = pd.DataFrame({"val": [1, 2, 1, 3]})
        srt = Sorter(name="SRT_DISTINCT", sort_keys=["val"], distinct=True)
        result = srt.execute(df)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# UpdateStrategy
# ---------------------------------------------------------------------------


class TestUpdateStrategy:
    def test_default_insert(self) -> None:
        df = pd.DataFrame({"id": [1, 2]})
        upd = UpdateStrategy(name="UPD_TEST")
        result = upd.execute(df)
        assert all(result["_disposition"] == RowDisposition.DD_INSERT)

    def test_custom_strategy(self) -> None:
        df = pd.DataFrame({"id": [1, 2, 3], "exists": [False, True, True]})
        upd = UpdateStrategy(
            name="UPD_CUSTOM",
            strategy_func=lambda d: d["exists"].apply(
                lambda x: RowDisposition.DD_UPDATE if x else RowDisposition.DD_INSERT
            ),
        )
        result = upd.execute(df)
        assert result["_disposition"].iloc[0] == RowDisposition.DD_INSERT
        assert result["_disposition"].iloc[1] == RowDisposition.DD_UPDATE


# ---------------------------------------------------------------------------
# SequenceGenerator
# ---------------------------------------------------------------------------


class TestSequenceGenerator:
    def test_basic_sequence(self) -> None:
        df = pd.DataFrame({"name": ["A", "B", "C"]})
        seq = SequenceGenerator(
            name="SEQ_TEST",
            output_column="sk",
            start_value=100,
            increment=1,
        )
        result = seq.execute(df)
        assert list(result["sk"]) == [100, 101, 102]

    def test_increment_by_10(self) -> None:
        df = pd.DataFrame({"x": [1, 2]})
        seq = SequenceGenerator(
            name="SEQ_INC10",
            output_column="key",
            start_value=1,
            increment=10,
        )
        result = seq.execute(df)
        assert list(result["key"]) == [1, 11]

    def test_overflow_raises(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        seq = SequenceGenerator(
            name="SEQ_OVERFLOW",
            start_value=1,
            increment=1,
            end_value=2,
            cycle=False,
        )
        with pytest.raises(OverflowError):
            seq.execute(df)

    def test_cycle(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3, 4]})
        seq = SequenceGenerator(
            name="SEQ_CYCLE",
            output_column="key",
            start_value=1,
            increment=1,
            end_value=2,
            cycle=True,
        )
        result = seq.execute(df)
        assert list(result["key"]) == [1, 2, 1, 2]


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------


class TestNormalizer:
    def test_normalize_columns(self) -> None:
        df = pd.DataFrame(
            {
                "patient_id": [1, 2],
                "drug_1": ["Aspirin", "Metformin"],
                "drug_2": ["Ibuprofen", None],
                "drug_3": [None, "Lisinopril"],
            }
        )
        nrm = Normalizer(
            name="NRM_TEST",
            group_columns=["drug_1", "drug_2", "drug_3"],
            normalized_column="drug",
            id_columns=["patient_id"],
        )
        result = nrm.execute(df)
        # Patient 1: drug_1 + drug_2 = 2 rows, Patient 2: drug_1 + drug_3 = 2 rows
        assert len(result) == 4
        assert "drug" in result.columns
        assert "occurrence" in result.columns

    def test_empty_groups_passthrough(self) -> None:
        df = pd.DataFrame({"a": [1]})
        nrm = Normalizer(name="NRM_EMPTY")
        result = nrm.execute(df)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Rank
# ---------------------------------------------------------------------------


class TestRank:
    def test_top_1_per_group(self) -> None:
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "score": [90, 95, 80, 85],
            }
        )
        rnk = Rank(
            name="RNK_TEST",
            group_by=["group"],
            rank_column="score",
            top_n=1,
            ascending=False,  # top = highest
        )
        result = rnk.execute(df)
        assert len(result) == 2
        a_score = result[result["group"] == "A"]["score"].iloc[0]
        assert a_score == 95

    def test_no_rank_column_raises(self) -> None:
        df = pd.DataFrame({"a": [1]})
        rnk = Rank(name="RNK_ERR")
        with pytest.raises(ValueError, match="rank_column is required"):
            rnk.execute(df)


# ---------------------------------------------------------------------------
# Utility: compute_row_hash
# ---------------------------------------------------------------------------


class TestComputeRowHash:
    def test_same_data_same_hash(self) -> None:
        df = pd.DataFrame({"a": ["x", "x"], "b": [1, 1]})
        hashes = compute_row_hash(df, ["a", "b"])
        assert hashes.iloc[0] == hashes.iloc[1]

    def test_different_data_different_hash(self) -> None:
        df = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
        hashes = compute_row_hash(df, ["a", "b"])
        assert hashes.iloc[0] != hashes.iloc[1]
