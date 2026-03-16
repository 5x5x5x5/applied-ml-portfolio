"""Tests for feature extractors (structured and semi-structured)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from feature_forge.extractors.semi_structured_extractor import (
    SemiStructuredFeatureExtractor,
)
from feature_forge.extractors.structured_extractor import (
    FeatureQuery,
    SnowflakeConfig,
    StructuredFeatureExtractor,
)


class TestStructuredFeatureExtractor:
    """Tests for StructuredFeatureExtractor."""

    def test_init(self, snowflake_config: SnowflakeConfig) -> None:
        """Extractor initialises with config and no connection."""
        extractor = StructuredFeatureExtractor(snowflake_config)
        assert extractor._config == snowflake_config
        assert extractor._conn is None

    def test_connect(
        self,
        snowflake_config: SnowflakeConfig,
        mock_snowflake_connect: Any,
    ) -> None:
        """Extractor connects to Snowflake with correct parameters."""
        extractor = StructuredFeatureExtractor(snowflake_config)
        conn = extractor.connect()

        mock_snowflake_connect.assert_called_once_with(
            account="test_account",
            user="test_user",
            password="test_password",
            warehouse="TEST_WH",
            database="TEST_DB",
            schema="TEST_SCHEMA",
            role="TEST_ROLE",
            login_timeout=30,
        )
        assert conn is not None

    def test_connect_reuses_existing(
        self,
        snowflake_config: SnowflakeConfig,
        mock_snowflake_connect: Any,
    ) -> None:
        """Subsequent connect() calls reuse the existing connection."""
        extractor = StructuredFeatureExtractor(snowflake_config)
        conn1 = extractor.connect()
        conn2 = extractor.connect()
        assert conn1 is conn2
        assert mock_snowflake_connect.call_count == 1

    def test_disconnect(
        self,
        snowflake_config: SnowflakeConfig,
        mock_snowflake_connect: Any,
        mock_snowflake_connection: MagicMock,
    ) -> None:
        """Disconnect closes the connection."""
        extractor = StructuredFeatureExtractor(snowflake_config)
        extractor.connect()
        extractor.disconnect()
        mock_snowflake_connection.close.assert_called_once()
        assert extractor._conn is None

    def test_context_manager(
        self,
        snowflake_config: SnowflakeConfig,
        mock_snowflake_connect: Any,
        mock_snowflake_connection: MagicMock,
    ) -> None:
        """Context manager connects on enter and disconnects on exit."""
        with StructuredFeatureExtractor(snowflake_config) as extractor:
            assert extractor._conn is not None
        mock_snowflake_connection.close.assert_called_once()

    def test_register_and_execute_query(
        self,
        snowflake_config: SnowflakeConfig,
        mock_snowflake_connect: Any,
        mock_snowflake_connection: MagicMock,
    ) -> None:
        """Registered queries can be executed by name."""
        cursor = mock_snowflake_connection.cursor.return_value
        cursor.fetchall.return_value = [{"patient_id": "P001", "value": 42.0}]

        extractor = StructuredFeatureExtractor(snowflake_config)
        query = FeatureQuery(
            name="test_query",
            description="A test query",
            sql="SELECT patient_id, value FROM test WHERE x = %(param)s",
            parameters={"param": "abc"},
        )
        extractor.register_query(query)
        df = extractor.execute_registered_query("test_query")

        assert not df.empty
        assert "patient_id" in df.columns

    def test_execute_unregistered_query_raises(
        self,
        snowflake_config: SnowflakeConfig,
    ) -> None:
        """Executing a non-existent registered query raises KeyError."""
        extractor = StructuredFeatureExtractor(snowflake_config)
        with pytest.raises(KeyError, match="no_such_query"):
            extractor.execute_registered_query("no_such_query")

    def test_post_process_demographics(self, sample_demographics_df: pd.DataFrame) -> None:
        """Demographics post-processing adds age buckets and gender encoding."""
        result = StructuredFeatureExtractor._post_process_demographics(sample_demographics_df)
        assert "age_bucket" in result.columns
        # Gender one-hot columns
        gender_cols = [c for c in result.columns if c.startswith("gender_")]
        assert len(gender_cols) >= 2

    def test_post_process_demographics_empty(self) -> None:
        """Post-processing handles empty DataFrames gracefully."""
        result = StructuredFeatureExtractor._post_process_demographics(pd.DataFrame())
        assert result.empty

    def test_post_process_lab_features(self, sample_lab_features_df: pd.DataFrame) -> None:
        """Lab post-processing computes derived ratio features."""
        result = StructuredFeatureExtractor._post_process_lab_features(sample_lab_features_df)
        assert "hgb_hct_ratio" in result.columns
        assert "bun_creatinine_ratio" in result.columns
        # Ratios should be positive
        assert (result["hgb_hct_ratio"].dropna() > 0).all()

    def test_post_process_lab_imputation(self) -> None:
        """Lab post-processing imputes NaN values with median."""
        df = pd.DataFrame(
            {
                "patient_id": ["P1", "P2", "P3"],
                "hemoglobin_mean": [14.0, np.nan, 12.0],
                "hematocrit_mean": [42.0, 35.0, np.nan],
            }
        )
        result = StructuredFeatureExtractor._post_process_lab_features(df)
        assert not result["hemoglobin_mean"].isna().any()


class TestSemiStructuredFeatureExtractor:
    """Tests for SemiStructuredFeatureExtractor."""

    def test_init(self, snowflake_config: SnowflakeConfig) -> None:
        """Semi-structured extractor initialises correctly."""
        extractor = SemiStructuredFeatureExtractor(snowflake_config)
        assert extractor._config == snowflake_config

    def test_connect(
        self,
        snowflake_config: SnowflakeConfig,
        mock_snowflake_connect: Any,
    ) -> None:
        """Connects to Snowflake."""
        extractor = SemiStructuredFeatureExtractor(snowflake_config)
        conn = extractor.connect()
        assert conn is not None

    def test_handle_missing_values_numeric(self) -> None:
        """Numeric nulls are imputed with median."""
        df = pd.DataFrame(
            {
                "patient_id": ["P1", "P2", "P3", "P4"],
                "score": [10.0, np.nan, 30.0, 20.0],
                "count": [1, 2, np.nan, 4],
            }
        )
        result = SemiStructuredFeatureExtractor._handle_missing_values(df)
        assert not result["score"].isna().any()
        assert not result["count"].isna().any()

    def test_handle_missing_values_categorical(self) -> None:
        """Categorical nulls are filled with 'UNKNOWN'."""
        df = pd.DataFrame(
            {
                "patient_id": ["P1", "P2", "P3"],
                "status": ["ACTIVE", None, "INACTIVE"],
            }
        )
        result = SemiStructuredFeatureExtractor._handle_missing_values(df)
        assert result["status"].iloc[1] == "UNKNOWN"

    def test_encode_categorical_low_cardinality(self) -> None:
        """Low-cardinality categoricals are one-hot encoded."""
        df = pd.DataFrame(
            {
                "patient_id": ["P1", "P2", "P3"],
                "smoking_status": ["NEVER", "CURRENT", "FORMER"],
            }
        )
        result = SemiStructuredFeatureExtractor._encode_categorical_variants(df)
        # Original column should be dropped, dummies should exist
        assert "smoking_status" not in result.columns
        encoded_cols = [c for c in result.columns if c.startswith("smoking_status_")]
        assert len(encoded_cols) == 3

    def test_encode_categorical_high_cardinality(self) -> None:
        """High-cardinality categoricals get frequency encoding."""
        categories = [f"CAT_{i}" for i in range(15)]
        df = pd.DataFrame(
            {
                "patient_id": [f"P{i}" for i in range(15)],
                "diagnosis": categories,
            }
        )
        result = SemiStructuredFeatureExtractor._encode_categorical_variants(df)
        assert "diagnosis_freq_enc" in result.columns
        assert "diagnosis" not in result.columns

    def test_encode_skips_patient_id(self) -> None:
        """Encoding skips patient_id and feature_ts columns."""
        df = pd.DataFrame(
            {
                "patient_id": ["P1", "P2"],
                "category": ["A", "B"],
            }
        )
        result = SemiStructuredFeatureExtractor._encode_categorical_variants(df)
        assert "patient_id" in result.columns

    def test_context_manager(
        self,
        snowflake_config: SnowflakeConfig,
        mock_snowflake_connect: Any,
        mock_snowflake_connection: MagicMock,
    ) -> None:
        """Context manager connects and disconnects."""
        with SemiStructuredFeatureExtractor(snowflake_config):
            pass
        mock_snowflake_connection.close.assert_called_once()
