"""Shared pytest fixtures for FeatureForge tests.

Provides mock Snowflake connections, sample DataFrames, and
pre-configured extractor/detector instances for unit testing
without a live Snowflake environment.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from feature_forge.extractors.structured_extractor import SnowflakeConfig

# ---------------------------------------------------------------------------
# Snowflake configuration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def snowflake_config() -> SnowflakeConfig:
    """Return a test Snowflake configuration."""
    return SnowflakeConfig(
        account="test_account",
        user="test_user",
        password="test_password",
        warehouse="TEST_WH",
        database="TEST_DB",
        schema="TEST_SCHEMA",
        role="TEST_ROLE",
    )


@pytest.fixture()
def mock_snowflake_connection() -> MagicMock:
    """Return a mock Snowflake connection object."""
    conn = MagicMock()
    conn.is_closed.return_value = False
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    cursor.fetchall.return_value = []
    return conn


@pytest.fixture()
def mock_snowflake_connect(mock_snowflake_connection: MagicMock) -> Any:
    """Patch snowflake.connector.connect to return a mock connection."""
    with patch(
        "snowflake.connector.connect", return_value=mock_snowflake_connection
    ) as mock_connect:
        yield mock_connect


# ---------------------------------------------------------------------------
# Sample DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_demographics_df() -> pd.DataFrame:
    """Sample patient demographics DataFrame."""
    return pd.DataFrame(
        {
            "patient_id": ["P001", "P002", "P003", "P004", "P005"],
            "age": [45, 62, 28, 71, 55],
            "gender": ["M", "F", "F", "M", "F"],
            "insurance_type": ["COMMERCIAL", "MEDICARE", "MEDICAID", "MEDICARE", "COMMERCIAL"],
            "chronic_condition_count": [2, 5, 0, 8, 3],
            "has_diabetes": [1, 1, 0, 1, 0],
            "has_hypertension": [1, 1, 0, 1, 1],
            "total_encounters_12m": [3, 12, 1, 20, 6],
            "ed_visit_count_12m": [0, 3, 0, 5, 1],
            "feature_ts": [datetime(2025, 1, 1)] * 5,
        }
    )


@pytest.fixture()
def sample_lab_features_df() -> pd.DataFrame:
    """Sample lab features DataFrame."""
    return pd.DataFrame(
        {
            "patient_id": ["P001", "P002", "P003", "P004", "P005"],
            "hemoglobin_mean": [14.2, 11.5, 13.8, 9.2, 12.1],
            "hemoglobin_std": [0.5, 1.2, 0.3, 2.1, 0.8],
            "hematocrit_mean": [42.0, 34.5, 41.0, 28.0, 36.0],
            "creatinine_mean": [0.9, 1.8, 0.7, 3.2, 1.1],
            "bun_mean": [15.0, 32.0, 12.0, 45.0, 18.0],
            "glucose_mean": [95.0, 180.0, 88.0, 220.0, 110.0],
            "feature_ts": [datetime(2025, 1, 1)] * 5,
        }
    )


@pytest.fixture()
def sample_clinical_notes_df() -> pd.DataFrame:
    """Sample clinical notes features DataFrame."""
    return pd.DataFrame(
        {
            "patient_id": ["P001", "P002", "P003"],
            "encounter_count": [3, 8, 1],
            "total_notes": [5, 15, 2],
            "avg_note_length": [1200.0, 2500.0, 800.0],
            "unique_diagnosis_count": [4, 12, 1],
            "urgent_note_count": [0, 2, 0],
            "feature_ts": [datetime(2025, 1, 1)] * 3,
        }
    )


@pytest.fixture()
def baseline_distribution() -> np.ndarray:
    """Sample baseline distribution for drift testing."""
    rng = np.random.default_rng(42)
    return rng.normal(loc=100, scale=15, size=1000)


@pytest.fixture()
def drifted_distribution() -> np.ndarray:
    """Sample drifted distribution (shifted mean)."""
    rng = np.random.default_rng(42)
    return rng.normal(loc=115, scale=18, size=1000)


@pytest.fixture()
def non_drifted_distribution() -> np.ndarray:
    """Sample non-drifted distribution (similar to baseline)."""
    rng = np.random.default_rng(99)
    return rng.normal(loc=101, scale=15, size=1000)


# ---------------------------------------------------------------------------
# Registry fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_feature_rows() -> list[dict[str, Any]]:
    """Sample rows as returned by Snowflake for feature registry queries."""
    return [
        {
            "FEATURE_ID": "abc123def456",
            "NAME": "hemoglobin_mean",
            "VERSION": 1,
            "DESCRIPTION": "Mean hemoglobin over 365 days",
            "DATA_TYPE": "FLOAT",
            "SOURCE_TABLE": "LAB_FEATURES",
            "SOURCE_QUERY": "",
            "ENTITY_KEY": "patient_id",
            "TIMESTAMP_COLUMN": "feature_ts",
            "FRESHNESS_SLA_HOURS": 24,
            "OWNER": "data-engineering",
            "TAGS": '["lab", "structured"]',
            "DEPENDENCIES": "[]",
            "STATUS": "ACTIVE",
            "CREATED_AT": datetime(2025, 1, 1),
            "UPDATED_AT": datetime(2025, 1, 1),
        }
    ]
