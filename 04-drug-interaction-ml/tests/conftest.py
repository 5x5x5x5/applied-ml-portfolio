"""Shared test fixtures for DrugInteractionML tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Molecular feature fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_smiles() -> list[str]:
    """Valid SMILES strings for common drugs."""
    return [
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CC(=O)Nc1ccc(O)cc1",  # Acetaminophen (Paracetamol)
        "CC12CCC3C(CCC4CC(=O)CCC34C)C1CCC2O",  # Testosterone
        "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",  # Diclofenac
    ]


@pytest.fixture
def sample_drug_pairs(sample_smiles: list[str]) -> list[tuple[str, str]]:
    """Pairs of drug SMILES for pairwise feature extraction."""
    return [
        (sample_smiles[0], sample_smiles[1]),  # Aspirin + Acetaminophen
        (sample_smiles[0], sample_smiles[3]),  # Aspirin + Diclofenac
        (sample_smiles[1], sample_smiles[2]),  # Acetaminophen + Testosterone
    ]


@pytest.fixture
def invalid_smiles() -> str:
    """An invalid SMILES string."""
    return "INVALID_SMILES_XYZ"


# ---------------------------------------------------------------------------
# Feature DataFrame fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    """Synthetic feature DataFrame for training/testing."""
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame(
        {
            "mw_diff": rng.normal(100, 30, n),
            "logp_diff": rng.normal(1.5, 0.8, n),
            "tpsa_diff": rng.normal(20, 10, n),
            "tanimoto_similarity": rng.uniform(0, 1, n),
            "co_prescribed_patients": rng.poisson(50, n).astype(float),
            "avg_overlap_days": rng.exponential(10, n),
            "avg_age": rng.normal(55, 15, n),
            "pct_female": rng.uniform(0.3, 0.7, n),
            "ae_count": rng.poisson(3, n).astype(float),
            "prr": rng.exponential(1.5, n),
        }
    )


@pytest.fixture
def sample_severity_labels() -> pd.Series:
    """Synthetic severity labels."""
    rng = np.random.RandomState(42)
    labels = rng.choice(["none", "mild", "moderate", "severe"], size=200, p=[0.4, 0.3, 0.2, 0.1])
    return pd.Series(labels, name="severity")


@pytest.fixture
def sample_type_labels() -> pd.Series:
    """Synthetic interaction type labels."""
    rng = np.random.RandomState(42)
    labels = rng.choice(["pharmacokinetic", "pharmacodynamic"], size=200, p=[0.6, 0.4])
    return pd.Series(labels, name="interaction_type")


# ---------------------------------------------------------------------------
# Drift detection fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def baseline_features() -> pd.DataFrame:
    """Baseline (training) feature distribution."""
    rng = np.random.RandomState(42)
    n = 500
    return pd.DataFrame(
        {
            "feature_a": rng.normal(10, 2, n),
            "feature_b": rng.normal(50, 10, n),
            "feature_c": rng.exponential(5, n),
            "feature_d": rng.uniform(0, 1, n),
        }
    )


@pytest.fixture
def current_features_no_drift(baseline_features: pd.DataFrame) -> pd.DataFrame:
    """Production features with no significant drift."""
    rng = np.random.RandomState(99)
    n = 500
    return pd.DataFrame(
        {
            "feature_a": rng.normal(10.1, 2.05, n),
            "feature_b": rng.normal(50.2, 10.1, n),
            "feature_c": rng.exponential(5.1, n),
            "feature_d": rng.uniform(0, 1, n),
        }
    )


@pytest.fixture
def current_features_with_drift(baseline_features: pd.DataFrame) -> pd.DataFrame:
    """Production features with significant drift on some features."""
    rng = np.random.RandomState(99)
    n = 500
    return pd.DataFrame(
        {
            "feature_a": rng.normal(15, 4, n),  # shifted mean and std
            "feature_b": rng.normal(80, 20, n),  # large shift
            "feature_c": rng.exponential(5.1, n),  # minimal drift
            "feature_d": rng.uniform(0, 1, n),  # no drift
        }
    )


# ---------------------------------------------------------------------------
# AWS / Snowflake mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_snowflake_connection() -> MagicMock:
    """Mocked Snowflake connection."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.description = [("col1",), ("col2",), ("col3",)]
    mock_cursor.fetchall.return_value = [
        (1, "val_a", 10.0),
        (2, "val_b", 20.0),
    ]
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn


@pytest.fixture
def mock_boto3_sagemaker() -> MagicMock:
    """Mocked boto3 SageMaker client."""
    mock_client = MagicMock()
    mock_client.create_model.return_value = {
        "ModelArn": "arn:aws:sagemaker:us-east-1:123:model/test"
    }
    mock_client.create_endpoint_config.return_value = {}
    mock_client.create_endpoint.return_value = {
        "EndpointArn": "arn:aws:sagemaker:us-east-1:123:endpoint/test"
    }
    mock_client.describe_endpoint.return_value = {
        "EndpointStatus": "InService",
        "EndpointConfigName": "test-config",
    }
    mock_client.exceptions = MagicMock()
    mock_client.exceptions.ClientError = Exception
    return mock_client
