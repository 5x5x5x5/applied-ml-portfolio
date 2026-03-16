"""Tests for clinical trial data generator."""

import numpy as np
import pandas as pd
import pytest

from clinical_eda.data_generator import (
    generate_biomarkers,
    generate_demographics,
    generate_trial_dataset,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


class TestGenerateDemographics:
    def test_returns_dataframe(self, rng: np.random.Generator) -> None:
        df = generate_demographics(100, rng)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100

    def test_age_bounds(self, rng: np.random.Generator) -> None:
        df = generate_demographics(500, rng)
        assert df["age"].min() >= 18
        assert df["age"].max() <= 85

    def test_bmi_bounds(self, rng: np.random.Generator) -> None:
        df = generate_demographics(500, rng)
        assert df["bmi"].min() >= 16
        assert df["bmi"].max() <= 55

    def test_expected_columns(self, rng: np.random.Generator) -> None:
        df = generate_demographics(10, rng)
        expected = {
            "age",
            "sex",
            "race",
            "bmi",
            "smoking_status",
            "disease_duration_years",
        }
        assert set(df.columns) == expected


class TestGenerateBiomarkers:
    def test_returns_dataframe(self, rng: np.random.Generator) -> None:
        df = generate_biomarkers(100, rng)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100

    def test_positive_values(self, rng: np.random.Generator) -> None:
        df = generate_biomarkers(500, rng)
        for col in df.columns:
            assert (df[col] > 0).all(), f"{col} has non-positive values"


class TestGenerateTrialDataset:
    def test_full_dataset_shape(self) -> None:
        df = generate_trial_dataset(n_patients=200, seed=123)
        assert len(df) == 200
        assert df.shape[1] > 15

    def test_treatment_arms_present(self) -> None:
        df = generate_trial_dataset(n_patients=200)
        arms = set(df["treatment_arm"].unique())
        assert arms == {"RX-7281", "Placebo"}

    def test_responder_binary(self) -> None:
        df = generate_trial_dataset(n_patients=200)
        assert set(df["responder"].unique()).issubset({0, 1})

    def test_patient_ids(self) -> None:
        df = generate_trial_dataset(n_patients=50)
        assert df.index[0] == "PT-0001"
        assert df.index[-1] == "PT-0050"

    def test_reproducible(self) -> None:
        df1 = generate_trial_dataset(100, seed=99)
        df2 = generate_trial_dataset(100, seed=99)
        pd.testing.assert_frame_equal(df1, df2)
