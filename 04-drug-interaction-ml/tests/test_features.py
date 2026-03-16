"""Tests for feature extraction modules."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Molecular feature tests
# ---------------------------------------------------------------------------


class TestMolecularFeatureExtractor:
    """Tests for MolecularFeatureExtractor."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self) -> None:
        """Set up RDKit mocks for all tests in this class."""
        # Create mock molecules with realistic return values
        self.mock_mol = MagicMock()
        self.mock_mol.GetNumHeavyAtoms.return_value = 13

        self.mock_fp = MagicMock()
        self.mock_fp.GetOnBits.return_value = [0, 5, 10, 42, 100]

        self.mock_chem = MagicMock()
        self.mock_chem.MolFromSmiles.return_value = self.mock_mol

        self.mock_allchem = MagicMock()
        self.mock_allchem.GetMorganFingerprintAsBitVect.return_value = self.mock_fp

        self.mock_descriptors = MagicMock()
        self.mock_descriptors.MolWt.return_value = 180.16
        self.mock_descriptors.MolLogP.return_value = 1.31
        self.mock_descriptors.TPSA.return_value = 63.6

        self.mock_rdmol = MagicMock()
        self.mock_rdmol.CalcNumHBD.return_value = 2
        self.mock_rdmol.CalcNumHBA.return_value = 4
        self.mock_rdmol.CalcNumRotatableBonds.return_value = 3
        self.mock_rdmol.CalcNumRings.return_value = 1
        self.mock_rdmol.CalcNumAromaticRings.return_value = 1
        self.mock_rdmol.CalcFractionCSP3.return_value = 0.25

        self._patches = [
            patch.dict(
                "sys.modules",
                {
                    "rdkit": MagicMock(),
                    "rdkit.Chem": self.mock_chem,
                    "rdkit.Chem.AllChem": self.mock_allchem,
                    "rdkit.Chem.Descriptors": self.mock_descriptors,
                    "rdkit.Chem.rdMolDescriptors": self.mock_rdmol,
                    "rdkit.DataStructs": MagicMock(),
                },
            ),
        ]
        for p in self._patches:
            p.start()

    @pytest.fixture(autouse=True)
    def _teardown_mocks(self) -> None:
        yield
        for p in self._patches:
            p.stop()

    def test_compute_descriptors_returns_valid_model(self) -> None:
        """Descriptor computation returns a MolecularDescriptors instance."""
        from drug_interaction.features.molecular_features import MolecularFeatureExtractor

        extractor = MolecularFeatureExtractor.__new__(MolecularFeatureExtractor)
        extractor.fingerprint_radius = 2
        extractor.fingerprint_nbits = 1024
        extractor._rdkit = {
            "Chem": self.mock_chem,
            "AllChem": self.mock_allchem,
            "Descriptors": self.mock_descriptors,
            "rdMolDescriptors": self.mock_rdmol,
        }

        result = extractor.compute_descriptors("CC(=O)Oc1ccccc1C(=O)O")

        assert result.molecular_weight == 180.16
        assert result.logp == 1.31
        assert result.hbd == 2
        assert result.hba == 4
        assert result.tpsa == 63.6
        assert result.rotatable_bonds == 3
        assert result.num_rings == 1

    def test_compute_descriptors_invalid_smiles_raises(self) -> None:
        """Invalid SMILES raises ValueError."""
        from drug_interaction.features.molecular_features import MolecularFeatureExtractor

        self.mock_chem.MolFromSmiles.return_value = None

        extractor = MolecularFeatureExtractor.__new__(MolecularFeatureExtractor)
        extractor.fingerprint_radius = 2
        extractor.fingerprint_nbits = 1024
        extractor._rdkit = {
            "Chem": self.mock_chem,
            "AllChem": self.mock_allchem,
            "Descriptors": self.mock_descriptors,
            "rdMolDescriptors": self.mock_rdmol,
        }

        with pytest.raises(ValueError, match="Invalid SMILES"):
            extractor.compute_descriptors("INVALID")

    def test_compute_morgan_fingerprint_shape(self) -> None:
        """Morgan fingerprint has correct shape."""
        from drug_interaction.features.molecular_features import MolecularFeatureExtractor

        extractor = MolecularFeatureExtractor.__new__(MolecularFeatureExtractor)
        extractor.fingerprint_radius = 2
        extractor.fingerprint_nbits = 1024
        extractor._rdkit = {
            "Chem": self.mock_chem,
            "AllChem": self.mock_allchem,
            "Descriptors": self.mock_descriptors,
            "rdMolDescriptors": self.mock_rdmol,
        }

        fp = extractor.compute_morgan_fingerprint("CC(=O)O")

        assert fp.shape == (1024,)
        assert fp.dtype == np.int32
        assert fp[0] == 1
        assert fp[5] == 1
        assert fp[1] == 0  # not in on-bits

    def test_extract_batch_descriptors_returns_dataframe(self) -> None:
        """Batch extraction returns a DataFrame with correct columns."""
        from drug_interaction.features.molecular_features import MolecularFeatureExtractor

        extractor = MolecularFeatureExtractor.__new__(MolecularFeatureExtractor)
        extractor.fingerprint_radius = 2
        extractor.fingerprint_nbits = 1024
        extractor._rdkit = {
            "Chem": self.mock_chem,
            "AllChem": self.mock_allchem,
            "Descriptors": self.mock_descriptors,
            "rdMolDescriptors": self.mock_rdmol,
        }

        smiles_list = ["CC(=O)O", "c1ccccc1", "CCO"]
        df = extractor.extract_batch_descriptors(smiles_list)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "molecular_weight" in df.columns
        assert "logp" in df.columns
        assert "smiles" in df.columns

    def test_extract_batch_skips_invalid(self) -> None:
        """Invalid SMILES are skipped with a warning."""
        from drug_interaction.features.molecular_features import MolecularFeatureExtractor

        call_count = 0
        original_return = self.mock_chem.MolFromSmiles.return_value

        def side_effect(smi: str) -> MagicMock | None:
            nonlocal call_count
            call_count += 1
            if smi == "INVALID":
                return None
            return original_return

        self.mock_chem.MolFromSmiles.side_effect = side_effect

        extractor = MolecularFeatureExtractor.__new__(MolecularFeatureExtractor)
        extractor.fingerprint_radius = 2
        extractor.fingerprint_nbits = 1024
        extractor._rdkit = {
            "Chem": self.mock_chem,
            "AllChem": self.mock_allchem,
            "Descriptors": self.mock_descriptors,
            "rdMolDescriptors": self.mock_rdmol,
        }

        df = extractor.extract_batch_descriptors(["CC(=O)O", "INVALID", "CCO"])
        assert len(df) == 2


# ---------------------------------------------------------------------------
# Snowflake feature tests
# ---------------------------------------------------------------------------


class TestSnowflakeFeatureExtractor:
    """Tests for SnowflakeFeatureExtractor with mocked Snowflake connection."""

    @patch("snowflake.connector.connect")
    def test_extract_co_prescription_features(
        self, mock_connect: MagicMock, mock_snowflake_connection: MagicMock
    ) -> None:
        """Co-prescription extraction returns a DataFrame."""
        from drug_interaction.features.snowflake_features import (
            SnowflakeConfig,
            SnowflakeFeatureExtractor,
        )

        mock_connect.return_value = mock_snowflake_connection

        config = SnowflakeConfig(
            account="test_account",
            user="test_user",
            password="test_pass",
            warehouse="TEST_WH",
            database="TEST_DB",
            schema="TEST_SCHEMA",
        )
        extractor = SnowflakeFeatureExtractor(config=config)

        result = extractor.extract_co_prescription_features(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        mock_snowflake_connection.cursor.return_value.execute.assert_called_once()

    @patch("snowflake.connector.connect")
    def test_extract_all_features_returns_dict(
        self, mock_connect: MagicMock, mock_snowflake_connection: MagicMock
    ) -> None:
        """extract_all_features returns dict with all 4 feature sets."""
        from drug_interaction.features.snowflake_features import (
            SnowflakeConfig,
            SnowflakeFeatureExtractor,
        )

        mock_connect.return_value = mock_snowflake_connection

        config = SnowflakeConfig(
            account="test_account",
            user="test_user",
            password="test_pass",
            warehouse="TEST_WH",
            database="TEST_DB",
            schema="TEST_SCHEMA",
        )
        extractor = SnowflakeFeatureExtractor(config=config)

        result = extractor.extract_all_features(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )

        assert isinstance(result, dict)
        assert "co_prescription" in result
        assert "demographics" in result
        assert "adverse_events" in result
        assert "temporal_patterns" in result
        for key, df in result.items():
            assert isinstance(df, pd.DataFrame)

    @patch("snowflake.connector.connect")
    def test_connection_is_closed_after_query(
        self, mock_connect: MagicMock, mock_snowflake_connection: MagicMock
    ) -> None:
        """Snowflake connection is properly closed after query."""
        from drug_interaction.features.snowflake_features import (
            SnowflakeConfig,
            SnowflakeFeatureExtractor,
        )

        mock_connect.return_value = mock_snowflake_connection

        config = SnowflakeConfig(
            account="test_account",
            user="test_user",
            password="test_pass",
            warehouse="TEST_WH",
            database="TEST_DB",
            schema="TEST_SCHEMA",
        )
        extractor = SnowflakeFeatureExtractor(config=config)
        extractor.extract_patient_demographics(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
        )

        mock_snowflake_connection.close.assert_called_once()
        mock_snowflake_connection.cursor.return_value.close.assert_called_once()
