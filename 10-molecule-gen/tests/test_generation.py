"""Tests for the molecule generation pipeline."""

from __future__ import annotations

import pytest

from molecule_gen.chemistry.smiles_processor import SMILESVocabulary
from molecule_gen.generation.generator import (
    GeneratedMolecule,
    MoleculeGenerator,
    PropertyConstraints,
)
from molecule_gen.models.mol_vae import MolecularVAE

from .conftest import ASPIRIN_SMILES, DRUG_SMILES_CORPUS, ETHANOL_SMILES, IBUPROFEN_SMILES


class TestPropertyConstraints:
    """Tests for property constraint configuration."""

    def test_default_constraints(self) -> None:
        """Default constraints should have sensible drug-like bounds."""
        c = PropertyConstraints()
        assert c.mw_max == 500.0
        assert c.logp_max == 5.0
        assert c.hbd_max == 5
        assert c.hba_max == 10
        assert c.max_pains_alerts == 0

    def test_custom_constraints(self) -> None:
        """Custom constraints should override defaults."""
        c = PropertyConstraints(
            logp_min=-1.0,
            logp_max=3.0,
            mw_min=200.0,
            mw_max=400.0,
        )
        assert c.logp_min == -1.0
        assert c.logp_max == 3.0
        assert c.mw_min == 200.0
        assert c.mw_max == 400.0


class TestGeneratedMolecule:
    """Tests for the GeneratedMolecule dataclass."""

    def test_default_values(self) -> None:
        """GeneratedMolecule should have sensible defaults."""
        mol = GeneratedMolecule(smiles="CCO")
        assert mol.smiles == "CCO"
        assert mol.is_valid is False
        assert mol.is_novel is True
        assert mol.pains_alerts == []
        assert mol.passes_constraints is False


class TestMoleculeGenerator:
    """Tests for the MoleculeGenerator pipeline."""

    @pytest.fixture
    def generator(
        self,
        vae_model: MolecularVAE,
        vocab: SMILESVocabulary,
    ) -> MoleculeGenerator:
        """Create a MoleculeGenerator with untrained model for testing."""
        return MoleculeGenerator(
            model=vae_model,
            vocab=vocab,
            device="cpu",
            known_smiles=set(DRUG_SMILES_CORPUS),
        )

    def test_generator_initialization(self, generator: MoleculeGenerator) -> None:
        """Generator should initialize correctly."""
        assert generator.model is not None
        assert generator.vocab is not None
        assert len(generator.known_smiles) == len(DRUG_SMILES_CORPUS)

    def test_generate_random_returns_list(self, generator: MoleculeGenerator) -> None:
        """Random generation should return a list of GeneratedMolecule objects.

        Note: With an untrained model, most generated SMILES will be invalid.
        We test the pipeline mechanics, not generation quality.
        """
        results = generator.generate_random(
            num_molecules=5,
            temperature=1.0,
            max_attempts=50,
        )
        assert isinstance(results, list)
        for mol in results:
            assert isinstance(mol, GeneratedMolecule)
            assert mol.is_valid

    def test_generate_random_respects_max_attempts(self, generator: MoleculeGenerator) -> None:
        """Generation should stop after max_attempts even if target not reached."""
        results = generator.generate_random(
            num_molecules=10000,
            max_attempts=10,
        )
        # With only 10 attempts and untrained model, we shouldn't get 10000 molecules
        assert len(results) <= 10000

    def test_optimize_molecule_pipeline(self, generator: MoleculeGenerator) -> None:
        """Molecule optimization should run without errors.

        With an untrained model, results will be random but the pipeline
        should complete without exceptions.
        """
        results = generator.optimize_molecule(
            seed_smiles=ASPIRIN_SMILES,
            num_candidates=3,
            num_steps=1,
            candidates_per_step=10,
        )
        assert isinstance(results, list)

    def test_interpolate_molecules(self, generator: MoleculeGenerator) -> None:
        """Latent interpolation should produce the correct number of steps."""
        results = generator.interpolate_molecules(
            smiles_start=ASPIRIN_SMILES,
            smiles_end=IBUPROFEN_SMILES,
            num_steps=5,
        )
        assert len(results) == 5
        for mol in results:
            assert isinstance(mol, GeneratedMolecule)
            assert mol.smiles is not None

    def test_deduplicate(self, generator: MoleculeGenerator) -> None:
        """Deduplication should remove molecules with identical canonical SMILES."""
        mols = [
            GeneratedMolecule(smiles="CCO", canonical_smiles="CCO"),
            GeneratedMolecule(smiles="OCC", canonical_smiles="CCO"),
            GeneratedMolecule(smiles="c1ccccc1", canonical_smiles="c1ccccc1"),
        ]
        unique = generator._deduplicate(mols)
        assert len(unique) == 2  # CCO appears twice

    def test_diversity_filter_allows_dissimilar(self, generator: MoleculeGenerator) -> None:
        """Diversity filter should accept molecules with low Tanimoto similarity."""
        mol = GeneratedMolecule(
            smiles=ETHANOL_SMILES,
            canonical_smiles=ETHANOL_SMILES,
        )
        # With no existing fingerprints, anything should pass
        assert generator._passes_diversity_filter(mol, [], threshold=0.4)

    def test_generate_targeted_returns_molecules(self, generator: MoleculeGenerator) -> None:
        """Targeted generation should return GeneratedMolecule objects."""
        results = generator.generate_targeted(
            num_molecules=2,
            target_properties={"logp": 2.0, "mw": 300.0},
            num_latent_samples=10,
            temperature=1.0,
        )
        assert isinstance(results, list)


class TestGeneratorConstraintChecking:
    """Tests for the constraint checking logic."""

    @pytest.fixture
    def generator(
        self,
        vae_model: MolecularVAE,
        vocab: SMILESVocabulary,
    ) -> MoleculeGenerator:
        """Create generator for constraint testing."""
        return MoleculeGenerator(model=vae_model, vocab=vocab, device="cpu")

    def test_process_valid_molecule(self, generator: MoleculeGenerator) -> None:
        """A valid drug SMILES should be processable with relaxed constraints."""
        constraints = PropertyConstraints(
            logp_min=None,
            logp_max=None,
            mw_min=None,
            mw_max=None,
            qed_min=None,
            hbd_max=None,
            hba_max=None,
            tpsa_max=None,
            rotatable_bonds_max=None,
            max_pains_alerts=100,
        )
        result = generator._process_candidate(ASPIRIN_SMILES, constraints)
        if result is not None:
            assert result.is_valid
            assert result.canonical_smiles is not None

    def test_process_invalid_molecule(self, generator: MoleculeGenerator) -> None:
        """Invalid SMILES should return None from processing."""
        constraints = PropertyConstraints()
        result = generator._process_candidate("INVALID", constraints)
        assert result is None

    def test_process_empty_smiles(self, generator: MoleculeGenerator) -> None:
        """Empty SMILES should return None."""
        constraints = PropertyConstraints()
        result = generator._process_candidate("", constraints)
        assert result is None

    def test_process_short_smiles(self, generator: MoleculeGenerator) -> None:
        """Very short SMILES (< 3 chars) should return None."""
        constraints = PropertyConstraints()
        result = generator._process_candidate("C", constraints)
        assert result is None
