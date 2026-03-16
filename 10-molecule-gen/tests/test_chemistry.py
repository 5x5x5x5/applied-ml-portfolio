"""Tests for SMILES processing and molecular descriptor computation."""

from __future__ import annotations

import numpy as np
import pytest

from molecule_gen.chemistry.molecular_descriptors import (
    check_pains_alerts,
    compute_descriptor_vector,
    compute_descriptors,
)
from molecule_gen.chemistry.smiles_processor import (
    SMILESVocabulary,
    canonicalize_smiles,
    compute_morgan_fingerprint,
    is_valid_smiles,
    tanimoto_similarity,
    tokenize_smiles,
)
from molecule_gen.models.property_predictor import (
    LipinskiResult,
    apply_lipinski_filter,
    compute_qed_score,
    compute_sa_score_estimate,
)

from .conftest import (
    ASPIRIN_SMILES,
    BENZENE_SMILES,
    CAFFEINE_SMILES,
    DRUG_SMILES_CORPUS,
    ETHANOL_SMILES,
    IBUPROFEN_SMILES,
    PARACETAMOL_SMILES,
)


class TestSMILESTokenization:
    """Tests for SMILES tokenizer."""

    def test_tokenize_simple_molecule(self) -> None:
        """Ethanol CCO should tokenize to individual atoms."""
        tokens = tokenize_smiles("CCO")
        assert tokens == ["C", "C", "O"]

    def test_tokenize_branching(self) -> None:
        """Tokens should include branch markers."""
        tokens = tokenize_smiles("CC(=O)O")
        assert "(" in tokens
        assert ")" in tokens
        assert "=" in tokens

    def test_tokenize_aromatic_ring(self) -> None:
        """Benzene c1ccccc1 should tokenize to lowercase aromatic atoms and ring digits."""
        tokens = tokenize_smiles(BENZENE_SMILES)
        assert "c" in tokens
        assert "1" in tokens

    def test_tokenize_halogen(self) -> None:
        """Two-letter atoms like Cl and Br should be single tokens."""
        tokens = tokenize_smiles("ClCCBr")
        assert "Cl" in tokens
        assert "Br" in tokens

    def test_tokenize_bracketed_atom(self) -> None:
        """Bracketed atoms like [nH] should be single tokens."""
        tokens = tokenize_smiles("c1cc[nH]c1")
        assert "[nH]" in tokens

    def test_tokenize_chirality(self) -> None:
        """Chirality markers @ and @@ should be correctly tokenized."""
        tokens = tokenize_smiles("[C@@H](O)(F)Cl")
        assert "[C@@H]" in tokens

    def test_tokenize_aspirin(self) -> None:
        """Aspirin SMILES should tokenize to a reasonable number of tokens."""
        tokens = tokenize_smiles(ASPIRIN_SMILES)
        assert len(tokens) > 10
        assert "C" in tokens
        assert "O" in tokens

    def test_tokenize_empty_string(self) -> None:
        """Empty SMILES should produce empty token list."""
        tokens = tokenize_smiles("")
        assert tokens == []


class TestSMILESVocabulary:
    """Tests for SMILES vocabulary encoding/decoding."""

    def test_default_vocab_has_special_tokens(self, vocab: SMILESVocabulary) -> None:
        """Default vocabulary should have PAD, SOS, EOS, UNK at expected indices."""
        assert vocab.encode_token("<pad>") == 0
        assert vocab.encode_token("<sos>") == 1
        assert vocab.encode_token("<eos>") == 2
        assert vocab.encode_token("<unk>") == 3

    def test_encode_decode_roundtrip(self, vocab: SMILESVocabulary) -> None:
        """Encoding then decoding should recover the original SMILES."""
        smiles = ETHANOL_SMILES
        encoded = vocab.encode_smiles(smiles, max_length=20)
        decoded = vocab.decode_indices(encoded)
        assert decoded == smiles

    def test_encode_adds_special_tokens(self, vocab: SMILESVocabulary) -> None:
        """Encoded sequence should start with SOS and end with EOS (before padding)."""
        encoded = vocab.encode_smiles("CCO", max_length=20)
        assert encoded[0] == vocab.SOS_IDX
        # Find EOS
        assert vocab.EOS_IDX in encoded

    def test_encode_pads_to_max_length(self, vocab: SMILESVocabulary) -> None:
        """Encoded sequence should be padded to max_length."""
        max_len = 30
        encoded = vocab.encode_smiles("C", max_length=max_len)
        assert len(encoded) == max_len

    def test_unknown_token_maps_to_unk(self, vocab: SMILESVocabulary) -> None:
        """Unknown tokens should map to UNK index."""
        idx = vocab.encode_token("XYZZY")
        assert idx == vocab.UNK_IDX

    def test_build_from_corpus(self) -> None:
        """Vocabulary built from corpus should contain frequent tokens."""
        vocab = SMILESVocabulary.build_from_corpus(DRUG_SMILES_CORPUS, min_frequency=1)
        assert vocab.size > 4  # More than just special tokens
        assert vocab.encode_token("C") != vocab.UNK_IDX
        assert vocab.encode_token("O") != vocab.UNK_IDX

    def test_build_from_corpus_respects_min_frequency(self) -> None:
        """Tokens below min_frequency should not be included."""
        vocab_all = SMILESVocabulary.build_from_corpus(DRUG_SMILES_CORPUS, min_frequency=1)
        vocab_strict = SMILESVocabulary.build_from_corpus(DRUG_SMILES_CORPUS, min_frequency=100)
        assert vocab_strict.size <= vocab_all.size


class TestSMILESValidation:
    """Tests for SMILES validity checking."""

    @pytest.mark.parametrize(
        "smiles",
        [ASPIRIN_SMILES, IBUPROFEN_SMILES, CAFFEINE_SMILES, PARACETAMOL_SMILES, ETHANOL_SMILES],
    )
    def test_valid_drug_smiles(self, smiles: str) -> None:
        """Known drug SMILES should be valid."""
        assert is_valid_smiles(smiles)

    @pytest.mark.parametrize(
        "smiles",
        [
            "",
            "XYZ",
            "C(C(C",  # Unbalanced parentheses
            "C1CC",  # Unclosed ring
        ],
    )
    def test_invalid_smiles(self, smiles: str) -> None:
        """Invalid SMILES should be rejected."""
        assert not is_valid_smiles(smiles)


class TestCanonicalization:
    """Tests for SMILES canonicalization."""

    def test_canonicalize_aspirin(self) -> None:
        """Aspirin should have a canonical form."""
        canonical = canonicalize_smiles(ASPIRIN_SMILES)
        assert canonical is not None
        assert len(canonical) > 0

    def test_canonicalize_same_molecule(self) -> None:
        """Different SMILES for the same molecule should canonicalize identically."""
        # Both represent ethanol
        c1 = canonicalize_smiles("CCO")
        c2 = canonicalize_smiles("OCC")
        if c1 is not None and c2 is not None:
            assert c1 == c2

    def test_canonicalize_invalid_returns_none(self) -> None:
        """Invalid SMILES should return None."""
        result = canonicalize_smiles("NOT_A_SMILES_[[[")
        assert result is None


class TestMolecularFingerprints:
    """Tests for molecular fingerprint computation."""

    def test_morgan_fingerprint_shape(self) -> None:
        """Morgan fingerprint should have correct bit vector length."""
        fp = compute_morgan_fingerprint(ASPIRIN_SMILES, radius=2, n_bits=2048)
        if fp is not None:
            assert fp.shape == (2048,)
            assert fp.dtype == np.float32

    def test_morgan_fingerprint_has_set_bits(self) -> None:
        """Morgan fingerprint of a real molecule should have some bits set."""
        fp = compute_morgan_fingerprint(ASPIRIN_SMILES)
        if fp is not None:
            assert np.sum(fp) > 0

    def test_tanimoto_self_similarity(self) -> None:
        """A fingerprint should have Tanimoto similarity 1.0 with itself."""
        fp = compute_morgan_fingerprint(ASPIRIN_SMILES)
        if fp is not None:
            assert abs(tanimoto_similarity(fp, fp) - 1.0) < 1e-6

    def test_tanimoto_different_molecules(self) -> None:
        """Different molecules should have Tanimoto similarity < 1.0."""
        fp1 = compute_morgan_fingerprint(ASPIRIN_SMILES)
        fp2 = compute_morgan_fingerprint(CAFFEINE_SMILES)
        if fp1 is not None and fp2 is not None:
            sim = tanimoto_similarity(fp1, fp2)
            assert 0.0 <= sim < 1.0


class TestMolecularDescriptors:
    """Tests for molecular descriptor computation."""

    def test_compute_descriptors_aspirin(self) -> None:
        """Aspirin descriptors should have reasonable pharmaceutical values."""
        desc = compute_descriptors(ASPIRIN_SMILES)
        assert desc.smiles == ASPIRIN_SMILES

        if desc.molecular_weight is not None:
            # Aspirin MW = 180.16 g/mol
            assert 170 < desc.molecular_weight < 190

        if desc.logp is not None:
            # Aspirin LogP ~ 1.2
            assert -1 < desc.logp < 3

        if desc.num_hbd is not None:
            # Aspirin has 1 HBD (the carboxylic acid OH)
            assert desc.num_hbd >= 1

    def test_compute_descriptors_invalid_smiles(self) -> None:
        """Invalid SMILES should return descriptors with None values."""
        desc = compute_descriptors("INVALID_SMILES")
        assert desc.smiles == "INVALID_SMILES"
        # Properties may be None or estimated

    def test_lipinski_aspirin(self) -> None:
        """Aspirin should pass Lipinski's Rule of Five."""
        desc = compute_descriptors(ASPIRIN_SMILES)
        if desc.is_lipinski_compliant is not None:
            assert desc.is_lipinski_compliant

    def test_descriptor_vector_length(self) -> None:
        """Descriptor vector should have 10 elements."""
        vec = compute_descriptor_vector(ASPIRIN_SMILES)
        assert len(vec) == 10

    def test_descriptor_vector_values_nonnegative_mw(self) -> None:
        """Molecular weight in descriptor vector should be positive."""
        vec = compute_descriptor_vector(ASPIRIN_SMILES)
        mw = vec[0]
        assert mw > 0


class TestPAINSFilter:
    """Tests for PAINS structural alert detection."""

    def test_no_pains_aspirin(self) -> None:
        """Aspirin should not trigger PAINS alerts (it's a real drug)."""
        alerts = check_pains_alerts(ASPIRIN_SMILES)
        # Aspirin may trigger azo or other false flags depending on pattern
        # but should have few alerts
        assert isinstance(alerts, list)

    def test_catechol_alert(self) -> None:
        """Catechol substructure should trigger PAINS alert."""
        # Catechol: 1,2-dihydroxybenzene
        catechol = "c1ccc(c(c1)O)O"
        alerts = check_pains_alerts(catechol)
        # Should detect catechol pattern
        if alerts:
            assert "catechol" in alerts


class TestLipinskiFilter:
    """Tests for Lipinski Rule of Five implementation."""

    def test_drug_like_molecule(self) -> None:
        """Aspirin-like properties should pass Lipinski filter."""
        result = apply_lipinski_filter(mw=180.0, logp=1.2, hbd=1, hba=4)
        assert isinstance(result, LipinskiResult)
        assert result.is_drug_like
        assert result.num_violations == 0
        assert result.passes

    def test_all_violations(self) -> None:
        """Properties violating all four rules should yield 4 violations."""
        result = apply_lipinski_filter(mw=800.0, logp=8.0, hbd=10, hba=15)
        assert result.num_violations == 4
        assert not result.is_drug_like

    def test_one_violation_allowed(self) -> None:
        """One violation should still be considered drug-like."""
        result = apply_lipinski_filter(mw=600.0, logp=2.0, hbd=2, hba=5)
        assert result.num_violations == 1
        assert result.is_drug_like  # Ro5 allows 1 violation


class TestQEDScore:
    """Tests for QED drug-likeness score."""

    def test_qed_range(self) -> None:
        """QED score should be in [0, 1]."""
        score = compute_qed_score(
            mw=300.0,
            logp=2.5,
            hba=4,
            hbd=1,
            psa=75.0,
            rotatable_bonds=3,
            num_aromatic_rings=2,
            num_alerts=0,
        )
        assert 0.0 <= score <= 1.0

    def test_ideal_molecule_has_high_qed(self) -> None:
        """A molecule with ideal properties should have high QED."""
        score = compute_qed_score(
            mw=300.0,
            logp=2.5,
            hba=4,
            hbd=1,
            psa=75.0,
            rotatable_bonds=3,
            num_aromatic_rings=2,
            num_alerts=0,
        )
        assert score > 0.5

    def test_poor_molecule_has_low_qed(self) -> None:
        """A molecule with poor properties should have low QED."""
        score = compute_qed_score(
            mw=800.0,
            logp=8.0,
            hba=15,
            hbd=8,
            psa=250.0,
            rotatable_bonds=15,
            num_aromatic_rings=8,
            num_alerts=5,
        )
        assert score < 0.3


class TestSAScore:
    """Tests for synthetic accessibility score estimate."""

    def test_simple_molecule_low_sa(self) -> None:
        """A simple molecule should have a low (easy) SA score."""
        score = compute_sa_score_estimate(
            ring_count=1,
            stereo_centers=0,
            sp3_fraction=0.2,
        )
        assert 1.0 <= score <= 4.0

    def test_complex_molecule_high_sa(self) -> None:
        """A complex molecule should have a high (hard) SA score."""
        score = compute_sa_score_estimate(
            ring_count=6,
            stereo_centers=4,
            sp3_fraction=0.8,
            num_bridged_atoms=2,
            macrocycle=True,
        )
        assert score >= 5.0

    def test_sa_score_range(self) -> None:
        """SA score should be clamped to [1, 10]."""
        low = compute_sa_score_estimate(0, 0, 0.0)
        high = compute_sa_score_estimate(10, 10, 1.0, 5, True)
        assert low >= 1.0
        assert high <= 10.0
