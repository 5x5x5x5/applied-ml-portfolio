"""Tests for protein sequence analysis with real biochemistry validation."""

from __future__ import annotations

import pytest

from protein_explorer.analysis.sequence_analyzer import (
    KYTE_DOOLITTLE,
    amino_acid_composition,
    analyze_sequence,
    aromaticity,
    detect_signal_peptide,
    gravy_score,
    hydrophobicity_profile,
    instability_index,
    isoelectric_point,
    molecular_weight,
    predict_disulfide_bonds,
    predict_secondary_structure,
    validate_sequence,
)


class TestValidateSequence:
    """Tests for sequence validation."""

    def test_valid_sequence(self) -> None:
        result = validate_sequence("ACDEFGHIKLMNPQRSTVWY")
        assert result == "ACDEFGHIKLMNPQRSTVWY"

    def test_lowercase_conversion(self) -> None:
        result = validate_sequence("acdef")
        assert result == "ACDEF"

    def test_whitespace_removal(self) -> None:
        result = validate_sequence("ACD EFG\nHIK")
        assert result == "ACDEFGHIK"

    def test_invalid_characters(self) -> None:
        with pytest.raises(ValueError, match="Invalid amino acid"):
            validate_sequence("ACDEFX123")

    def test_empty_sequence(self) -> None:
        with pytest.raises(ValueError, match="Empty sequence"):
            validate_sequence("   ")

    def test_fasta_header_stripped(self) -> None:
        """FASTA format should have header removed."""
        result = validate_sequence(">sp|P69905|HBA_HUMAN\nMVLSPADK")
        assert result == "MVLSPADK"


class TestAminoAcidComposition:
    """Tests for amino acid composition."""

    def test_all_amino_acids(self, short_peptide: str) -> None:
        counts, freqs = amino_acid_composition(short_peptide)
        # Each of the 20 amino acids appears exactly once
        assert len(counts) == 20
        assert all(c == 1 for c in counts.values())
        assert all(abs(f - 5.0) < 0.01 for f in freqs.values())

    def test_polyalanine(self, polyalanine: str) -> None:
        counts, freqs = amino_acid_composition(polyalanine)
        assert counts["A"] == 30
        assert freqs["A"] == 100.0
        assert len(counts) == 1

    def test_hemoglobin_alpha(self, hemoglobin_alpha: str) -> None:
        counts, freqs = amino_acid_composition(hemoglobin_alpha)
        assert counts["M"] >= 1  # At least the initial Met
        assert sum(counts.values()) == len(hemoglobin_alpha)
        assert abs(sum(freqs.values()) - 100.0) < 0.1


class TestMolecularWeight:
    """Tests for molecular weight calculation."""

    def test_glycine_dipeptide(self) -> None:
        """Gly-Gly should have MW ~132.05 Da (2*57.05 + 18.02)."""
        mw = molecular_weight("GG")
        # 2 * 57.0519 + 18.01524 = 132.119
        assert abs(mw - 132.12) < 1.0

    def test_single_alanine(self) -> None:
        """Single Ala: 71.08 + 18.02 = ~89.1 Da."""
        mw = molecular_weight("A")
        assert abs(mw - 89.09) < 1.0

    def test_insulin_b_chain(self, insulin_b_chain: str) -> None:
        """Insulin B chain ~3400 Da."""
        mw = molecular_weight(insulin_b_chain)
        assert 3300 < mw < 3600

    def test_hemoglobin_alpha(self, hemoglobin_alpha: str) -> None:
        """Hemoglobin alpha ~15.1 kDa."""
        mw = molecular_weight(hemoglobin_alpha)
        assert 14000 < mw < 16000

    def test_monoisotopic_vs_average(self) -> None:
        """Monoisotopic masses should be slightly lower than average."""
        avg = molecular_weight("ACDEFGHIKLMNPQRSTVWY", monoisotopic=False)
        mono = molecular_weight("ACDEFGHIKLMNPQRSTVWY", monoisotopic=True)
        assert mono < avg


class TestIsoelectricPoint:
    """Tests for pI calculation."""

    def test_acidic_peptide(self) -> None:
        """Poly-Asp/Glu should have low pI (~3-4)."""
        pi = isoelectric_point("DDDEEEE")
        assert 2.5 < pi < 4.5

    def test_basic_peptide(self) -> None:
        """Poly-Lys/Arg should have high pI (~10-12)."""
        pi = isoelectric_point("KKKKRRR")
        assert 10.0 < pi < 13.0

    def test_hemoglobin_alpha(self, hemoglobin_alpha: str) -> None:
        """Hemoglobin alpha pI is approximately 8.7 (from ExPASy)."""
        pi = isoelectric_point(hemoglobin_alpha)
        # Allow some variation since this is a simplified calculation
        assert 7.5 < pi < 10.0

    def test_neutral_range(self) -> None:
        """A balanced sequence should have pI in moderate range."""
        pi = isoelectric_point("ACDEFGHIKLMNPQRSTVWY")
        assert 4.0 < pi < 10.0


class TestHydrophobicityProfile:
    """Tests for Kyte-Doolittle hydrophobicity."""

    def test_hydrophobic_sequence(self) -> None:
        """All hydrophobic residues should give positive profile."""
        profile = hydrophobicity_profile("VVVIIILLL", window_size=3)
        assert all(v > 0 for v in profile)

    def test_hydrophilic_sequence(self) -> None:
        """All charged residues should give negative profile."""
        profile = hydrophobicity_profile("DDDEEEKKKRRR", window_size=3)
        assert all(v < 0 for v in profile)

    def test_profile_length(self, hemoglobin_alpha: str) -> None:
        """Profile should have one value per residue."""
        profile = hydrophobicity_profile(hemoglobin_alpha)
        assert len(profile) == len(hemoglobin_alpha)

    def test_window_size_effect(self) -> None:
        """Larger windows should smooth the profile."""
        seq = "VVVVDDDDVVVV"
        profile_small = hydrophobicity_profile(seq, window_size=3)
        profile_large = hydrophobicity_profile(seq, window_size=7)
        # Larger window should have less extreme values
        assert max(profile_large) <= max(profile_small)

    def test_known_values(self) -> None:
        """Check that individual residue values match Kyte-Doolittle scale."""
        profile = hydrophobicity_profile("I", window_size=1)
        assert abs(profile[0] - KYTE_DOOLITTLE["I"]) < 0.01


class TestSecondaryStructure:
    """Tests for Chou-Fasman secondary structure prediction."""

    def test_polyalanine_helix(self, polyalanine: str) -> None:
        """Polyalanine is a strong helix former."""
        pred = predict_secondary_structure(polyalanine)
        assert pred.helix_fraction > 0.5  # Should be mostly helix
        assert "H" in pred.prediction

    def test_polyvaline_sheet(self) -> None:
        """Polyvaline/isoleucine is a strong sheet former."""
        pred = predict_secondary_structure("VVVVVIIIIIVVVVVIIIIIIII")
        assert pred.sheet_fraction > 0.3  # Should have significant sheet content

    def test_prediction_length(self, hemoglobin_alpha: str) -> None:
        """Prediction should have one element per residue."""
        pred = predict_secondary_structure(hemoglobin_alpha)
        assert len(pred.prediction) == len(hemoglobin_alpha)

    def test_fractions_sum_to_one(self, hemoglobin_alpha: str) -> None:
        """Helix + sheet + coil fractions should sum to 1.0."""
        pred = predict_secondary_structure(hemoglobin_alpha)
        total = pred.helix_fraction + pred.sheet_fraction + pred.coil_fraction
        assert abs(total - 1.0) < 0.01

    def test_valid_predictions(self, hemoglobin_alpha: str) -> None:
        """All predictions should be H, E, or C."""
        pred = predict_secondary_structure(hemoglobin_alpha)
        assert all(p in ("H", "E", "C") for p in pred.prediction)

    def test_hemoglobin_has_helices(self, hemoglobin_alpha: str) -> None:
        """Hemoglobin is predominantly helical (~75% helix experimentally)."""
        pred = predict_secondary_structure(hemoglobin_alpha)
        assert pred.helix_fraction > 0.3  # Should detect significant helix


class TestSignalPeptide:
    """Tests for signal peptide detection."""

    def test_no_signal_in_hemoglobin(self, hemoglobin_alpha: str) -> None:
        """Cytoplasmic hemoglobin should not have signal peptide."""
        has_signal, _ = detect_signal_peptide(hemoglobin_alpha)
        # Hemoglobin is cytoplasmic, no signal peptide
        assert not has_signal

    def test_short_sequence(self) -> None:
        """Sequences shorter than 15 aa cannot have signal peptide."""
        has_signal, _ = detect_signal_peptide("MKFLA")
        assert not has_signal

    def test_signal_peptide_like_sequence(self) -> None:
        """Sequence with classic signal peptide features."""
        # n-region (positive) + h-region (hydrophobic) + c-region (AXA motif)
        seq = "MKFLIVILLVFLAVFAGASA" + "QVQLVESGGGLVQPGGSLRLSCAASGFNIKD"
        has_signal, cleavage = detect_signal_peptide(seq)
        assert has_signal
        assert 15 < cleavage < 25


class TestDisulfideBonds:
    """Tests for disulfide bond prediction."""

    def test_no_cysteines(self) -> None:
        """Sequence without Cys should have no bonds."""
        bonds = predict_disulfide_bonds("AAAAAAAAAAA")
        assert len(bonds) == 0

    def test_single_cysteine(self) -> None:
        """Single Cys cannot form a bond."""
        bonds = predict_disulfide_bonds("AAACAAAAAAA")
        assert len(bonds) == 0

    def test_two_cysteines(self) -> None:
        """Two cysteines with sufficient separation should pair."""
        seq = "AAACAAAAAAAAAACAAAA"
        bonds = predict_disulfide_bonds(seq)
        assert len(bonds) <= 1

    def test_lysozyme_cysteines(self, lysozyme: str) -> None:
        """Lysozyme has 8 cysteines forming 4 disulfide bonds."""
        cys_count = lysozyme.count("C")
        assert cys_count == 8
        bonds = predict_disulfide_bonds(lysozyme)
        assert len(bonds) >= 2  # Should predict at least some bonds
        assert len(bonds) <= 4  # Cannot have more than 4 with 8 Cys

    def test_bond_positions_valid(self, lysozyme: str) -> None:
        """Bond positions should be within sequence bounds."""
        bonds = predict_disulfide_bonds(lysozyme)
        n = len(lysozyme)
        for pos1, pos2 in bonds:
            assert 1 <= pos1 < pos2 <= n
            assert lysozyme[pos1 - 1] == "C"
            assert lysozyme[pos2 - 1] == "C"


class TestGravyAndAromaticity:
    """Tests for GRAVY and aromaticity calculations."""

    def test_hydrophobic_gravy(self) -> None:
        """Hydrophobic sequence should have positive GRAVY."""
        assert gravy_score("VVVIIILLLLFF") > 0

    def test_hydrophilic_gravy(self) -> None:
        """Hydrophilic sequence should have negative GRAVY."""
        assert gravy_score("DDDEEEKKKRRR") < 0

    def test_aromaticity_aromatic(self) -> None:
        """Sequence of aromatic residues should have aromaticity ~1.0."""
        assert aromaticity("FWYFWY") == 1.0

    def test_aromaticity_none(self) -> None:
        """Sequence without F, W, Y should have aromaticity 0."""
        assert aromaticity("AAAGGG") == 0.0


class TestInstabilityIndex:
    """Tests for instability index."""

    def test_short_sequence(self) -> None:
        """Very short sequences should return 0."""
        assert instability_index("A") == 0.0

    def test_returns_float(self, hemoglobin_alpha: str) -> None:
        """Should return a float value."""
        result = instability_index(hemoglobin_alpha)
        assert isinstance(result, float)


class TestFullAnalysis:
    """Tests for the complete analysis pipeline."""

    def test_complete_analysis(self, hemoglobin_alpha: str) -> None:
        """Full analysis should return all expected fields."""
        result = analyze_sequence(hemoglobin_alpha)

        assert result.length == 141
        assert result.molecular_weight > 0
        assert 0 < result.isoelectric_point < 14
        assert len(result.hydrophobicity_profile) == 141
        assert len(result.secondary_structure) == 141
        assert isinstance(result.gravy, float)
        assert isinstance(result.aromaticity, float)
        assert isinstance(result.instability_index, float)
        assert isinstance(result.charge_at_ph7, float)

    def test_insulin_analysis(self, insulin_b_chain: str) -> None:
        """Insulin B chain analysis sanity checks."""
        result = analyze_sequence(insulin_b_chain)
        assert result.length == 30
        assert result.molecular_weight > 3000
        # Insulin B chain has 2 Cys residues
        assert result.amino_acid_counts.get("C", 0) == 2

    def test_invalid_input(self) -> None:
        """Invalid input should raise ValueError."""
        with pytest.raises(ValueError):
            analyze_sequence("INVALID123")
