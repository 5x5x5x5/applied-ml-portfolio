"""Tests for sequence alignment (Needleman-Wunsch)."""

from __future__ import annotations

import pytest

from protein_explorer.analysis.alignment import (
    BLOSUM62,
    blosum62_score,
    format_alignment,
    needleman_wunsch,
    optimize_gap_penalties,
)


class TestBlosum62:
    """Tests for BLOSUM62 scoring matrix."""

    def test_diagonal_positive(self) -> None:
        """Identical amino acid pairs should have positive scores."""
        for aa in "ARNDCQEGHILKMFPSTWYV":
            assert BLOSUM62[(aa, aa)] > 0, f"{aa}-{aa} should be positive"

    def test_symmetric(self) -> None:
        """Matrix should be symmetric: score(A,B) == score(B,A)."""
        for aa1 in "ARNDCQEGHILKMFPSTWYV":
            for aa2 in "ARNDCQEGHILKMFPSTWYV":
                assert BLOSUM62[(aa1, aa2)] == BLOSUM62[(aa2, aa1)]

    def test_known_scores(self) -> None:
        """Verify specific well-known BLOSUM62 scores."""
        # Identical residues
        assert BLOSUM62[("W", "W")] == 11  # Trp self-score is highest
        assert BLOSUM62[("C", "C")] == 9  # Cys self-score
        assert BLOSUM62[("A", "A")] == 4  # Ala self-score

        # Similar residues (positive scores)
        assert BLOSUM62[("D", "E")] == 2  # Acidic pair
        assert BLOSUM62[("K", "R")] == 2  # Basic pair
        assert BLOSUM62[("I", "V")] == 3  # Hydrophobic pair
        assert BLOSUM62[("F", "Y")] == 3  # Aromatic pair

        # Dissimilar residues (negative scores)
        assert BLOSUM62[("D", "W")] < 0  # Acidic vs aromatic
        assert BLOSUM62[("K", "D")] < 0  # Basic vs acidic

    def test_matrix_completeness(self) -> None:
        """All 400 amino acid pairs should be present."""
        aas = "ARNDCQEGHILKMFPSTWYV"
        for aa1 in aas:
            for aa2 in aas:
                assert (aa1, aa2) in BLOSUM62

    def test_blosum62_score_function(self) -> None:
        """Helper function should match dictionary lookup."""
        assert blosum62_score("A", "A") == BLOSUM62[("A", "A")]
        assert blosum62_score("W", "W") == 11


class TestNeedlemanWunsch:
    """Tests for the Needleman-Wunsch alignment algorithm."""

    def test_identical_sequences(self) -> None:
        """Identical sequences should have perfect alignment."""
        result = needleman_wunsch("ACDEF", "ACDEF")
        assert result.aligned_seq1 == "ACDEF"
        assert result.aligned_seq2 == "ACDEF"
        assert result.identity == 1.0
        assert result.gaps == 0

    def test_single_substitution(self) -> None:
        """One mismatch in otherwise identical sequences."""
        result = needleman_wunsch("ACDEF", "ACDKF")
        assert result.identity < 1.0
        assert result.alignment_length == 5
        # E->K substitution, no gaps
        assert result.gaps == 0

    def test_simple_gap(self) -> None:
        """Should introduce gap for insertion."""
        result = needleman_wunsch("ACDEF", "ACDXEF")
        assert "-" in result.aligned_seq1 or "-" in result.aligned_seq2
        assert result.gaps >= 1

    def test_alignment_score_positive(self) -> None:
        """Similar sequences should have positive score."""
        result = needleman_wunsch("MVLSPADKTNVK", "MVHLTPEEKSAV")
        assert result.score > 0

    def test_hemoglobin_alignment(self, hemoglobin_alpha: str, hemoglobin_beta: str) -> None:
        """Hemoglobin alpha vs beta should show ~43% identity."""
        result = needleman_wunsch(hemoglobin_alpha, hemoglobin_beta)

        # Known: human Hb alpha and beta have ~43% identity
        assert 0.30 < result.identity < 0.60
        assert result.similarity > result.identity
        assert result.score > 0
        assert result.alignment_length > 0

    def test_alignment_lengths_consistent(self) -> None:
        """Aligned sequences and midline should have same length."""
        result = needleman_wunsch("ACDEFGHIK", "ACDGHIKLM")
        assert len(result.aligned_seq1) == len(result.aligned_seq2)
        assert len(result.aligned_seq1) == len(result.midline)
        assert len(result.aligned_seq1) == result.alignment_length

    def test_midline_symbols(self) -> None:
        """Midline should use | for identity, + for similarity, space for mismatch."""
        result = needleman_wunsch("ACDEF", "ACDEF")
        assert all(c == "|" for c in result.midline)

    def test_gap_penalty_effect(self) -> None:
        """More severe gap penalties should reduce gaps."""
        seq1 = "ACDEFGHIKLMNPQR"
        seq2 = "ACDEGHIKLMPQR"

        result_mild = needleman_wunsch(seq1, seq2, gap_open=-5.0, gap_extend=-0.1)
        result_severe = needleman_wunsch(seq1, seq2, gap_open=-20.0, gap_extend=-5.0)

        # Severe penalties should discourage gaps
        assert (
            result_severe.gaps <= result_mild.gaps
            or result_severe.gap_opens <= result_mild.gap_opens
        )

    def test_affine_gap_penalty(self) -> None:
        """Affine gaps should prefer one long gap over multiple short gaps."""
        seq1 = "ACDEFGHIKLMNPQRSTVWY"
        seq2 = "ACDPQRSTVWY"

        # With proper affine gap, should get one contiguous gap
        result = needleman_wunsch(seq1, seq2, gap_open=-10.0, gap_extend=-0.5)
        assert result.gap_opens <= 2  # Should be 1 or 2 gap events

    def test_empty_after_validation_error(self) -> None:
        """Invalid sequences should raise ValueError."""
        with pytest.raises(ValueError):
            needleman_wunsch("ACDE123", "ACDEF")

    def test_single_residue_alignment(self) -> None:
        """Alignment of single residues should work."""
        result = needleman_wunsch("A", "A")
        assert result.identity == 1.0
        assert result.score == BLOSUM62[("A", "A")]

    def test_very_different_sequences(self) -> None:
        """Very different sequences should have low identity."""
        result = needleman_wunsch("WWWWWWWWWW", "DDDDDDDDDD")
        assert result.identity == 0.0


class TestAlignmentFormatting:
    """Tests for alignment output formatting."""

    def test_format_contains_score(self) -> None:
        """Formatted output should contain score."""
        result = needleman_wunsch("ACDEF", "ACDEF")
        formatted = format_alignment(result)
        assert "Score:" in formatted

    def test_format_contains_stats(self) -> None:
        """Formatted output should contain identity and similarity."""
        result = needleman_wunsch("ACDEFGHIK", "ACDKFGHIK")
        formatted = format_alignment(result)
        assert "Identity:" in formatted
        assert "Similarity:" in formatted
        assert "Gaps:" in formatted

    def test_format_line_width(self) -> None:
        """Should respect line width parameter."""
        result = needleman_wunsch(
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
            "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLST",
        )
        formatted = format_alignment(result, line_width=30)
        # Lines should not exceed 30 characters for alignment portion
        for line in formatted.split("\n"):
            if line.startswith("Seq"):
                # Extract the alignment part (after the position prefix)
                parts = line.split()
                if len(parts) >= 3:
                    alignment_part = parts[2]
                    assert len(alignment_part) <= 30


class TestGapOptimization:
    """Tests for gap penalty optimization."""

    def test_returns_valid_penalties(self) -> None:
        """Should return gap penalties within specified range."""
        go, ge, identity = optimize_gap_penalties(
            "ACDEFGHIK",
            "ACDGHIK",
            gap_open_range=(-15.0, -5.0),
            gap_extend_range=(-2.0, -0.1),
            steps=3,
        )
        assert -15.0 <= go <= -5.0
        assert -2.0 <= ge <= -0.1
        assert 0.0 <= identity <= 1.0

    def test_optimization_improves_identity(
        self, hemoglobin_alpha: str, hemoglobin_beta: str
    ) -> None:
        """Optimized penalties should yield reasonable identity."""
        go, ge, identity = optimize_gap_penalties(
            hemoglobin_alpha,
            hemoglobin_beta,
            steps=3,
        )
        # The optimized alignment should have positive identity
        assert identity > 0.2
