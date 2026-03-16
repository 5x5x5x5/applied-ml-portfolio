"""Sequence alignment using Needleman-Wunsch algorithm.

Implements pairwise global alignment with BLOSUM62 scoring matrix,
affine gap penalties, and alignment statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

from protein_explorer.analysis.sequence_analyzer import validate_sequence

logger = logging.getLogger(__name__)

# BLOSUM62 substitution matrix
# Standard matrix from Henikoff & Henikoff (1992) PNAS 89:10915-10919
# Order: A R N D C Q E G H I L K M F P S T W Y V
_BLOSUM62_AA_ORDER = "ARNDCQEGHILKMFPSTWYV"

_BLOSUM62_DATA: list[list[int]] = [
    # A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
    [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],  # A
    [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],  # R
    [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],  # N
    [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],  # D
    [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],  # C
    [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],  # Q
    [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],  # E
    [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],  # G
    [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],  # H
    [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],  # I
    [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],  # L
    [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],  # K
    [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],  # M
    [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],  # F
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],  # P
    [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],  # S
    [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],  # T
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],  # W
    [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],  # Y
    [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],  # V
]

# Build lookup dictionary for fast access
BLOSUM62: dict[tuple[str, str], int] = {}
for i, aa_i in enumerate(_BLOSUM62_AA_ORDER):
    for j, aa_j in enumerate(_BLOSUM62_AA_ORDER):
        BLOSUM62[(aa_i, aa_j)] = _BLOSUM62_DATA[i][j]


class TracebackDirection(Enum):
    """Direction in the traceback matrix."""

    DIAGONAL = 0
    UP = 1
    LEFT = 2
    NONE = 3


@dataclass
class AlignmentResult:
    """Result from pairwise sequence alignment."""

    sequence1: str
    sequence2: str
    aligned_seq1: str
    aligned_seq2: str
    score: float
    identity: float  # Fraction of identical positions
    similarity: float  # Fraction of similar positions (positive BLOSUM62 score)
    gaps: int
    gap_opens: int
    alignment_length: int
    midline: str  # | for identity, + for similarity, space for mismatch


def blosum62_score(aa1: str, aa2: str) -> int:
    """Get BLOSUM62 score for a pair of amino acids.

    Args:
        aa1: First amino acid (single letter).
        aa2: Second amino acid (single letter).

    Returns:
        BLOSUM62 substitution score.
    """
    return BLOSUM62.get((aa1, aa2), -4)


def needleman_wunsch(
    seq1: str,
    seq2: str,
    gap_open: float = -10.0,
    gap_extend: float = -0.5,
    matrix: dict[tuple[str, str], int] | None = None,
) -> AlignmentResult:
    """Perform global pairwise alignment using Needleman-Wunsch with affine gaps.

    Implements the Gotoh modification for affine gap penalties using three
    matrices: M (match/mismatch), X (gap in seq2), Y (gap in seq1).

    Args:
        seq1: First protein sequence.
        seq2: Second protein sequence.
        gap_open: Gap opening penalty (negative value).
        gap_extend: Gap extension penalty (negative value).
        matrix: Substitution matrix (defaults to BLOSUM62).

    Returns:
        AlignmentResult with aligned sequences and statistics.
    """
    s1 = validate_sequence(seq1)
    s2 = validate_sequence(seq2)

    if matrix is None:
        matrix = BLOSUM62

    m = len(s1)
    n = len(s2)

    # Three score matrices for affine gap penalties
    # M[i][j] = best score ending with s1[i-1] aligned to s2[j-1]
    # X[i][j] = best score ending with gap in s2 (deletion from s1)
    # Y[i][j] = best score ending with gap in s1 (insertion from s2)
    NEG_INF = float("-inf")

    M = np.full((m + 1, n + 1), NEG_INF, dtype=np.float64)
    X = np.full((m + 1, n + 1), NEG_INF, dtype=np.float64)
    Y = np.full((m + 1, n + 1), NEG_INF, dtype=np.float64)

    # Traceback matrices
    trace_M = np.full((m + 1, n + 1), TracebackDirection.NONE.value, dtype=np.int8)
    trace_X = np.full((m + 1, n + 1), TracebackDirection.NONE.value, dtype=np.int8)
    trace_Y = np.full((m + 1, n + 1), TracebackDirection.NONE.value, dtype=np.int8)

    # Initialize
    M[0, 0] = 0.0
    for i in range(1, m + 1):
        X[i, 0] = gap_open + (i - 1) * gap_extend
        M[i, 0] = X[i, 0]
    for j in range(1, n + 1):
        Y[0, j] = gap_open + (j - 1) * gap_extend
        M[0, j] = Y[0, j]

    # Fill matrices
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Score for aligning s1[i-1] with s2[j-1]
            sub_score = matrix.get((s1[i - 1], s2[j - 1]), -4)

            # M[i][j]: best score ending with match/mismatch
            m_scores = [
                M[i - 1, j - 1] + sub_score,
                X[i - 1, j - 1] + sub_score,
                Y[i - 1, j - 1] + sub_score,
            ]
            best_m = max(range(3), key=lambda k: m_scores[k])
            M[i, j] = m_scores[best_m]
            trace_M[i, j] = best_m  # 0=from M, 1=from X, 2=from Y

            # X[i][j]: gap in seq2 (consuming s1[i-1])
            x_scores = [
                M[i - 1, j] + gap_open,  # Open new gap
                X[i - 1, j] + gap_extend,  # Extend existing gap
            ]
            best_x = max(range(2), key=lambda k: x_scores[k])
            X[i, j] = x_scores[best_x]
            trace_X[i, j] = best_x  # 0=from M (open), 1=from X (extend)

            # Y[i][j]: gap in seq1 (consuming s2[j-1])
            y_scores = [
                M[i, j - 1] + gap_open,  # Open new gap
                Y[i, j - 1] + gap_extend,  # Extend existing gap
            ]
            best_y = max(range(2), key=lambda k: y_scores[k])
            Y[i, j] = y_scores[best_y]
            trace_Y[i, j] = best_y  # 0=from M (open), 1=from Y (extend)

    # Find best ending score
    final_scores = [M[m, n], X[m, n], Y[m, n]]
    best_final = max(range(3), key=lambda k: final_scores[k])
    best_score = final_scores[best_final]

    # Traceback
    aligned1: list[str] = []
    aligned2: list[str] = []

    i, j = m, n
    current_matrix = best_final  # 0=M, 1=X, 2=Y

    while i > 0 or j > 0:
        if current_matrix == 0:  # In M matrix (match/mismatch)
            if i == 0:
                aligned1.append("-")
                aligned2.append(s2[j - 1])
                j -= 1
                continue
            if j == 0:
                aligned1.append(s1[i - 1])
                aligned2.append("-")
                i -= 1
                continue

            prev = int(trace_M[i, j])
            aligned1.append(s1[i - 1])
            aligned2.append(s2[j - 1])
            i -= 1
            j -= 1
            current_matrix = prev

        elif current_matrix == 1:  # In X matrix (gap in seq2)
            aligned1.append(s1[i - 1])
            aligned2.append("-")
            prev = int(trace_X[i, j])
            i -= 1
            current_matrix = 0 if prev == 0 else 1

        else:  # In Y matrix (gap in seq1)
            aligned1.append("-")
            aligned2.append(s2[j - 1])
            prev = int(trace_Y[i, j])
            j -= 1
            current_matrix = 0 if prev == 0 else 2

    # Reverse since we traced back from end
    aligned1.reverse()
    aligned2.reverse()

    aln1 = "".join(aligned1)
    aln2 = "".join(aligned2)

    # Compute statistics
    stats = _alignment_statistics(aln1, aln2, matrix)

    return AlignmentResult(
        sequence1=s1,
        sequence2=s2,
        aligned_seq1=aln1,
        aligned_seq2=aln2,
        score=best_score,
        identity=stats["identity"],
        similarity=stats["similarity"],
        gaps=stats["gaps"],
        gap_opens=stats["gap_opens"],
        alignment_length=len(aln1),
        midline=stats["midline"],
    )


def _alignment_statistics(
    aln1: str, aln2: str, matrix: dict[tuple[str, str], int]
) -> dict[str, float | int | str]:
    """Compute alignment statistics.

    Args:
        aln1: First aligned sequence (with gaps).
        aln2: Second aligned sequence (with gaps).
        matrix: Substitution matrix used.

    Returns:
        Dictionary with identity, similarity, gaps, gap_opens, midline.
    """
    length = len(aln1)
    identities = 0
    similarities = 0
    gaps = 0
    gap_opens = 0
    in_gap_1 = False
    in_gap_2 = False
    midline_chars: list[str] = []

    for k in range(length):
        a = aln1[k]
        b = aln2[k]

        if a == "-" or b == "-":
            gaps += 1
            midline_chars.append(" ")
            if a == "-":
                if not in_gap_1:
                    gap_opens += 1
                    in_gap_1 = True
                in_gap_2 = False
            else:
                if not in_gap_2:
                    gap_opens += 1
                    in_gap_2 = True
                in_gap_1 = False
        else:
            in_gap_1 = False
            in_gap_2 = False

            if a == b:
                identities += 1
                midline_chars.append("|")
            elif matrix.get((a, b), -4) > 0:
                similarities += 1
                midline_chars.append("+")
            else:
                midline_chars.append(" ")

    aligned_positions = length - gaps
    identity_frac = identities / aligned_positions if aligned_positions > 0 else 0.0
    similarity_frac = (
        (identities + similarities) / aligned_positions if aligned_positions > 0 else 0.0
    )

    return {
        "identity": round(identity_frac, 4),
        "similarity": round(similarity_frac, 4),
        "gaps": gaps,
        "gap_opens": gap_opens,
        "midline": "".join(midline_chars),
    }


def format_alignment(result: AlignmentResult, line_width: int = 60) -> str:
    """Format alignment result for display.

    Args:
        result: AlignmentResult to format.
        line_width: Characters per line.

    Returns:
        Formatted alignment string.
    """
    lines: list[str] = []
    lines.append(f"Score: {result.score:.1f}")
    lines.append(
        f"Identity: {result.identity:.1%} | "
        f"Similarity: {result.similarity:.1%} | "
        f"Gaps: {result.gaps} ({result.gap_opens} opens)"
    )
    lines.append(f"Length: {result.alignment_length}")
    lines.append("")

    pos1 = 0  # Position counter for seq1
    pos2 = 0  # Position counter for seq2

    for start in range(0, result.alignment_length, line_width):
        end = min(start + line_width, result.alignment_length)

        chunk1 = result.aligned_seq1[start:end]
        chunk2 = result.aligned_seq2[start:end]
        midline = result.midline[start:end]

        # Count non-gap positions
        p1_start = pos1 + 1
        p2_start = pos2 + 1
        pos1 += sum(1 for c in chunk1 if c != "-")
        pos2 += sum(1 for c in chunk2 if c != "-")

        lines.append(f"Seq1 {p1_start:>5d}  {chunk1}  {pos1}")
        lines.append(f"             {midline}")
        lines.append(f"Seq2 {p2_start:>5d}  {chunk2}  {pos2}")
        lines.append("")

    return "\n".join(lines)


def optimize_gap_penalties(
    seq1: str,
    seq2: str,
    gap_open_range: tuple[float, float] = (-15.0, -5.0),
    gap_extend_range: tuple[float, float] = (-2.0, -0.1),
    steps: int = 5,
) -> tuple[float, float, float]:
    """Find gap penalties that maximize alignment score density.

    Tests a grid of gap_open and gap_extend values and returns
    the combination that yields the highest identity.

    Args:
        seq1: First sequence.
        seq2: Second sequence.
        gap_open_range: (min, max) for gap opening penalty.
        gap_extend_range: (min, max) for gap extension penalty.
        steps: Number of steps in the grid for each parameter.

    Returns:
        Tuple of (best_gap_open, best_gap_extend, best_identity).
    """
    s1 = validate_sequence(seq1)
    s2 = validate_sequence(seq2)

    best_identity = -1.0
    best_open = gap_open_range[0]
    best_extend = gap_extend_range[0]

    open_values = np.linspace(gap_open_range[0], gap_open_range[1], steps)
    extend_values = np.linspace(gap_extend_range[0], gap_extend_range[1], steps)

    for go in open_values:
        for ge in extend_values:
            result = needleman_wunsch(s1, s2, gap_open=float(go), gap_extend=float(ge))
            if result.identity > best_identity:
                best_identity = result.identity
                best_open = float(go)
                best_extend = float(ge)

    return round(best_open, 2), round(best_extend, 2), round(best_identity, 4)
