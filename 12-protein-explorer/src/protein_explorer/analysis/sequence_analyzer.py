"""Protein sequence analysis with accurate biochemistry.

Implements amino acid composition, molecular weight, isoelectric point,
hydrophobicity profiling, secondary structure prediction, signal peptide
detection, and disulfide bond prediction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Standard amino acid single-letter codes
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# Monoisotopic residue masses (Da) - mass of amino acid minus water
RESIDUE_MASSES: dict[str, float] = {
    "A": 71.03711,
    "R": 156.10111,
    "N": 114.04293,
    "D": 115.02694,
    "C": 103.00919,
    "E": 129.04259,
    "Q": 128.05858,
    "G": 57.02146,
    "H": 137.05891,
    "I": 113.08406,
    "L": 113.08406,
    "K": 128.09496,
    "M": 131.04049,
    "F": 147.06841,
    "P": 97.05276,
    "S": 87.03203,
    "T": 101.04768,
    "W": 186.07931,
    "Y": 163.06333,
    "V": 99.06841,
}

# Average residue masses (Da) - more commonly used for proteins
AVERAGE_MASSES: dict[str, float] = {
    "A": 71.0788,
    "R": 156.1875,
    "N": 114.1038,
    "D": 115.0886,
    "C": 103.1388,
    "E": 129.1155,
    "Q": 128.1307,
    "G": 57.0519,
    "H": 137.1411,
    "I": 113.1594,
    "L": 113.1594,
    "K": 128.1741,
    "M": 131.1926,
    "F": 147.1766,
    "P": 97.1167,
    "S": 87.0782,
    "T": 101.1051,
    "W": 186.2132,
    "Y": 163.1760,
    "V": 99.1326,
}

WATER_MASS = 18.01524

# pKa values for isoelectric point calculation
# Format: (pKa_COOH, pKa_NH3+, pKa_sidechain)
# Side-chain pKa values from Lehninger Principles of Biochemistry
PKA_CTERM = 3.65  # C-terminal carboxyl
PKA_NTERM = 8.00  # N-terminal amino
PKA_SIDECHAINS: dict[str, tuple[float, int]] = {
    # (pKa, charge at low pH: +1 for basic, -1 for acidic)
    "C": (8.18, -1),
    "D": (3.65, -1),
    "E": (4.25, -1),
    "H": (6.00, 1),
    "K": (10.53, 1),
    "R": (12.48, 1),
    "Y": (10.07, -1),
}

# Kyte-Doolittle hydrophobicity scale
KYTE_DOOLITTLE: dict[str, float] = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

# Chou-Fasman parameters: (P_alpha, P_beta, P_turn)
# From Chou & Fasman (1978) Adv. Enzymol. 47:45-148
CHOU_FASMAN: dict[str, tuple[int, int, int]] = {
    "A": (142, 83, 66),
    "R": (98, 93, 95),
    "N": (67, 89, 156),
    "D": (101, 54, 146),
    "C": (70, 119, 119),
    "Q": (111, 110, 98),
    "E": (151, 37, 74),
    "G": (57, 75, 156),
    "H": (100, 87, 95),
    "I": (108, 160, 47),
    "L": (121, 130, 59),
    "K": (114, 74, 101),
    "M": (145, 105, 60),
    "F": (113, 138, 60),
    "P": (57, 55, 152),
    "S": (77, 75, 143),
    "T": (83, 119, 96),
    "W": (108, 137, 96),
    "Y": (69, 147, 114),
    "V": (106, 170, 50),
}

# Von Heijne signal peptide characteristics
# Hydrophobic core residues typical in signal peptides
SIGNAL_HYDROPHOBIC = set("AILMFVW")


@dataclass
class SequenceAnalysisResult:
    """Complete result from sequence analysis."""

    sequence: str
    length: int
    amino_acid_composition: dict[str, float]
    amino_acid_counts: dict[str, int]
    molecular_weight: float
    isoelectric_point: float
    hydrophobicity_profile: list[float]
    secondary_structure: list[str]
    has_signal_peptide: bool
    signal_peptide_length: int
    disulfide_bonds: list[tuple[int, int]]
    charge_at_ph7: float
    gravy: float  # Grand Average of Hydropathicity
    aromaticity: float
    instability_index: float


@dataclass
class SecondaryStructurePrediction:
    """Secondary structure prediction result."""

    sequence: str
    prediction: list[str]  # H=helix, E=sheet, C=coil
    helix_propensities: list[float]
    sheet_propensities: list[float]
    turn_propensities: list[float]
    helix_fraction: float
    sheet_fraction: float
    coil_fraction: float


def validate_sequence(sequence: str) -> str:
    """Validate and clean a protein sequence.

    Args:
        sequence: Raw protein sequence string.

    Returns:
        Cleaned uppercase sequence.

    Raises:
        ValueError: If sequence contains invalid characters.
    """
    cleaned = sequence.upper().replace(" ", "").replace("\n", "").replace("\r", "")

    # Remove FASTA header if present
    if cleaned.startswith(">"):
        lines = cleaned.split("\n")
        cleaned = "".join(lines[1:])

    invalid = set(cleaned) - AMINO_ACIDS
    if invalid:
        raise ValueError(f"Invalid amino acid characters: {invalid}")

    if len(cleaned) == 0:
        raise ValueError("Empty sequence provided")

    return cleaned


def amino_acid_composition(sequence: str) -> tuple[dict[str, int], dict[str, float]]:
    """Calculate amino acid composition.

    Args:
        sequence: Validated protein sequence.

    Returns:
        Tuple of (counts dict, frequency dict as percentages).
    """
    counts: dict[str, int] = {}
    for aa in AMINO_ACIDS:
        count = sequence.count(aa)
        if count > 0:
            counts[aa] = count

    total = len(sequence)
    frequencies = {aa: round(count / total * 100, 2) for aa, count in counts.items()}

    return counts, frequencies


def molecular_weight(sequence: str, monoisotopic: bool = False) -> float:
    """Calculate molecular weight of a protein sequence.

    Uses the sum of residue masses plus one water molecule (for the
    uncleaved termini).

    Args:
        sequence: Validated protein sequence.
        monoisotopic: If True, use monoisotopic masses instead of average.

    Returns:
        Molecular weight in Daltons.
    """
    masses = RESIDUE_MASSES if monoisotopic else AVERAGE_MASSES
    weight = sum(masses[aa] for aa in sequence) + WATER_MASS
    return round(weight, 4)


def _charge_at_ph(sequence: str, ph: float) -> float:
    """Calculate the net charge of a protein at a given pH.

    Uses the Henderson-Hasselbalch equation.

    Args:
        sequence: Validated protein sequence.
        ph: pH value.

    Returns:
        Net charge at the given pH.
    """
    # N-terminal positive charge
    charge = 1.0 / (1.0 + 10 ** (ph - PKA_NTERM))

    # C-terminal negative charge
    charge -= 1.0 / (1.0 + 10 ** (PKA_CTERM - ph))

    # Side chain contributions
    for aa, (pka, sign) in PKA_SIDECHAINS.items():
        count = sequence.count(aa)
        if count == 0:
            continue
        if sign == 1:  # Basic residue (positive at low pH)
            charge += count * (1.0 / (1.0 + 10 ** (ph - pka)))
        else:  # Acidic residue (negative at high pH)
            charge -= count * (1.0 / (1.0 + 10 ** (pka - ph)))

    return charge


def isoelectric_point(sequence: str, precision: float = 0.01) -> float:
    """Estimate the isoelectric point (pI) using bisection method.

    The pI is the pH at which the net charge is zero.

    Args:
        sequence: Validated protein sequence.
        precision: Desired precision of the result.

    Returns:
        Estimated isoelectric point.
    """
    ph_low = 0.0
    ph_high = 14.0

    while (ph_high - ph_low) > precision:
        ph_mid = (ph_low + ph_high) / 2.0
        charge = _charge_at_ph(sequence, ph_mid)

        if charge > 0:
            ph_low = ph_mid
        else:
            ph_high = ph_mid

    return round((ph_low + ph_high) / 2.0, 2)


def hydrophobicity_profile(sequence: str, window_size: int = 9) -> list[float]:
    """Calculate Kyte-Doolittle hydrophobicity profile.

    Uses a sliding window average of hydrophobicity values.

    Args:
        sequence: Validated protein sequence.
        window_size: Size of the sliding window (default 9, commonly 7-21).

    Returns:
        List of hydrophobicity values, one per residue position.
        Values at the edges are averaged over smaller windows.
    """
    if window_size < 1:
        raise ValueError("Window size must be >= 1")

    n = len(sequence)
    profile = []

    half_window = window_size // 2

    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        window = sequence[start:end]
        avg = sum(KYTE_DOOLITTLE[aa] for aa in window) / len(window)
        profile.append(round(avg, 3))

    return profile


def predict_secondary_structure(sequence: str) -> SecondaryStructurePrediction:
    """Predict secondary structure using the Chou-Fasman method.

    The algorithm:
    1. Calculate helix, sheet, and turn propensities for each residue.
    2. Use a sliding window to identify nucleation sites for helices (window=6)
       and sheets (window=5).
    3. A helix nucleation occurs when >= 4 of 6 residues have P_alpha >= 100.
    4. A sheet nucleation occurs when >= 3 of 5 residues have P_beta >= 100.
    5. Extend regions in both directions while average propensity > 100.
    6. Turn predictions based on turn propensity at position i through i+3.
    7. Resolve conflicts: helix wins if P_alpha > P_beta, else sheet.

    Args:
        sequence: Validated protein sequence.

    Returns:
        SecondaryStructurePrediction with per-residue assignments.
    """
    n = len(sequence)
    p_alpha = [CHOU_FASMAN[aa][0] for aa in sequence]
    p_beta = [CHOU_FASMAN[aa][1] for aa in sequence]
    p_turn = [CHOU_FASMAN[aa][2] for aa in sequence]

    # Initialize assignments
    helix_regions = [False] * n
    sheet_regions = [False] * n
    turn_regions = [False] * n

    # Step 1: Find helix nucleation sites (4 of 6 consecutive with P_alpha >= 100)
    for i in range(n - 5):
        window = p_alpha[i : i + 6]
        if sum(1 for p in window if p >= 100) >= 4:
            # Nucleation found - mark core
            for j in range(i, i + 6):
                helix_regions[j] = True

    # Extend helix regions
    for i in range(n):
        if helix_regions[i]:
            # Extend left
            j = i - 1
            while j >= 0 and not helix_regions[j]:
                # Check if average P_alpha of extending region > 100
                if p_alpha[j] >= 100:
                    helix_regions[j] = True
                    j -= 1
                else:
                    break
            # Extend right
            j = i + 1
            while j < n and not helix_regions[j]:
                if p_alpha[j] >= 100:
                    helix_regions[j] = True
                    j += 1
                else:
                    break

    # Step 2: Find sheet nucleation sites (3 of 5 consecutive with P_beta >= 100)
    for i in range(n - 4):
        window = p_beta[i : i + 5]
        if sum(1 for p in window if p >= 100) >= 3:
            for j in range(i, i + 5):
                sheet_regions[j] = True

    # Extend sheet regions
    for i in range(n):
        if sheet_regions[i]:
            j = i - 1
            while j >= 0 and not sheet_regions[j]:
                if p_beta[j] >= 100:
                    sheet_regions[j] = True
                    j -= 1
                else:
                    break
            j = i + 1
            while j < n and not sheet_regions[j]:
                if p_beta[j] >= 100:
                    sheet_regions[j] = True
                    j += 1
                else:
                    break

    # Step 3: Turn prediction (4-residue windows)
    for i in range(n - 3):
        turn_score = sum(p_turn[i : i + 4]) / 4
        if turn_score > 100 and p_turn[i] > p_alpha[i] and p_turn[i] > p_beta[i]:
            for j in range(i, min(i + 4, n)):
                turn_regions[j] = True

    # Step 4: Resolve conflicts
    prediction: list[str] = []
    for i in range(n):
        if helix_regions[i] and sheet_regions[i]:
            # Conflict: higher propensity wins
            if p_alpha[i] >= p_beta[i]:
                prediction.append("H")
            else:
                prediction.append("E")
        elif helix_regions[i]:
            prediction.append("H")
        elif sheet_regions[i]:
            prediction.append("E")
        elif turn_regions[i]:
            prediction.append("C")  # Turns are part of coil
        else:
            prediction.append("C")

    helix_count = prediction.count("H")
    sheet_count = prediction.count("E")
    coil_count = prediction.count("C")

    return SecondaryStructurePrediction(
        sequence=sequence,
        prediction=prediction,
        helix_propensities=[p / 100.0 for p in p_alpha],
        sheet_propensities=[p / 100.0 for p in p_beta],
        turn_propensities=[p / 100.0 for p in p_turn],
        helix_fraction=round(helix_count / n, 3) if n > 0 else 0.0,
        sheet_fraction=round(sheet_count / n, 3) if n > 0 else 0.0,
        coil_fraction=round(coil_count / n, 3) if n > 0 else 0.0,
    )


def detect_signal_peptide(sequence: str) -> tuple[bool, int]:
    """Detect putative signal peptide using Von Heijne's method.

    Signal peptides typically have:
    1. A positively charged n-region (1-5 residues, contains K or R)
    2. A hydrophobic h-region (7-15 residues, mostly hydrophobic)
    3. A c-region with a cleavage site (3-7 residues, small neutral at -1,-3)

    This is a simplified heuristic, not a full SignalP implementation.

    Args:
        sequence: Validated protein sequence.

    Returns:
        Tuple of (has_signal, predicted_cleavage_position).
    """
    if len(sequence) < 15:
        return False, 0

    # Check n-region (first 1-5 residues): should have positive charge
    n_region = sequence[:5]
    has_positive = any(aa in "KR" for aa in n_region)

    if not has_positive:
        return False, 0

    # Check h-region (residues ~3-20): should be predominantly hydrophobic
    best_score = 0.0
    best_end = 0

    for h_start in range(2, 6):
        for h_end in range(h_start + 7, min(h_start + 16, len(sequence))):
            h_region = sequence[h_start:h_end]
            hydrophobic_fraction = sum(1 for aa in h_region if aa in SIGNAL_HYDROPHOBIC) / len(
                h_region
            )
            avg_hydro = sum(KYTE_DOOLITTLE[aa] for aa in h_region) / len(h_region)

            if hydrophobic_fraction >= 0.6 and avg_hydro > 1.0:
                score = hydrophobic_fraction * avg_hydro
                if score > best_score:
                    best_score = score
                    best_end = h_end

    if best_score == 0:
        return False, 0

    # Check c-region: small neutral residues at -1 and -3 from cleavage site
    # The Ala-X-Ala rule (AXA rule): positions -3 and -1 should be small neutral
    small_neutral = set("AGSTV")
    for cleavage in range(best_end + 1, min(best_end + 8, len(sequence))):
        if (
            cleavage >= 2
            and sequence[cleavage - 1] in small_neutral
            and sequence[cleavage - 3] in small_neutral
        ):
            return True, cleavage

    # Fallback: if hydrophobic region found, estimate cleavage
    if best_end + 3 < len(sequence):
        return True, best_end + 3

    return False, 0


def predict_disulfide_bonds(sequence: str) -> list[tuple[int, int]]:
    """Predict potential disulfide bonds between cysteine residues.

    Uses a simplified approach based on:
    1. Identifying all cysteine positions
    2. Scoring pairs based on sequence separation and local context
    3. Cysteines in hydrophobic environments are more likely to form bonds
    4. Minimum sequence separation of 4 residues

    This is a heuristic prediction - real prediction requires 3D structure.

    Args:
        sequence: Validated protein sequence.

    Returns:
        List of (position_i, position_j) tuples (1-indexed).
    """
    cys_positions = [i for i, aa in enumerate(sequence) if aa == "C"]

    if len(cys_positions) < 2:
        return []

    # Score all possible cysteine pairs
    pair_scores: list[tuple[int, int, float]] = []

    for i in range(len(cys_positions)):
        for j in range(i + 1, len(cys_positions)):
            pos_i = cys_positions[i]
            pos_j = cys_positions[j]
            sep = pos_j - pos_i

            # Minimum separation of 4 residues for disulfide bond
            if sep < 4:
                continue

            # Score based on sequence separation (optimal around 10-50)
            if 10 <= sep <= 50:
                sep_score = 1.0
            elif sep < 10:
                sep_score = 0.5
            else:
                sep_score = max(0.3, 1.0 - (sep - 50) / 200)

            # Score based on local hydrophobicity (disulfide bonds
            # tend to form in structured, somewhat hydrophobic regions)
            window = 5
            start_i = max(0, pos_i - window)
            end_i = min(len(sequence), pos_i + window + 1)
            start_j = max(0, pos_j - window)
            end_j = min(len(sequence), pos_j + window + 1)

            local_hydro_i = np.mean([KYTE_DOOLITTLE[aa] for aa in sequence[start_i:end_i]])
            local_hydro_j = np.mean([KYTE_DOOLITTLE[aa] for aa in sequence[start_j:end_j]])

            # Moderate hydrophobicity is better (not too hydrophilic, not transmembrane)
            hydro_score_i = 1.0 - abs(float(local_hydro_i) - 0.5) / 5.0
            hydro_score_j = 1.0 - abs(float(local_hydro_j) - 0.5) / 5.0
            hydro_score = (hydro_score_i + hydro_score_j) / 2

            total_score = sep_score * 0.5 + hydro_score * 0.5
            pair_scores.append((pos_i, pos_j, total_score))

    # Greedy pairing: take best scoring pairs without reusing cysteines
    pair_scores.sort(key=lambda x: x[2], reverse=True)
    used: set[int] = set()
    bonds: list[tuple[int, int]] = []

    for pos_i, pos_j, score in pair_scores:
        if pos_i not in used and pos_j not in used and score > 0.4:
            bonds.append((pos_i + 1, pos_j + 1))  # 1-indexed
            used.add(pos_i)
            used.add(pos_j)

    bonds.sort()
    return bonds


def gravy_score(sequence: str) -> float:
    """Calculate Grand Average of Hydropathicity (GRAVY).

    Args:
        sequence: Validated protein sequence.

    Returns:
        GRAVY score (positive = hydrophobic, negative = hydrophilic).
    """
    return round(sum(KYTE_DOOLITTLE[aa] for aa in sequence) / len(sequence), 4)


def aromaticity(sequence: str) -> float:
    """Calculate aromaticity (frequency of Phe + Trp + Tyr).

    Args:
        sequence: Validated protein sequence.

    Returns:
        Aromaticity as a fraction.
    """
    aromatic_count = sum(1 for aa in sequence if aa in "FWY")
    return round(aromatic_count / len(sequence), 4)


# Instability index dipeptide weights (DIWV)
# From Guruprasad, Reddy, Pandit (1990) Protein Eng. 4:155-161
# Subset of the 400 dipeptide weights - using simplified version
_INSTABILITY_WEIGHTS: dict[str, float] = {
    "WW": 1.0,
    "WC": 1.0,
    "WM": 24.68,
    "WH": 24.68,
    "CW": -14.03,
    "CH": 33.60,
    "CK": 1.0,
    "CC": 1.0,
    "MM": -5.46,
    "MH": 58.28,
    "MK": 1.0,
    "MC": 1.0,
    "HH": 1.0,
    "HK": 1.0,
    "HC": 1.0,
    "HM": 1.0,
    "EE": 33.60,
    "ED": -6.54,
    "EK": 1.0,
    "EH": -6.54,
    "DD": 1.0,
    "DK": -7.49,
    "DH": 1.0,
    "DE": 1.0,
    "KK": 1.0,
    "KH": 1.0,
    "KE": 1.0,
    "KD": 1.0,
    "AA": 1.0,
    "AG": 1.0,
    "AV": 1.0,
    "AL": 1.0,
    "GG": 13.34,
    "GA": 1.0,
    "GV": 1.0,
    "GL": 1.0,
    "VV": 1.0,
    "VA": 1.0,
    "VG": -7.49,
    "VL": 1.0,
    "LL": 1.0,
    "LA": 1.0,
    "LG": 14.45,
    "LV": 1.0,
}


def instability_index(sequence: str) -> float:
    """Calculate the instability index.

    A protein with instability index > 40 is predicted to be unstable.

    Args:
        sequence: Validated protein sequence.

    Returns:
        Instability index value.
    """
    n = len(sequence)
    if n < 2:
        return 0.0

    total = 0.0
    for i in range(n - 1):
        dipeptide = sequence[i : i + 2]
        total += _INSTABILITY_WEIGHTS.get(dipeptide, 1.0)

    return round((10.0 / n) * total, 2)


def analyze_sequence(sequence: str) -> SequenceAnalysisResult:
    """Perform complete sequence analysis.

    Args:
        sequence: Raw protein sequence (may contain whitespace, FASTA header).

    Returns:
        SequenceAnalysisResult with all computed properties.

    Raises:
        ValueError: If sequence is invalid.
    """
    seq = validate_sequence(sequence)
    counts, composition = amino_acid_composition(seq)
    mw = molecular_weight(seq)
    pi = isoelectric_point(seq)
    hydro_profile = hydrophobicity_profile(seq)
    ss_pred = predict_secondary_structure(seq)
    has_signal, signal_len = detect_signal_peptide(seq)
    ss_bonds = predict_disulfide_bonds(seq)
    charge7 = _charge_at_ph(seq, 7.0)
    grav = gravy_score(seq)
    arom = aromaticity(seq)
    instab = instability_index(seq)

    return SequenceAnalysisResult(
        sequence=seq,
        length=len(seq),
        amino_acid_composition=composition,
        amino_acid_counts=counts,
        molecular_weight=mw,
        isoelectric_point=pi,
        hydrophobicity_profile=hydro_profile,
        secondary_structure=ss_pred.prediction,
        has_signal_peptide=has_signal,
        signal_peptide_length=signal_len,
        disulfide_bonds=ss_bonds,
        charge_at_ph7=round(charge7, 2),
        gravy=grav,
        aromaticity=arom,
        instability_index=instab,
    )
