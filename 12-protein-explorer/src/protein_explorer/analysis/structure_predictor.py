"""Simple protein structure feature prediction.

Implements contact map prediction, solvent accessibility prediction,
disorder prediction (IUPred-like), and domain boundary detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from protein_explorer.analysis.sequence_analyzer import (
    AMINO_ACIDS,
    CHOU_FASMAN,
    KYTE_DOOLITTLE,
    validate_sequence,
)

logger = logging.getLogger(__name__)

# Amino acid index for one-hot encoding
AA_INDEX: dict[str, int] = {aa: i for i, aa in enumerate(sorted(AMINO_ACIDS))}

# Relative solvent accessibility propensities
# Based on Rost & Sander (1994) - average exposed surface area
# Normalized to max ASA values from Tien et al. (2013) PLOS ONE
MAX_ASA: dict[str, float] = {
    "A": 129.0,
    "R": 274.0,
    "N": 195.0,
    "D": 193.0,
    "C": 167.0,
    "Q": 225.0,
    "E": 223.0,
    "G": 104.0,
    "H": 224.0,
    "I": 197.0,
    "L": 201.0,
    "K": 236.0,
    "M": 224.0,
    "F": 240.0,
    "P": 159.0,
    "S": 155.0,
    "T": 172.0,
    "W": 285.0,
    "Y": 263.0,
    "V": 174.0,
}

# Solvent accessibility propensity (higher = more likely exposed)
SA_PROPENSITY: dict[str, float] = {
    "A": 0.49,
    "R": 0.95,
    "N": 0.81,
    "D": 0.81,
    "C": 0.26,
    "Q": 0.84,
    "E": 0.84,
    "G": 0.48,
    "H": 0.66,
    "I": 0.22,
    "L": 0.28,
    "K": 0.97,
    "M": 0.36,
    "F": 0.24,
    "P": 0.75,
    "S": 0.65,
    "T": 0.59,
    "W": 0.30,
    "Y": 0.41,
    "V": 0.25,
}

# Disorder propensity values (higher = more likely disordered)
# Based on amino acid frequencies in disordered regions from DisProt
DISORDER_PROPENSITY: dict[str, float] = {
    "A": 0.06,
    "R": 0.18,
    "N": 0.07,
    "D": 0.19,
    "C": -0.20,
    "Q": 0.23,
    "E": 0.30,
    "G": 0.17,
    "H": 0.01,
    "I": -0.49,
    "L": -0.34,
    "K": 0.26,
    "M": -0.20,
    "F": -0.42,
    "P": 0.41,
    "S": 0.14,
    "T": -0.05,
    "W": -0.49,
    "Y": -0.34,
    "V": -0.39,
}


@dataclass
class ContactMapPrediction:
    """Contact map prediction result."""

    sequence: str
    contact_map: list[list[float]]  # N x N probability matrix
    predicted_contacts: list[tuple[int, int, float]]  # (i, j, probability)


@dataclass
class SolventAccessibilityPrediction:
    """Solvent accessibility prediction result."""

    sequence: str
    accessibility: list[float]  # Per-residue RSA (0-1)
    burial_state: list[str]  # B=buried, E=exposed, I=intermediate
    fraction_buried: float
    fraction_exposed: float


@dataclass
class DisorderPrediction:
    """Intrinsic disorder prediction result."""

    sequence: str
    disorder_scores: list[float]  # Per-residue disorder probability
    disordered_residues: list[bool]
    disordered_regions: list[tuple[int, int]]  # (start, end) 1-indexed
    fraction_disordered: float


@dataclass
class DomainPrediction:
    """Domain boundary detection result."""

    sequence: str
    domain_boundaries: list[int]  # Boundary positions
    domains: list[tuple[int, int]]  # (start, end) 1-indexed
    boundary_scores: list[float]  # Per-residue boundary score
    num_domains: int


def _encode_residue_window(
    sequence: str, position: int, window_size: int = 7
) -> NDArray[np.float64]:
    """Encode a residue and its local window as feature vector.

    Features per position:
    - One-hot amino acid encoding (20)
    - Hydrophobicity value (1)
    - Secondary structure propensities (3)
    - Solvent accessibility propensity (1)
    Total: 25 features per position in window

    Args:
        sequence: Protein sequence.
        position: Center position (0-indexed).
        window_size: Size of the window around the position.

    Returns:
        Feature vector of shape (window_size * 25,).
    """
    half = window_size // 2
    n = len(sequence)
    features = []

    for offset in range(-half, half + 1):
        idx = position + offset
        if 0 <= idx < n:
            aa = sequence[idx]
            # One-hot encoding
            one_hot = [0.0] * 20
            one_hot[AA_INDEX[aa]] = 1.0
            # Hydrophobicity
            hydro = [KYTE_DOOLITTLE[aa] / 4.5]  # Normalize to ~[-1, 1]
            # Chou-Fasman propensities
            cf = CHOU_FASMAN[aa]
            props = [cf[0] / 160.0, cf[1] / 170.0, cf[2] / 160.0]
            # Solvent accessibility
            sa = [SA_PROPENSITY[aa]]
            features.extend(one_hot + hydro + props + sa)
        else:
            # Padding for edges
            features.extend([0.0] * 25)

    return np.array(features, dtype=np.float64)


def predict_contact_map(sequence: str, threshold: float = 8.0) -> ContactMapPrediction:
    """Predict residue-residue contact map.

    Uses a simple statistical potential based on:
    1. Sequence separation (contacts rare at short distances)
    2. Hydrophobic packing (hydrophobic residues tend to contact each other)
    3. Charge complementarity (opposite charges attract)
    4. Cysteine proximity (for disulfide bonds)

    This is a simplified approach - real contact prediction uses deep learning.

    Args:
        sequence: Protein sequence.
        threshold: Distance threshold for contact definition (Angstroms).

    Returns:
        ContactMapPrediction with probability matrix.
    """
    seq = validate_sequence(sequence)
    n = len(seq)

    contact_map = np.zeros((n, n), dtype=np.float64)

    # Charge mapping
    charge_map = {"K": 1, "R": 1, "D": -1, "E": -1}

    for i in range(n):
        for j in range(i + 1, n):
            sep = j - i

            # No contacts at very short sequence separations
            if sep < 4:
                continue

            prob = 0.0

            # Base probability from sequence separation
            # Contacts follow roughly 1/separation relationship
            base_prob = min(0.3, 3.0 / sep)
            prob += base_prob

            # Hydrophobic packing: both hydrophobic residues more likely to contact
            h_i = KYTE_DOOLITTLE[seq[i]]
            h_j = KYTE_DOOLITTLE[seq[j]]
            if h_i > 1.0 and h_j > 1.0:
                prob += 0.15 * min(h_i, h_j) / 4.5

            # Charge complementarity
            c_i = charge_map.get(seq[i], 0)
            c_j = charge_map.get(seq[j], 0)
            if c_i * c_j < 0:  # Opposite charges
                prob += 0.1
            elif c_i * c_j > 0:  # Same charges repel
                prob -= 0.05

            # Cysteine-cysteine contacts (disulfide)
            if seq[i] == "C" and seq[j] == "C" and 10 <= sep <= 50:
                prob += 0.2

            # Aromatic stacking
            if seq[i] in "FWY" and seq[j] in "FWY":
                prob += 0.08

            prob = max(0.0, min(1.0, prob))
            contact_map[i, j] = prob
            contact_map[j, i] = prob

    # Extract top predicted contacts
    contacts = []
    for i in range(n):
        for j in range(i + 5, n):  # Minimum separation of 5
            if contact_map[i, j] > 0.15:
                contacts.append((i + 1, j + 1, round(float(contact_map[i, j]), 3)))

    contacts.sort(key=lambda x: x[2], reverse=True)
    contacts = contacts[: n * 2]  # Keep top 2L contacts

    return ContactMapPrediction(
        sequence=seq,
        contact_map=contact_map.round(3).tolist(),
        predicted_contacts=contacts,
    )


def predict_solvent_accessibility(
    sequence: str, window_size: int = 9
) -> SolventAccessibilityPrediction:
    """Predict relative solvent accessibility (RSA) per residue.

    Uses a sliding-window approach combining:
    1. Intrinsic amino acid accessibility propensity
    2. Local sequence composition (hydrophobic neighborhood buries residues)
    3. Secondary structure context (helices and sheets tend to be buried)

    Args:
        sequence: Protein sequence.
        window_size: Size of the local window.

    Returns:
        SolventAccessibilityPrediction with per-residue values.
    """
    seq = validate_sequence(sequence)
    n = len(seq)
    half = window_size // 2
    accessibility = []

    for i in range(n):
        # Intrinsic propensity
        sa = SA_PROPENSITY[seq[i]]

        # Local environment effect
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window = seq[start:end]

        # Average hydrophobicity of neighbors (hydrophobic neighbors = more buried)
        avg_hydro = np.mean([KYTE_DOOLITTLE[aa] for aa in window])
        hydro_effect = -float(avg_hydro) / 10.0  # More hydrophobic = more buried

        # Composition of charged residues in window (charged = more exposed)
        charged_frac = sum(1 for aa in window if aa in "DERKH") / len(window)
        charge_effect = charged_frac * 0.2

        # Terminal residues tend to be more exposed
        terminal_effect = 0.0
        if i < 5 or i >= n - 5:
            terminal_effect = 0.1

        rsa = sa + hydro_effect + charge_effect + terminal_effect
        rsa = max(0.0, min(1.0, rsa))
        accessibility.append(round(rsa, 3))

    # Classify burial state
    burial_state = []
    for rsa in accessibility:
        if rsa < 0.25:
            burial_state.append("B")  # Buried
        elif rsa > 0.50:
            burial_state.append("E")  # Exposed
        else:
            burial_state.append("I")  # Intermediate

    buried_count = burial_state.count("B")
    exposed_count = burial_state.count("E")

    return SolventAccessibilityPrediction(
        sequence=seq,
        accessibility=accessibility,
        burial_state=burial_state,
        fraction_buried=round(buried_count / n, 3),
        fraction_exposed=round(exposed_count / n, 3),
    )


def predict_disorder(sequence: str, window_size: int = 21) -> DisorderPrediction:
    """Predict intrinsically disordered regions (IUPred-like).

    Uses a simplified energy-based approach inspired by IUPred:
    1. Amino acid disorder propensity
    2. Local sequence complexity (low complexity = more disordered)
    3. Charge-hydropathy plot relationship
    4. Proline and glycine enrichment

    A residue is predicted as disordered if its smoothed score > 0.5.

    Args:
        sequence: Protein sequence.
        window_size: Size of the smoothing window.

    Returns:
        DisorderPrediction with per-residue scores and regions.
    """
    seq = validate_sequence(sequence)
    n = len(seq)
    half = window_size // 2

    raw_scores = []

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window = seq[start:end]
        w_len = len(window)

        # Component 1: Amino acid disorder propensity
        aa_score = np.mean([DISORDER_PROPENSITY[aa] for aa in window])

        # Component 2: Sequence complexity (Shannon entropy)
        aa_counts: dict[str, int] = {}
        for aa in window:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1

        entropy = 0.0
        for count in aa_counts.values():
            p = count / w_len
            if p > 0:
                entropy -= p * np.log2(p)

        max_entropy = np.log2(min(20, w_len))
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
        # Low complexity -> more disordered
        complexity_score = (1.0 - norm_entropy) * 0.3

        # Component 3: Net charge and hydrophobicity
        charge = sum(1 for aa in window if aa in "KR") - sum(1 for aa in window if aa in "DE")
        abs_charge = abs(charge) / w_len
        avg_hydro = np.mean([KYTE_DOOLITTLE[aa] for aa in window])

        # High charge + low hydrophobicity = disordered (Uversky plot)
        ch_score = 0.0
        if abs_charge > 0.1 and avg_hydro < 0.0:
            ch_score = min(0.3, abs_charge * 0.5 + (-float(avg_hydro)) * 0.05)

        # Component 4: Pro/Gly enrichment
        pg_frac = sum(1 for aa in window if aa in "PG") / w_len
        pg_score = pg_frac * 0.2

        total = float(aa_score) + complexity_score + ch_score + pg_score

        # Normalize to 0-1 range
        normalized = 1.0 / (1.0 + np.exp(-total * 3))  # Sigmoid
        raw_scores.append(round(float(normalized), 3))

    # Determine disordered residues (threshold 0.5)
    disordered = [s > 0.5 for s in raw_scores]

    # Find contiguous disordered regions (minimum length 4)
    regions: list[tuple[int, int]] = []
    in_region = False
    region_start = 0

    for i in range(n):
        if disordered[i] and not in_region:
            region_start = i
            in_region = True
        elif not disordered[i] and in_region:
            if i - region_start >= 4:
                regions.append((region_start + 1, i))  # 1-indexed
            in_region = False

    if in_region and n - region_start >= 4:
        regions.append((region_start + 1, n))

    frac_disordered = sum(disordered) / n if n > 0 else 0.0

    return DisorderPrediction(
        sequence=seq,
        disorder_scores=raw_scores,
        disordered_residues=disordered,
        disordered_regions=regions,
        fraction_disordered=round(frac_disordered, 3),
    )


def detect_domain_boundaries(
    sequence: str, window_size: int = 15, min_domain_length: int = 30
) -> DomainPrediction:
    """Detect putative domain boundaries.

    Uses a multi-signal approach:
    1. Hydrophobicity profile changes (linker regions are hydrophilic)
    2. Sequence complexity changes at boundaries
    3. Secondary structure propensity transitions
    4. Proline/glycine enrichment (linkers are enriched)

    Args:
        sequence: Protein sequence.
        window_size: Window for computing local properties.
        min_domain_length: Minimum domain length.

    Returns:
        DomainPrediction with boundaries and domain regions.
    """
    seq = validate_sequence(sequence)
    n = len(seq)

    if n < min_domain_length * 2:
        return DomainPrediction(
            sequence=seq,
            domain_boundaries=[],
            domains=[(1, n)],
            boundary_scores=[0.0] * n,
            num_domains=1,
        )

    half = window_size // 2
    boundary_scores = np.zeros(n, dtype=np.float64)

    # Compute per-residue properties
    hydro = np.array([KYTE_DOOLITTLE[aa] for aa in seq])
    ss_alpha = np.array([CHOU_FASMAN[aa][0] for aa in seq], dtype=np.float64)
    ss_beta = np.array([CHOU_FASMAN[aa][1] for aa in seq], dtype=np.float64)

    for i in range(half, n - half):
        left_start = max(0, i - window_size)
        right_end = min(n, i + window_size)

        left_window = seq[left_start:i]
        right_window = seq[i:right_end]

        if len(left_window) < 3 or len(right_window) < 3:
            continue

        # Signal 1: Hydrophobicity profile change
        left_hydro = np.mean(hydro[left_start:i])
        right_hydro = np.mean(hydro[i:right_end])
        hydro_change = abs(float(left_hydro - right_hydro))

        # Signal 2: Secondary structure propensity change
        left_alpha = np.mean(ss_alpha[left_start:i])
        right_alpha = np.mean(ss_alpha[i:right_end])
        left_beta = np.mean(ss_beta[left_start:i])
        right_beta = np.mean(ss_beta[i:right_end])
        ss_change = (
            abs(float(left_alpha - right_alpha)) / 100 + abs(float(left_beta - right_beta)) / 100
        )

        # Signal 3: Local region is linker-like (hydrophilic, flexible)
        local = seq[max(0, i - 5) : min(n, i + 6)]
        local_hydro = np.mean([KYTE_DOOLITTLE[aa] for aa in local])
        linker_score = 0.0
        if local_hydro < -0.5:
            linker_score = min(1.0, -float(local_hydro) / 3.0)

        # Signal 4: Pro/Gly enrichment in local region
        pg_frac = sum(1 for aa in local if aa in "PGS") / len(local)
        pg_score = pg_frac

        boundary_scores[i] = (
            hydro_change * 0.3 + ss_change * 0.3 + linker_score * 0.25 + pg_score * 0.15
        )

    # Smooth boundary scores
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    if n > kernel_size:
        boundary_scores = np.convolve(boundary_scores, kernel, mode="same")

    # Find peaks in boundary score
    threshold = float(np.mean(boundary_scores) + 1.5 * np.std(boundary_scores))
    threshold = max(threshold, 0.2)  # Minimum threshold

    boundaries: list[int] = []
    for i in range(min_domain_length, n - min_domain_length):
        if boundary_scores[i] > threshold:
            # Local maximum check
            local_region = boundary_scores[max(0, i - 10) : min(n, i + 11)]
            if boundary_scores[i] >= np.max(local_region) - 1e-6:
                # Check minimum distance from other boundaries
                if not boundaries or (i - boundaries[-1]) >= min_domain_length:
                    boundaries.append(i)

    # Build domain list
    domains: list[tuple[int, int]] = []
    prev = 0
    for b in boundaries:
        domains.append((prev + 1, b))  # 1-indexed
        prev = b
    domains.append((prev + 1, n))

    return DomainPrediction(
        sequence=seq,
        domain_boundaries=[b + 1 for b in boundaries],  # 1-indexed
        domains=domains,
        boundary_scores=[round(float(s), 3) for s in boundary_scores],
        num_domains=len(domains),
    )
