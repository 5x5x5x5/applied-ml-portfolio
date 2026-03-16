"""SMILES string processing utilities for molecular VAE input/output.

SMILES (Simplified Molecular Input Line Entry System) is the standard
line notation for representing molecular structures as text strings.
This module handles all SMILES-related preprocessing:

    - Character-level tokenization with chemical awareness
    - Vocabulary construction and index mapping
    - Canonical SMILES normalization (via RDKit)
    - Structural validity checking
    - Molecular fingerprint computation (Morgan/ECFP, MACCS)

Tokenization splits SMILES into chemically meaningful tokens including:
    - Single-letter atoms: C, N, O, S, F, P, etc.
    - Two-letter atoms in brackets: Cl, Br, Si, Se, etc.
    - Bond symbols: =, #, -, :
    - Ring closure digits: 0-9
    - Branch markers: (, )
    - Stereo/charge indicators in brackets: [NH+], [C@@H], etc.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Regex pattern for SMILES tokenization
# Captures multi-character tokens like Br, Cl, [nH], [C@@H], etc.
SMILES_TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+\]"  # Bracketed atoms: [nH], [C@@H], [Fe+2], etc.
    r"|Br|Cl"  # Two-letter organic subset atoms
    r"|Si|Se|se"  # Additional two-letter atoms
    r"|@@|@"  # Chirality markers
    r"|%\d{2}"  # Ring closure >= 10
    r"|.)"  # All other single characters
)


# Default SMILES vocabulary covering common organic/medicinal chemistry
DEFAULT_VOCAB_TOKENS: list[str] = [
    "<pad>",  # 0: Padding
    "<sos>",  # 1: Start of sequence
    "<eos>",  # 2: End of sequence
    "<unk>",  # 3: Unknown token
    "C",
    "c",
    "N",
    "n",
    "O",
    "o",
    "S",
    "s",
    "F",
    "Cl",
    "Br",
    "I",
    "P",
    "p",
    "(",
    ")",
    "[",
    "]",
    "=",
    "#",
    "-",
    "+",
    "\\",
    "/",
    ":",
    ".",
    "@",
    "@@",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "0",
    "[nH]",
    "[NH]",
    "[OH]",
    "[N+]",
    "[O-]",
    "[S+]",
    "[C@@H]",
    "[C@H]",
    "[C@@]",
    "[C@]",
    "[Si]",
    "[Se]",
    "[se]",
    "[N]",
    "[S]",
    "[n+]",
    "[NH+]",
    "[NH2+]",
    "[NH3+]",
]


class SMILESVocabulary:
    """Character-level vocabulary for SMILES tokenization.

    Maps between SMILES tokens and integer indices for neural network
    processing. Handles special tokens (PAD, SOS, EOS, UNK) and supports
    dynamic vocabulary building from training corpora.
    """

    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

    def __init__(self, tokens: list[str] | None = None) -> None:
        """Initialize vocabulary from a list of tokens.

        Args:
            tokens: Ordered list of tokens. If None, uses DEFAULT_VOCAB_TOKENS.
                    First 4 tokens must be PAD, SOS, EOS, UNK.
        """
        if tokens is None:
            tokens = list(DEFAULT_VOCAB_TOKENS)

        self.token_to_idx: dict[str, int] = {}
        self.idx_to_token: dict[int, str] = {}

        for idx, token in enumerate(tokens):
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token

    @property
    def size(self) -> int:
        """Number of tokens in the vocabulary."""
        return len(self.token_to_idx)

    def encode_token(self, token: str) -> int:
        """Convert a single token to its integer index.

        Args:
            token: A SMILES token string.

        Returns:
            Integer index, or UNK_IDX if the token is not in vocabulary.
        """
        return self.token_to_idx.get(token, self.UNK_IDX)

    def decode_index(self, idx: int) -> str:
        """Convert an integer index to its token string.

        Args:
            idx: Token index.

        Returns:
            Token string, or UNK_TOKEN if index not found.
        """
        return self.idx_to_token.get(idx, self.UNK_TOKEN)

    def encode_smiles(
        self, smiles: str, max_length: int = 120, add_special: bool = True
    ) -> list[int]:
        """Tokenize and encode a full SMILES string to index sequence.

        Args:
            smiles: Input SMILES string.
            max_length: Maximum sequence length (truncates if exceeded).
            add_special: Whether to prepend SOS and append EOS tokens.

        Returns:
            List of token indices, padded to max_length.
        """
        tokens = tokenize_smiles(smiles)
        indices = [self.encode_token(t) for t in tokens]

        if add_special:
            indices = [self.SOS_IDX] + indices + [self.EOS_IDX]

        # Truncate if needed
        if len(indices) > max_length:
            indices = indices[:max_length]

        # Pad to max_length
        while len(indices) < max_length:
            indices.append(self.PAD_IDX)

        return indices

    def decode_indices(self, indices: list[int], strip_special: bool = True) -> str:
        """Decode a sequence of token indices back to a SMILES string.

        Args:
            indices: List of token indices.
            strip_special: Whether to remove PAD, SOS, EOS tokens.

        Returns:
            Reconstructed SMILES string.
        """
        tokens: list[str] = []
        for idx in indices:
            token = self.decode_index(idx)

            if strip_special:
                if token in (self.PAD_TOKEN, self.SOS_TOKEN):
                    continue
                if token == self.EOS_TOKEN:
                    break

            tokens.append(token)

        return "".join(tokens)

    @classmethod
    def build_from_corpus(
        cls,
        smiles_list: list[str],
        min_frequency: int = 2,
    ) -> SMILESVocabulary:
        """Build a vocabulary from a corpus of SMILES strings.

        Tokenizes all SMILES and includes tokens that appear at least
        min_frequency times. Special tokens are always included.

        Args:
            smiles_list: List of SMILES strings (training set).
            min_frequency: Minimum token count for inclusion.

        Returns:
            Constructed SMILESVocabulary instance.
        """
        counter: Counter[str] = Counter()
        for smiles in smiles_list:
            tokens = tokenize_smiles(smiles)
            counter.update(tokens)

        # Start with special tokens
        vocab_tokens = [cls.PAD_TOKEN, cls.SOS_TOKEN, cls.EOS_TOKEN, cls.UNK_TOKEN]

        # Add frequent tokens in descending frequency order
        for token, count in counter.most_common():
            if count >= min_frequency and token not in vocab_tokens:
                vocab_tokens.append(token)

        logger.info(
            "Built vocabulary with %d tokens from %d SMILES (min_freq=%d)",
            len(vocab_tokens),
            len(smiles_list),
            min_frequency,
        )

        return cls(vocab_tokens)


def tokenize_smiles(smiles: str) -> list[str]:
    """Tokenize a SMILES string into chemically meaningful tokens.

    Uses regex-based tokenization that correctly handles:
        - Multi-character atoms: Br, Cl, Si, Se
        - Bracketed atoms: [nH], [C@@H], [Fe+2]
        - Ring closure numbers: 1-9, %10-%99
        - Stereo markers: @, @@
        - All standard bond/branch symbols

    Args:
        smiles: Input SMILES string.

    Returns:
        List of token strings.
    """
    return SMILES_TOKEN_PATTERN.findall(smiles)


def canonicalize_smiles(smiles: str) -> str | None:
    """Convert a SMILES string to its canonical (unique) form using RDKit.

    Canonical SMILES ensures that the same molecule always produces the
    same string representation, regardless of atom ordering or notation
    variations in the input.

    Args:
        smiles: Input SMILES string.

    Returns:
        Canonical SMILES string, or None if the input is invalid.
    """
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except ImportError:
        logger.warning("RDKit not available; returning original SMILES without canonicalization")
        return smiles
    except Exception:
        logger.debug("Failed to canonicalize SMILES: %s", smiles)
        return None


def is_valid_smiles(smiles: str) -> bool:
    """Check whether a SMILES string represents a valid molecular structure.

    Validates by attempting to parse the SMILES into an RDKit Mol object
    and applying sanitization checks (valence, kekulization, aromaticity).

    Args:
        smiles: SMILES string to validate.

    Returns:
        True if the SMILES is chemically valid, False otherwise.
    """
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except ImportError:
        # Fallback: basic syntax check without RDKit
        return _basic_smiles_syntax_check(smiles)
    except Exception:
        return False


def _basic_smiles_syntax_check(smiles: str) -> bool:
    """Basic SMILES syntax validation without RDKit.

    Checks for balanced parentheses, valid characters, and matched ring
    closure digits. This does NOT validate chemical correctness (valence, etc.).

    Args:
        smiles: SMILES string to check.

    Returns:
        True if basic syntax is valid.
    """
    if not smiles or len(smiles) == 0:
        return False

    # Check balanced parentheses
    depth = 0
    for char in smiles:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if depth < 0:
            return False
    if depth != 0:
        return False

    # Check balanced brackets
    bracket_depth = 0
    for char in smiles:
        if char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
        if bracket_depth < 0:
            return False
    if bracket_depth != 0:
        return False

    # Check ring closure digits are paired
    ring_digits: dict[str, int] = {}
    tokens = tokenize_smiles(smiles)
    for token in tokens:
        if token.isdigit() or (token.startswith("%") and len(token) == 3):
            ring_digits[token] = ring_digits.get(token, 0) + 1

    for count in ring_digits.values():
        if count % 2 != 0:
            return False

    # Check only valid characters present
    valid_chars = set("CNOSFPIHBrcnospbKfleaigu=@#+-\\/:()[].%0123456789")
    for char in smiles:
        if char not in valid_chars:
            return False

    return True


def compute_morgan_fingerprint(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> npt.NDArray[np.float32] | None:
    """Compute Morgan (ECFP-like) circular fingerprint from SMILES.

    Morgan fingerprints (Rogers & Hahn, 2010) encode the local chemical
    environment around each atom up to a given radius, producing a fixed-
    length bit vector suitable for similarity searching and ML input.

    ECFP4 corresponds to radius=2, ECFP6 to radius=3.

    Args:
        smiles: Input SMILES string.
        radius: Fingerprint radius (2 = ECFP4, 3 = ECFP6).
        n_bits: Length of the bit vector.

    Returns:
        Numpy array of shape (n_bits,) with float32 values (0 or 1),
        or None if the SMILES is invalid.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        for idx in fp.GetOnBits():
            arr[idx] = 1.0
        return arr

    except ImportError:
        logger.warning("RDKit not available; cannot compute Morgan fingerprint")
        return None
    except Exception:
        logger.debug("Failed to compute fingerprint for SMILES: %s", smiles)
        return None


def compute_maccs_fingerprint(smiles: str) -> npt.NDArray[np.float32] | None:
    """Compute MACCS structural keys fingerprint from SMILES.

    MACCS (Molecular ACCess System) keys are a set of 166 predefined
    structural patterns. Each bit indicates the presence/absence of a
    specific substructure. MACCS keys are widely used for similarity
    searching in pharmaceutical databases.

    Args:
        smiles: Input SMILES string.

    Returns:
        Numpy array of shape (167,) with float32 values, or None if invalid.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import MACCSkeys

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros(167, dtype=np.float32)
        for idx in fp.GetOnBits():
            arr[idx] = 1.0
        return arr

    except ImportError:
        logger.warning("RDKit not available; cannot compute MACCS fingerprint")
        return None
    except Exception:
        logger.debug("Failed to compute MACCS keys for SMILES: %s", smiles)
        return None


def tanimoto_similarity(fp1: npt.NDArray[Any], fp2: npt.NDArray[Any]) -> float:
    """Compute Tanimoto (Jaccard) similarity between two binary fingerprints.

    Tanimoto coefficient is the standard similarity metric in cheminformatics:
        Tc = |A & B| / |A | B| = intersection / union

    Args:
        fp1: First fingerprint bit vector.
        fp2: Second fingerprint bit vector.

    Returns:
        Tanimoto coefficient in [0, 1].
    """
    intersection = np.sum(np.logical_and(fp1, fp2))
    union = np.sum(np.logical_or(fp1, fp2))
    if union == 0:
        return 0.0
    return float(intersection / union)
