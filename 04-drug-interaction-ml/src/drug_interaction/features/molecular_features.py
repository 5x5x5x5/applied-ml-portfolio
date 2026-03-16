"""Molecular feature extraction from drug SMILES strings.

Extracts physicochemical properties and structural fingerprints for
drug-drug interaction prediction. Supports both single-drug descriptors
and pairwise interaction feature construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy RDKit import -- allows the module to be imported even when rdkit
# is not installed (useful for testing with stubs).
# ---------------------------------------------------------------------------


def _get_rdkit():  # noqa: ANN202
    """Lazily import RDKit modules and return them as a namespace dict."""
    from rdkit import Chem  # type: ignore[import-untyped]
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors  # type: ignore[import-untyped]

    return {
        "Chem": Chem,
        "AllChem": AllChem,
        "Descriptors": Descriptors,
        "rdMolDescriptors": rdMolDescriptors,
    }


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class MolecularDescriptors(BaseModel):
    """Physicochemical descriptors for a single drug molecule."""

    smiles: str
    molecular_weight: float = Field(..., description="Molecular weight in Da")
    logp: float = Field(..., description="Wildman-Crippen LogP")
    hbd: int = Field(..., ge=0, description="Hydrogen-bond donors")
    hba: int = Field(..., ge=0, description="Hydrogen-bond acceptors")
    tpsa: float = Field(..., ge=0.0, description="Topological polar surface area")
    rotatable_bonds: int = Field(..., ge=0, description="Number of rotatable bonds")
    num_rings: int = Field(..., ge=0, description="Number of rings")
    num_aromatic_rings: int = Field(..., ge=0, description="Number of aromatic rings")
    num_heavy_atoms: int = Field(..., ge=0, description="Number of heavy (non-H) atoms")
    fraction_csp3: float = Field(..., ge=0.0, le=1.0, description="Fraction of sp3 carbons")


class PairwiseFeatures(BaseModel):
    """Interaction features derived from a pair of drugs."""

    drug_a_smiles: str
    drug_b_smiles: str
    mw_diff: float
    mw_ratio: float
    logp_diff: float
    logp_sum: float
    hbd_diff: int
    hba_diff: int
    tpsa_diff: float
    tpsa_ratio: float
    rotatable_bonds_diff: int
    tanimoto_similarity: float = Field(
        ..., ge=0.0, le=1.0, description="Tanimoto similarity of Morgan fingerprints"
    )
    dice_similarity: float = Field(..., ge=0.0, le=1.0)
    combined_fingerprint: list[int] = Field(
        default_factory=list,
        description="Concatenated + element-wise product of Morgan fingerprints",
    )


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


@dataclass
class MolecularFeatureExtractor:
    """Extract molecular features from SMILES strings.

    Parameters
    ----------
    fingerprint_radius : int
        Radius for Morgan circular fingerprint.
    fingerprint_nbits : int
        Number of bits for the folded fingerprint.
    """

    fingerprint_radius: int = 2
    fingerprint_nbits: int = 1024
    _rdkit: dict[str, Any] = field(default_factory=dict, repr=False, init=False)

    def __post_init__(self) -> None:
        self._rdkit = _get_rdkit()

    # -- single molecule ---------------------------------------------------

    def compute_descriptors(self, smiles: str) -> MolecularDescriptors:
        """Compute physicochemical descriptors for a single molecule.

        Parameters
        ----------
        smiles : str
            Canonical or isomeric SMILES string.

        Returns
        -------
        MolecularDescriptors
            Pydantic model with computed descriptor values.

        Raises
        ------
        ValueError
            If the SMILES string cannot be parsed.
        """
        Chem = self._rdkit["Chem"]
        Descriptors = self._rdkit["Descriptors"]
        rdMolDescriptors = self._rdkit["rdMolDescriptors"]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles!r}")

        return MolecularDescriptors(
            smiles=smiles,
            molecular_weight=Descriptors.MolWt(mol),
            logp=Descriptors.MolLogP(mol),
            hbd=rdMolDescriptors.CalcNumHBD(mol),
            hba=rdMolDescriptors.CalcNumHBA(mol),
            tpsa=Descriptors.TPSA(mol),
            rotatable_bonds=rdMolDescriptors.CalcNumRotatableBonds(mol),
            num_rings=rdMolDescriptors.CalcNumRings(mol),
            num_aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(mol),
            num_heavy_atoms=mol.GetNumHeavyAtoms(),
            fraction_csp3=rdMolDescriptors.CalcFractionCSP3(mol),
        )

    def compute_morgan_fingerprint(self, smiles: str) -> NDArray[np.int32]:
        """Compute Morgan circular fingerprint as a bit vector.

        Parameters
        ----------
        smiles : str
            SMILES string for the molecule.

        Returns
        -------
        NDArray[np.int32]
            Bit vector of shape ``(fingerprint_nbits,)``.
        """
        Chem = self._rdkit["Chem"]
        AllChem = self._rdkit["AllChem"]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles!r}")

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, self.fingerprint_radius, nBits=self.fingerprint_nbits
        )
        arr = np.zeros(self.fingerprint_nbits, dtype=np.int32)
        for idx in fp.GetOnBits():
            arr[idx] = 1
        return arr

    # -- pairwise features -------------------------------------------------

    def compute_pairwise_features(
        self,
        smiles_a: str,
        smiles_b: str,
        *,
        include_fingerprint: bool = True,
    ) -> PairwiseFeatures:
        """Build interaction features from a pair of drugs.

        The pairwise feature vector includes:
        * absolute differences and ratios of physicochemical properties
        * Tanimoto and Dice similarity of Morgan fingerprints
        * optionally, a combined fingerprint (concatenation + element-wise product)

        Parameters
        ----------
        smiles_a, smiles_b : str
            SMILES for drug A and drug B.
        include_fingerprint : bool
            Whether to include the combined fingerprint vector (can be large).
        """
        from rdkit import DataStructs  # type: ignore[import-untyped]

        desc_a = self.compute_descriptors(smiles_a)
        desc_b = self.compute_descriptors(smiles_b)
        fp_a = self.compute_morgan_fingerprint(smiles_a)
        fp_b = self.compute_morgan_fingerprint(smiles_b)

        # Convert numpy arrays to RDKit ExplicitBitVect for similarity
        Chem = self._rdkit["Chem"]
        AllChem = self._rdkit["AllChem"]
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)
        rdkit_fp_a = AllChem.GetMorganFingerprintAsBitVect(
            mol_a, self.fingerprint_radius, nBits=self.fingerprint_nbits
        )
        rdkit_fp_b = AllChem.GetMorganFingerprintAsBitVect(
            mol_b, self.fingerprint_radius, nBits=self.fingerprint_nbits
        )
        tanimoto = DataStructs.TanimotoSimilarity(rdkit_fp_a, rdkit_fp_b)
        dice = DataStructs.DiceSimilarity(rdkit_fp_a, rdkit_fp_b)

        # Combined fingerprint: concatenation + element-wise product
        combined_fp: list[int] = []
        if include_fingerprint:
            product = (fp_a * fp_b).tolist()
            combined_fp = fp_a.tolist() + fp_b.tolist() + product

        safe_mw_b = desc_b.molecular_weight if desc_b.molecular_weight != 0.0 else 1e-9
        safe_tpsa_b = desc_b.tpsa if desc_b.tpsa != 0.0 else 1e-9

        return PairwiseFeatures(
            drug_a_smiles=smiles_a,
            drug_b_smiles=smiles_b,
            mw_diff=abs(desc_a.molecular_weight - desc_b.molecular_weight),
            mw_ratio=desc_a.molecular_weight / safe_mw_b,
            logp_diff=abs(desc_a.logp - desc_b.logp),
            logp_sum=desc_a.logp + desc_b.logp,
            hbd_diff=abs(desc_a.hbd - desc_b.hbd),
            hba_diff=abs(desc_a.hba - desc_b.hba),
            tpsa_diff=abs(desc_a.tpsa - desc_b.tpsa),
            tpsa_ratio=desc_a.tpsa / safe_tpsa_b,
            rotatable_bonds_diff=abs(desc_a.rotatable_bonds - desc_b.rotatable_bonds),
            tanimoto_similarity=tanimoto,
            dice_similarity=dice,
            combined_fingerprint=combined_fp,
        )

    # -- batch processing --------------------------------------------------

    def extract_batch_descriptors(self, smiles_list: list[str]) -> pd.DataFrame:
        """Extract descriptors for a batch of SMILES strings.

        Invalid SMILES are logged and skipped.

        Returns
        -------
        pd.DataFrame
            One row per valid molecule with descriptor columns.
        """
        records: list[dict[str, Any]] = []
        for smi in smiles_list:
            try:
                desc = self.compute_descriptors(smi)
                records.append(desc.model_dump())
            except ValueError:
                logger.warning("Skipping invalid SMILES: %s", smi)
        return pd.DataFrame(records)

    def extract_pairwise_batch(
        self,
        pairs: list[tuple[str, str]],
        *,
        include_fingerprint: bool = False,
    ) -> pd.DataFrame:
        """Extract pairwise features for a batch of drug pairs.

        Parameters
        ----------
        pairs : list of (smiles_a, smiles_b) tuples
        include_fingerprint : bool
            Whether to embed the combined fingerprint vector in the DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per valid drug pair.
        """
        records: list[dict[str, Any]] = []
        for smiles_a, smiles_b in pairs:
            try:
                pf = self.compute_pairwise_features(
                    smiles_a, smiles_b, include_fingerprint=include_fingerprint
                )
                data = pf.model_dump()
                if include_fingerprint:
                    # Flatten fingerprint into separate columns
                    fp = data.pop("combined_fingerprint")
                    for i, val in enumerate(fp):
                        data[f"fp_{i}"] = val
                else:
                    data.pop("combined_fingerprint", None)
                records.append(data)
            except ValueError:
                logger.warning("Skipping invalid pair: (%s, %s)", smiles_a, smiles_b)
        logger.info("Extracted pairwise features for %d / %d pairs", len(records), len(pairs))
        return pd.DataFrame(records)
