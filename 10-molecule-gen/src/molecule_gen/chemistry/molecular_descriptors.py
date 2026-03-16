"""Molecular descriptor computation for drug-likeness profiling.

Computes physicochemical and topological descriptors used in pharmaceutical
lead optimization and ADMET (Absorption, Distribution, Metabolism, Excretion,
Toxicity) profiling. Descriptors fall into several categories:

    2D Topological Descriptors:
        Ring counts, bond counts, atom types, graph-based indices.

    Physicochemical Properties:
        LogP, TPSA, molecular weight, H-bond donors/acceptors, etc.

    Drug-likeness Metrics:
        Lipinski Ro5, Veber rules, Ghose filter, lead-likeness.

    Toxicity Alerts:
        PAINS (Pan-Assay INterference compoundS) substructure filter to
        flag molecules likely to produce false positives in HTS screens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MolecularDescriptors:
    """Comprehensive set of computed molecular descriptors.

    All values are set to None if computation fails (e.g., invalid SMILES
    or RDKit unavailable).
    """

    # Identity
    smiles: str
    canonical_smiles: str | None = None

    # Basic counts
    num_atoms: int | None = None
    num_heavy_atoms: int | None = None
    num_bonds: int | None = None
    num_rings: int | None = None
    num_aromatic_rings: int | None = None
    num_heteroatoms: int | None = None

    # Physicochemical properties
    molecular_weight: float | None = None
    exact_mass: float | None = None
    logp: float | None = None  # Wildman-Crippen LogP
    tpsa: float | None = None  # Topological Polar Surface Area
    molar_refractivity: float | None = None

    # H-bonding
    num_hbd: int | None = None  # Hydrogen bond donors
    num_hba: int | None = None  # Hydrogen bond acceptors

    # Flexibility
    num_rotatable_bonds: int | None = None
    num_rigid_bonds: int | None = None

    # Fraction-based
    fraction_sp3: float | None = None  # Fraction of sp3-hybridized carbons

    # Drug-likeness flags
    lipinski_violations: int | None = None
    is_lipinski_compliant: bool | None = None
    veber_compliant: bool | None = None
    ghose_compliant: bool | None = None
    is_lead_like: bool | None = None

    # Complexity
    bertz_complexity: float | None = None


def compute_descriptors(smiles: str) -> MolecularDescriptors:
    """Compute a full panel of molecular descriptors from SMILES.

    Uses RDKit for accurate descriptor computation. Falls back to
    estimated values from SMILES string analysis if RDKit is unavailable.

    Args:
        smiles: Input SMILES string.

    Returns:
        MolecularDescriptors dataclass with computed values.
    """
    desc = MolecularDescriptors(smiles=smiles)

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning("Invalid SMILES, cannot compute descriptors: %s", smiles)
            return desc

        desc.canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

        # Basic atom/bond counts
        desc.num_atoms = mol.GetNumAtoms()
        desc.num_heavy_atoms = Descriptors.HeavyAtomCount(mol)
        desc.num_bonds = mol.GetNumBonds()

        # Ring information
        ring_info = mol.GetRingInfo()
        desc.num_rings = ring_info.NumRings()
        desc.num_aromatic_rings = Descriptors.NumAromaticRings(mol)

        # Heteroatom count
        desc.num_heteroatoms = Descriptors.NumHeteroatoms(mol)

        # Molecular weight
        desc.molecular_weight = Descriptors.MolWt(mol)
        desc.exact_mass = Descriptors.ExactMolWt(mol)

        # LogP (Wildman-Crippen method)
        desc.logp = Descriptors.MolLogP(mol)

        # Topological Polar Surface Area (Ertl et al., 2000)
        # Key predictor of intestinal absorption and blood-brain barrier
        # penetration. TPSA < 140 A^2 favors oral absorption.
        desc.tpsa = Descriptors.TPSA(mol)

        # Molar refractivity (related to molecular volume/polarizability)
        desc.molar_refractivity = Descriptors.MolMR(mol)

        # Hydrogen bonding capacity
        desc.num_hbd = Descriptors.NumHDonors(mol)
        desc.num_hba = Descriptors.NumHAcceptors(mol)

        # Bond flexibility
        desc.num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        total_bonds = mol.GetNumBonds()
        desc.num_rigid_bonds = total_bonds - (desc.num_rotatable_bonds or 0)

        # Carbon sp3 fraction (important for 3D shape and selectivity)
        desc.fraction_sp3 = Descriptors.FractionCSP3(mol)

        # Bertz topological complexity index
        desc.bertz_complexity = Descriptors.BertzCT(mol)

        # Drug-likeness filters
        desc.lipinski_violations = _count_lipinski_violations(
            mw=desc.molecular_weight,
            logp=desc.logp,
            hbd=desc.num_hbd,
            hba=desc.num_hba,
        )
        desc.is_lipinski_compliant = desc.lipinski_violations <= 1

        desc.veber_compliant = _check_veber_rules(
            tpsa=desc.tpsa,
            rotatable_bonds=desc.num_rotatable_bonds,
        )

        desc.ghose_compliant = _check_ghose_filter(
            mw=desc.molecular_weight,
            logp=desc.logp,
            mr=desc.molar_refractivity,
            num_atoms=desc.num_atoms,
        )

        desc.is_lead_like = _check_lead_likeness(
            mw=desc.molecular_weight,
            logp=desc.logp,
            rotatable_bonds=desc.num_rotatable_bonds,
        )

    except ImportError:
        logger.warning("RDKit not available; computing estimated descriptors from SMILES string")
        desc = _estimate_descriptors_from_smiles(smiles, desc)

    except Exception as e:
        logger.error("Error computing descriptors for %s: %s", smiles, e)

    return desc


def _count_lipinski_violations(
    mw: float | None,
    logp: float | None,
    hbd: int | None,
    hba: int | None,
) -> int:
    """Count Lipinski Rule of Five violations.

    Lipinski's rules (Adv. Drug Deliv. Rev., 1997) provide simple criteria
    for predicting poor absorption or permeation:

        1. MW > 500 Da
        2. LogP > 5
        3. H-bond donors > 5
        4. H-bond acceptors > 10

    Args:
        mw: Molecular weight.
        logp: Calculated LogP.
        hbd: Number of H-bond donors.
        hba: Number of H-bond acceptors.

    Returns:
        Number of violations (0-4).
    """
    violations = 0
    if mw is not None and mw > 500:
        violations += 1
    if logp is not None and logp > 5:
        violations += 1
    if hbd is not None and hbd > 5:
        violations += 1
    if hba is not None and hba > 10:
        violations += 1
    return violations


def _check_veber_rules(
    tpsa: float | None,
    rotatable_bonds: int | None,
) -> bool:
    """Check Veber's rules for oral bioavailability (Veber et al., 2002).

    Compounds with:
        - TPSA <= 140 Angstrom^2
        - Rotatable bonds <= 10
    have a higher probability of good oral bioavailability in rats.

    Args:
        tpsa: Topological polar surface area.
        rotatable_bonds: Number of freely rotatable bonds.

    Returns:
        True if both criteria are satisfied.
    """
    if tpsa is None or rotatable_bonds is None:
        return False
    return tpsa <= 140.0 and rotatable_bonds <= 10


def _check_ghose_filter(
    mw: float | None,
    logp: float | None,
    mr: float | None,
    num_atoms: int | None,
) -> bool:
    """Apply the Ghose filter for drug-likeness (Ghose et al., 1999).

    Qualified ranges for drug-like compounds:
        - 160 <= MW <= 480
        - -0.4 <= LogP <= 5.6
        - 40 <= Molar refractivity <= 130
        - 20 <= Total atom count <= 70

    Args:
        mw: Molecular weight.
        logp: Calculated LogP.
        mr: Molar refractivity.
        num_atoms: Total number of atoms.

    Returns:
        True if all criteria are satisfied.
    """
    if any(v is None for v in [mw, logp, mr, num_atoms]):
        return False
    return (
        160 <= mw <= 480  # type: ignore[operator]
        and -0.4 <= logp <= 5.6  # type: ignore[operator]
        and 40 <= mr <= 130  # type: ignore[operator]
        and 20 <= num_atoms <= 70  # type: ignore[operator]
    )


def _check_lead_likeness(
    mw: float | None,
    logp: float | None,
    rotatable_bonds: int | None,
) -> bool:
    """Check lead-likeness criteria for hit-to-lead optimization.

    Lead-like compounds (Teague et al., 1999) are smaller and less
    lipophilic than drug-like compounds, leaving room for optimization:
        - MW <= 350 Da
        - LogP <= 3.5
        - Rotatable bonds <= 7

    Args:
        mw: Molecular weight.
        logp: Calculated LogP.
        rotatable_bonds: Number of rotatable bonds.

    Returns:
        True if all lead-likeness criteria are met.
    """
    if any(v is None for v in [mw, logp, rotatable_bonds]):
        return False
    return mw <= 350 and logp <= 3.5 and rotatable_bonds <= 7  # type: ignore[operator]


def _estimate_descriptors_from_smiles(
    smiles: str, desc: MolecularDescriptors
) -> MolecularDescriptors:
    """Rough estimation of molecular descriptors directly from SMILES string.

    This fallback method provides approximate values when RDKit is unavailable.
    Estimates are heuristic and should not be used for publication-quality analysis.

    Args:
        smiles: Input SMILES string.
        desc: Partially filled MolecularDescriptors to update.

    Returns:
        Updated MolecularDescriptors with estimated values.
    """
    desc.canonical_smiles = smiles

    # Rough atom count from SMILES (ignoring implicit hydrogens)
    atom_chars = set("CNOSFPIBKcnospb")
    heavy_atoms = sum(1 for c in smiles if c in atom_chars)
    desc.num_heavy_atoms = heavy_atoms
    desc.num_atoms = heavy_atoms  # undercount without explicit H

    # Rough MW estimate: average heavy atom weight ~12-14 Da
    # plus hydrogens
    desc.molecular_weight = heavy_atoms * 13.0

    # Count rings from ring closure digits
    ring_digits = [c for c in smiles if c.isdigit()]
    desc.num_rings = len(ring_digits) // 2

    # Count double/triple bonds
    desc.num_bonds = heavy_atoms - 1 + (desc.num_rings or 0)

    # HBD: roughly count NH and OH patterns
    desc.num_hbd = smiles.count("N") + smiles.count("[NH") + smiles.count("O") - smiles.count("O=")
    desc.num_hba = smiles.upper().count("N") + smiles.upper().count("O")

    return desc


# ---- PAINS (Pan-Assay INterference compoundS) Filters ----

# Common PAINS substructure SMARTS patterns
# These flag molecules with functional groups known to cause false
# positives in high-throughput screening (Baell & Holloway, 2010)
PAINS_SMARTS: dict[str, str] = {
    "quinone": "[#6]1=[#6][#6](=[O])[#6]=[#6][#6]1=[O]",
    "catechol": "c1cc(O)c(O)cc1",
    "rhodanine": "O=C1CSC(=S)N1",
    "michael_acceptor_1": "[#6]=[#6]-[#6]=[O]",
    "phenol_sulfonamide": "c1ccc(S(=O)(=O)N)cc1O",
    "hydroxamic_acid": "[OH]NC(=O)",
    "alkylidene_barbiturate": "O=C1CC(=O)NC(=O)N1",
    "azo_compound": "[#7]=[#7]",
    "hydrazine": "NN",
    "acyl_hydrazine": "C(=O)NN",
    "sulfonyl_hydrazide": "S(=O)(=O)NN",
    "thiocarbonyl": "[#6](=[S])",
    "isothiourea": "SC(=N)",
    "alpha_halo_carbonyl": "[F,Cl,Br,I]CC(=O)",
}


def check_pains_alerts(smiles: str) -> list[str]:
    """Screen a molecule for PAINS substructural alerts.

    PAINS (Baell & Holloway, J. Med. Chem., 2010) are functional groups
    known to interfere with multiple bioassays through nonspecific mechanisms
    such as aggregation, redox cycling, or covalent protein modification.

    Molecules flagged by PAINS should be treated with caution in drug
    discovery campaigns, as their apparent activity may be artifactual.

    Args:
        smiles: Input SMILES string.

    Returns:
        List of triggered alert names. Empty list if no alerts found.
    """
    alerts_found: list[str] = []

    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return alerts_found

        for alert_name, smarts in PAINS_SMARTS.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None and mol.HasSubstructMatch(pattern):
                alerts_found.append(alert_name)

    except ImportError:
        logger.warning("RDKit not available; PAINS filtering requires RDKit")
        # Basic string-based heuristic checks
        alerts_found = _basic_pains_check(smiles)

    return alerts_found


def _basic_pains_check(smiles: str) -> list[str]:
    """Simplified PAINS check using SMILES substrings (no RDKit required).

    This is a rough heuristic and will miss many patterns that require
    substructure matching. Use RDKit-based check_pains_alerts for accuracy.

    Args:
        smiles: Input SMILES string.

    Returns:
        List of approximate alert names.
    """
    alerts: list[str] = []

    if "N=N" in smiles:
        alerts.append("azo_compound")
    if "NN" in smiles and "N=N" not in smiles:
        alerts.append("hydrazine")
    if "C(=S)" in smiles or "C=S" in smiles:
        alerts.append("thiocarbonyl")

    return alerts


def compute_descriptor_vector(smiles: str) -> list[float]:
    """Compute a numeric descriptor vector suitable for ML model input.

    Extracts a fixed-length vector of key physicochemical descriptors
    commonly used as features in QSAR (Quantitative Structure-Activity
    Relationship) models.

    Args:
        smiles: Input SMILES string.

    Returns:
        List of descriptor values: [MW, LogP, TPSA, HBD, HBA, RotBonds,
        Rings, AromaticRings, FracSP3, HeavyAtoms].
    """
    desc = compute_descriptors(smiles)

    return [
        desc.molecular_weight or 0.0,
        desc.logp or 0.0,
        desc.tpsa or 0.0,
        float(desc.num_hbd or 0),
        float(desc.num_hba or 0),
        float(desc.num_rotatable_bonds or 0),
        float(desc.num_rings or 0),
        float(desc.num_aromatic_rings or 0),
        desc.fraction_sp3 or 0.0,
        float(desc.num_heavy_atoms or 0),
    ]
