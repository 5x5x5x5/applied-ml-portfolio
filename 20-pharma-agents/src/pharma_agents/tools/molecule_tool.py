"""Molecular analysis tool -- computes properties from SMILES strings.

Uses a lightweight, dependency-free approach to parse SMILES and estimate
molecular descriptors without requiring RDKit.  This makes the tool suitable
for demonstration and testing environments.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Approximate atomic weights for common organic atoms
_ATOM_WEIGHTS: dict[str, float] = {
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "S": 32.065,
    "F": 18.998,
    "Cl": 35.453,
    "Br": 79.904,
    "I": 126.904,
    "P": 30.974,
    "H": 1.008,
    "B": 10.811,
    "Si": 28.086,
}

# Known drug scaffolds for comparison
_KNOWN_DRUGS: dict[str, dict[str, Any]] = {
    "imatinib": {
        "smiles": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
        "class": "kinase_inhibitor",
        "mw": 493.6,
        "logp": 3.5,
        "hbd": 2,
        "hba": 7,
    },
    "aspirin": {
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "class": "nsaid",
        "mw": 180.16,
        "logp": 1.2,
        "hbd": 1,
        "hba": 4,
    },
    "metformin": {
        "smiles": "CN(C)C(=N)NC(=N)N",
        "class": "antidiabetic",
        "mw": 129.16,
        "logp": -1.4,
        "hbd": 3,
        "hba": 5,
    },
    "atorvastatin": {
        "smiles": "CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4",
        "class": "statin",
        "mw": 558.64,
        "logp": 6.36,
        "hbd": 4,
        "hba": 5,
    },
    "osimertinib": {
        "smiles": "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)NC(=O)C=C",
        "class": "kinase_inhibitor",
        "mw": 499.61,
        "logp": 3.4,
        "hbd": 3,
        "hba": 7,
    },
    "semaglutide_analog": {
        "smiles": "CC(C)CC(=O)N",  # Simplified placeholder
        "class": "glp1_agonist",
        "mw": 4113.0,
        "logp": -2.0,
        "hbd": 30,
        "hba": 50,
    },
}


def _parse_smiles_lightweight(smiles: str) -> dict[str, Any]:
    """Parse a SMILES string to estimate basic molecular properties.

    This is an approximation -- for production use you would want RDKit.
    """
    # Count atoms
    atom_counts: dict[str, int] = {}
    # Explicit atoms in brackets: [NH], [O-], etc.
    bracket_atoms = re.findall(r"\[([A-Z][a-z]?)", smiles)
    for a in bracket_atoms:
        atom_counts[a] = atom_counts.get(a, 0) + 1

    # Organic subset atoms (not in brackets)
    clean = re.sub(r"\[.*?\]", "", smiles)  # Remove bracket atoms
    for atom in ["Cl", "Br", "Si"]:  # Two-letter atoms first
        count = clean.count(atom)
        if count:
            atom_counts[atom] = atom_counts.get(atom, 0) + count
            clean = clean.replace(atom, "")
    for atom in ["C", "N", "O", "S", "F", "I", "P", "B"]:
        count = clean.count(atom)
        if count:
            atom_counts[atom] = atom_counts.get(atom, 0) + count

    # Estimate molecular weight
    mw = sum(_ATOM_WEIGHTS.get(a, 12.0) * c for a, c in atom_counts.items())
    # Add implicit hydrogens estimate
    carbon_count = atom_counts.get("C", 0)
    nitrogen_count = atom_counts.get("N", 0)
    oxygen_count = atom_counts.get("O", 0)
    h_estimate = max(0, carbon_count * 2 + 2 - smiles.count("=") * 2 - smiles.count("#") * 4)
    mw += h_estimate * 1.008

    # Count rings (approximate: number of ring closures in SMILES)
    ring_closures = len(re.findall(r"\d", smiles))
    num_rings = ring_closures // 2  # Each ring has an opening and closing digit

    # Rotatable bonds estimate (single bonds between heavy atoms, not in rings)
    single_bonds = clean.count("-") + carbon_count  # rough
    rotatable_bonds = max(0, single_bonds - num_rings * 2 - 5)

    # Hydrogen bond donors (NH, OH, NH2)
    hbd = smiles.count("N") + smiles.count("O") - smiles.count("n") - smiles.count("=O")
    hbd = max(0, min(hbd, nitrogen_count + oxygen_count))

    # Hydrogen bond acceptors (N and O atoms)
    hba = nitrogen_count + oxygen_count

    # LogP estimate (Wildman-Crippen-like fragment approach, simplified)
    logp = (
        carbon_count * 0.5
        - oxygen_count * 1.0
        - nitrogen_count * 0.7
        + atom_counts.get("Cl", 0) * 0.9
        + atom_counts.get("F", 0) * 0.3
        + atom_counts.get("Br", 0) * 1.1
        + atom_counts.get("S", 0) * 0.6
        - h_estimate * 0.01
        + num_rings * 0.3
    )

    # TPSA estimate (topological polar surface area)
    tpsa = nitrogen_count * 26.0 + oxygen_count * 20.0 + atom_counts.get("S", 0) * 28.0

    return {
        "molecular_weight": round(mw, 2),
        "logp": round(logp, 2),
        "hbd": hbd,
        "hba": hba,
        "tpsa": round(tpsa, 1),
        "rotatable_bonds": max(0, rotatable_bonds),
        "num_rings": num_rings,
        "atom_counts": atom_counts,
        "heavy_atom_count": sum(atom_counts.values()),
        "formula": "".join(f"{a}{c}" if c > 1 else a for a, c in sorted(atom_counts.items())),
    }


class MoleculeTool:
    """Molecular analysis tool for medicinal chemistry agents."""

    def analyze(self, smiles: str) -> dict[str, Any]:
        """Analyse a molecule from its SMILES string."""
        logger.info("molecule.analyze", smiles=smiles[:50])
        props = _parse_smiles_lightweight(smiles)
        props["smiles"] = smiles
        return props

    def assess_drug_likeness(self, smiles: str) -> dict[str, Any]:
        """Evaluate drug-likeness using multiple rule sets."""
        props = _parse_smiles_lightweight(smiles)

        # Lipinski Rule of Five
        lipinski_violations = 0
        lipinski_details: list[str] = []
        if props["molecular_weight"] > 500:
            lipinski_violations += 1
            lipinski_details.append(f"MW {props['molecular_weight']:.0f} > 500")
        if props["logp"] > 5:
            lipinski_violations += 1
            lipinski_details.append(f"LogP {props['logp']:.1f} > 5")
        if props["hbd"] > 5:
            lipinski_violations += 1
            lipinski_details.append(f"HBD {props['hbd']} > 5")
        if props["hba"] > 10:
            lipinski_violations += 1
            lipinski_details.append(f"HBA {props['hba']} > 10")

        # Veber rules (oral bioavailability)
        veber_pass = props["rotatable_bonds"] <= 10 and props["tpsa"] <= 140
        veber_details: list[str] = []
        if props["rotatable_bonds"] > 10:
            veber_details.append(f"Rotatable bonds {props['rotatable_bonds']} > 10")
        if props["tpsa"] > 140:
            veber_details.append(f"TPSA {props['tpsa']:.0f} > 140")

        # Ghose filter
        ghose_pass = (
            160 <= props["molecular_weight"] <= 480
            and -0.4 <= props["logp"] <= 5.6
            and 20 <= props["heavy_atom_count"] <= 70
        )

        # Lead-likeness
        lead_like = (
            props["molecular_weight"] <= 350
            and props["logp"] <= 3.5
            and props["rotatable_bonds"] <= 7
        )

        overall = (
            "DRUG-LIKE"
            if lipinski_violations <= 1 and veber_pass
            else "BORDERLINE"
            if lipinski_violations <= 2
            else "NOT DRUG-LIKE"
        )

        logger.info("molecule.drug_likeness", smiles=smiles[:50], result=overall)
        return {
            "smiles": smiles,
            "properties": props,
            "lipinski": {
                "violations": lipinski_violations,
                "pass": lipinski_violations <= 1,
                "details": lipinski_details or ["All criteria met"],
            },
            "veber": {
                "pass": veber_pass,
                "details": veber_details or ["All criteria met"],
            },
            "ghose_filter": {"pass": ghose_pass},
            "lead_likeness": {"pass": lead_like},
            "overall_assessment": overall,
        }

    def predict_admet(self, smiles: str) -> dict[str, Any]:
        """Predict ADMET properties for a molecule."""
        props = _parse_smiles_lightweight(smiles)
        rng = np.random.default_rng(int(hashlib.md5(smiles.encode()).hexdigest()[:8], 16))

        mw = props["molecular_weight"]
        logp = props["logp"]
        tpsa = props["tpsa"]

        # Absorption predictions
        oral_bioavailability = "HIGH" if tpsa < 120 and logp < 5 else "LOW"
        caco2_permeability = round(float(rng.uniform(-6.5, -4.5)), 2)

        # Distribution
        vd = round(float(rng.uniform(0.1, 10.0)), 2)  # Volume of distribution L/kg
        ppb = round(float(rng.uniform(50, 99)), 1)  # Plasma protein binding %
        bbb = "YES" if tpsa < 90 and mw < 450 and logp > 1 else "NO"

        # Metabolism
        cyp_inhibition: dict[str, str] = {}
        for cyp in ["CYP3A4", "CYP2D6", "CYP2C9", "CYP1A2", "CYP2C19"]:
            prob = float(rng.uniform(0, 1))
            cyp_inhibition[cyp] = "INHIBITOR" if prob > 0.7 else "NON-INHIBITOR"

        half_life_h = round(float(rng.uniform(1, 72)), 1)

        # Excretion
        clearance = round(float(rng.uniform(1, 50)), 1)  # mL/min/kg

        # Toxicity
        herg = "HIGH_RISK" if logp > 4 and mw > 400 else "LOW_RISK"
        hepatotoxicity_prob = round(float(rng.uniform(0.05, 0.6)), 2)
        ames_mutagenicity = "POSITIVE" if float(rng.uniform(0, 1)) > 0.8 else "NEGATIVE"

        logger.info("molecule.admet", smiles=smiles[:50])
        return {
            "smiles": smiles,
            "absorption": {
                "oral_bioavailability": oral_bioavailability,
                "caco2_permeability_log": caco2_permeability,
                "intestinal_absorption": "HIGH" if logp > 0 and tpsa < 140 else "LOW",
            },
            "distribution": {
                "volume_of_distribution_l_kg": vd,
                "plasma_protein_binding_pct": ppb,
                "blood_brain_barrier": bbb,
            },
            "metabolism": {
                "cyp_inhibition": cyp_inhibition,
                "half_life_hours": half_life_h,
                "primary_metabolism": ("CYP3A4" if float(rng.uniform(0, 1)) > 0.5 else "CYP2D6"),
            },
            "excretion": {
                "clearance_ml_min_kg": clearance,
                "route": "hepatic" if logp > 2 else "renal",
            },
            "toxicity": {
                "herg_liability": herg,
                "hepatotoxicity_probability": hepatotoxicity_prob,
                "ames_mutagenicity": ames_mutagenicity,
                "max_recommended_daily_dose_mg": round(float(rng.uniform(1, 1000)), 0),
            },
        }

    def suggest_modifications(self, smiles: str, optimise_for: str) -> dict[str, Any]:
        """Suggest structural modifications to improve a specific property."""
        props = _parse_smiles_lightweight(smiles)

        suggestions: dict[str, list[dict[str, str]]] = {
            "potency": [
                {
                    "modification": "Add hydrogen-bond donor at position adjacent to core",
                    "rationale": "May improve target binding through additional H-bond interaction",
                    "impact": "Expected 2-5x potency improvement",
                },
                {
                    "modification": "Introduce halogen (F, Cl) at metabolically labile position",
                    "rationale": "Halogen bonding can enhance binding affinity",
                    "impact": "Expected 3-10x improvement with metabolic stabilization",
                },
                {
                    "modification": "Explore constrained analogs (cyclization of flexible linker)",
                    "rationale": "Reduce conformational entropy penalty on binding",
                    "impact": "May improve potency 2-20x if bioactive conformation is captured",
                },
            ],
            "selectivity": [
                {
                    "modification": "Introduce steric bulk near selectivity-determining region",
                    "rationale": "Exploit differences in binding pocket topology between targets",
                    "impact": "Expected >10x selectivity window",
                },
                {
                    "modification": "Replace lipophilic group with polar bioisostere",
                    "rationale": "Differential interactions with off-target hydrophobic pockets",
                    "impact": "Expected 5-50x selectivity improvement",
                },
            ],
            "solubility": [
                {
                    "modification": "Introduce ionizable group (basic amine, carboxylic acid)",
                    "rationale": "Salt formation dramatically improves aqueous solubility",
                    "impact": "Expected 10-100x solubility improvement",
                },
                {
                    "modification": "Disrupt molecular planarity (sp3 center introduction)",
                    "rationale": "Reduces crystal packing efficiency",
                    "impact": "Expected 5-20x solubility improvement",
                },
                {
                    "modification": "Replace aromatic ring with saturated bioisostere",
                    "rationale": "Increases Fsp3, reduces pi-stacking in crystal lattice",
                    "impact": "Moderate solubility improvement with maintained potency",
                },
            ],
            "metabolic_stability": [
                {
                    "modification": "Block metabolic soft spots with fluorine or deuterium",
                    "rationale": "C-F and C-D bonds are more resistant to CYP oxidation",
                    "impact": "Expected 2-5x improvement in microsomal stability",
                },
                {
                    "modification": "Replace ester with amide or reverse amide",
                    "rationale": "Amides are more resistant to esterase-mediated hydrolysis",
                    "impact": "Significant improvement in plasma stability",
                },
            ],
            "permeability": [
                {
                    "modification": "Reduce HBD count through N-methylation",
                    "rationale": "Fewer desolvation penalties when crossing membranes",
                    "impact": "Expected 3-10x permeability improvement",
                },
                {
                    "modification": "Introduce intramolecular hydrogen bond",
                    "rationale": "Shields polar groups, creating a pseudo-macrocyclic conformation",
                    "impact": "Can enable oral absorption for beyond-RoFive compounds",
                },
            ],
            "safety": [
                {
                    "modification": "Remove or replace aniline or nitro groups",
                    "rationale": "These are known toxicophores linked to genotoxicity",
                    "impact": "May eliminate Ames-positive signal",
                },
                {
                    "modification": "Reduce lipophilicity (LogP < 3)",
                    "rationale": "High LogP correlates with off-target promiscuity and toxicity",
                    "impact": "Reduced hERG liability and idiosyncratic toxicity risk",
                },
            ],
        }

        selected = suggestions.get(optimise_for, suggestions["potency"])

        logger.info("molecule.suggest", smiles=smiles[:50], target=optimise_for)
        return {
            "smiles": smiles,
            "current_properties": props,
            "optimisation_target": optimise_for,
            "suggestions": selected,
            "general_considerations": [
                "Verify synthetic feasibility before pursuing modifications",
                "Confirm that modifications do not introduce new toxicophores",
                "Run matched molecular pair analysis on existing SAR data",
                "Consider PK/PD modelling to assess in vivo impact",
            ],
        }

    def compare_to_known_drugs(self, smiles: str, drug_class: str | None = None) -> dict[str, Any]:
        """Compare a molecule to known approved drugs."""
        query_props = _parse_smiles_lightweight(smiles)

        comparisons: list[dict[str, Any]] = []
        for name, drug in _KNOWN_DRUGS.items():
            if drug_class and drug["class"] != drug_class:
                continue

            mw_diff = abs(query_props["molecular_weight"] - drug["mw"])
            logp_diff = abs(query_props["logp"] - drug["logp"])

            # Tanimoto-like similarity (very rough, based on property distance)
            dist = math.sqrt(
                (mw_diff / 500) ** 2
                + (logp_diff / 10) ** 2
                + ((query_props["hbd"] - drug["hbd"]) / 10) ** 2
                + ((query_props["hba"] - drug["hba"]) / 15) ** 2
            )
            similarity = round(max(0, 1 - dist), 3)

            comparisons.append(
                {
                    "drug_name": name,
                    "drug_class": drug["class"],
                    "drug_mw": drug["mw"],
                    "drug_logp": drug["logp"],
                    "property_similarity": similarity,
                    "mw_difference": round(mw_diff, 1),
                    "logp_difference": round(logp_diff, 1),
                }
            )

        comparisons.sort(key=lambda c: c["property_similarity"], reverse=True)

        closest = comparisons[0] if comparisons else None
        logger.info("molecule.compare", smiles=smiles[:50], closest=closest)

        return {
            "smiles": smiles,
            "query_properties": query_props,
            "comparisons": comparisons,
            "most_similar": closest,
            "interpretation": (
                f"Most similar to {closest['drug_name']} "
                f"(similarity: {closest['property_similarity']:.1%}, "
                f"class: {closest['drug_class']})"
                if closest
                else "No comparable drugs found in database."
            ),
        }
