"""Drug database lookup tool -- simulated drug reference database.

Provides a comprehensive simulated drug database with information about
approved drugs, their properties, indications, and clinical data.
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Simulated drug database
# ---------------------------------------------------------------------------

_DRUG_DATABASE: dict[str, dict[str, Any]] = {
    "imatinib": {
        "generic_name": "imatinib mesylate",
        "brand_name": "Gleevec",
        "drug_class": "small_molecule",
        "mechanism": "BCR-ABL tyrosine kinase inhibitor",
        "therapeutic_area": "oncology",
        "indications": [
            "Chronic myeloid leukemia (CML)",
            "Gastrointestinal stromal tumors (GIST)",
            "Ph+ acute lymphoblastic leukemia",
        ],
        "approval_year": 2001,
        "route": "oral",
        "dosage_forms": ["100mg tablet", "400mg tablet"],
        "smiles": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
        "molecular_weight": 493.6,
        "half_life_hours": 18,
        "bioavailability_pct": 98,
        "protein_binding_pct": 95,
        "metabolism": "CYP3A4",
        "black_box_warning": False,
        "rems": False,
    },
    "pembrolizumab": {
        "generic_name": "pembrolizumab",
        "brand_name": "Keytruda",
        "drug_class": "biologic",
        "mechanism": "PD-1 immune checkpoint inhibitor",
        "therapeutic_area": "oncology",
        "indications": [
            "Non-small cell lung cancer",
            "Melanoma",
            "Head and neck squamous cell carcinoma",
            "Classical Hodgkin lymphoma",
            "Urothelial carcinoma",
            "Microsatellite instability-high cancers",
        ],
        "approval_year": 2014,
        "route": "intravenous",
        "dosage_forms": ["100mg/4mL vial", "25mg/mL solution"],
        "smiles": None,
        "molecular_weight": 146000,
        "half_life_hours": 552,
        "bioavailability_pct": 100,
        "protein_binding_pct": None,
        "metabolism": "proteolytic degradation",
        "black_box_warning": False,
        "rems": False,
    },
    "semaglutide": {
        "generic_name": "semaglutide",
        "brand_name": "Ozempic / Wegovy",
        "drug_class": "biologic",
        "mechanism": "GLP-1 receptor agonist",
        "therapeutic_area": "endocrinology",
        "indications": [
            "Type 2 diabetes mellitus",
            "Chronic weight management",
        ],
        "approval_year": 2017,
        "route": "subcutaneous / oral",
        "dosage_forms": ["0.25mg/0.5mL pen", "0.5mg/0.5mL pen", "1mg/0.5mL pen", "14mg tablet"],
        "smiles": None,
        "molecular_weight": 4113.6,
        "half_life_hours": 168,
        "bioavailability_pct": 89,
        "protein_binding_pct": 99,
        "metabolism": "proteolytic cleavage and beta-oxidation",
        "black_box_warning": True,
        "rems": False,
    },
    "osimertinib": {
        "generic_name": "osimertinib mesylate",
        "brand_name": "Tagrisso",
        "drug_class": "small_molecule",
        "mechanism": "Third-generation EGFR tyrosine kinase inhibitor (T790M mutant selective)",
        "therapeutic_area": "oncology",
        "indications": [
            "EGFR T790M mutation-positive NSCLC",
            "First-line EGFR-mutated NSCLC",
        ],
        "approval_year": 2015,
        "route": "oral",
        "dosage_forms": ["40mg tablet", "80mg tablet"],
        "smiles": "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)NC(=O)C=C",
        "molecular_weight": 499.6,
        "half_life_hours": 48,
        "bioavailability_pct": 70,
        "protein_binding_pct": 95,
        "metabolism": "CYP3A4/CYP3A5",
        "black_box_warning": False,
        "rems": False,
    },
    "trastuzumab": {
        "generic_name": "trastuzumab",
        "brand_name": "Herceptin",
        "drug_class": "biologic",
        "mechanism": "HER2-targeted monoclonal antibody",
        "therapeutic_area": "oncology",
        "indications": [
            "HER2-positive breast cancer",
            "HER2-positive metastatic gastric cancer",
        ],
        "approval_year": 1998,
        "route": "intravenous",
        "dosage_forms": ["440mg vial", "150mg vial"],
        "smiles": None,
        "molecular_weight": 148000,
        "half_life_hours": 672,
        "bioavailability_pct": 100,
        "protein_binding_pct": None,
        "metabolism": "proteolytic degradation",
        "black_box_warning": True,
        "rems": False,
    },
    "metformin": {
        "generic_name": "metformin hydrochloride",
        "brand_name": "Glucophage",
        "drug_class": "small_molecule",
        "mechanism": "Biguanide - reduces hepatic glucose production, increases insulin sensitivity",
        "therapeutic_area": "endocrinology",
        "indications": ["Type 2 diabetes mellitus"],
        "approval_year": 1995,
        "route": "oral",
        "dosage_forms": ["500mg tablet", "850mg tablet", "1000mg tablet"],
        "smiles": "CN(C)C(=N)NC(=N)N",
        "molecular_weight": 129.16,
        "half_life_hours": 6.2,
        "bioavailability_pct": 55,
        "protein_binding_pct": 0,
        "metabolism": "Not metabolized (renal excretion)",
        "black_box_warning": True,
        "rems": False,
    },
    "atorvastatin": {
        "generic_name": "atorvastatin calcium",
        "brand_name": "Lipitor",
        "drug_class": "small_molecule",
        "mechanism": "HMG-CoA reductase inhibitor",
        "therapeutic_area": "cardiology",
        "indications": [
            "Hyperlipidemia",
            "Prevention of cardiovascular disease",
        ],
        "approval_year": 1996,
        "route": "oral",
        "dosage_forms": ["10mg tablet", "20mg tablet", "40mg tablet", "80mg tablet"],
        "smiles": "CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4",
        "molecular_weight": 558.64,
        "half_life_hours": 14,
        "bioavailability_pct": 14,
        "protein_binding_pct": 98,
        "metabolism": "CYP3A4",
        "black_box_warning": False,
        "rems": False,
    },
    "adalimumab": {
        "generic_name": "adalimumab",
        "brand_name": "Humira",
        "drug_class": "biologic",
        "mechanism": "TNF-alpha inhibitor (fully human monoclonal antibody)",
        "therapeutic_area": "immunology",
        "indications": [
            "Rheumatoid arthritis",
            "Psoriatic arthritis",
            "Ankylosing spondylitis",
            "Crohn's disease",
            "Ulcerative colitis",
            "Plaque psoriasis",
        ],
        "approval_year": 2002,
        "route": "subcutaneous",
        "dosage_forms": ["40mg/0.8mL pen", "40mg/0.4mL pen"],
        "smiles": None,
        "molecular_weight": 148000,
        "half_life_hours": 336,
        "bioavailability_pct": 64,
        "protein_binding_pct": None,
        "metabolism": "proteolytic degradation",
        "black_box_warning": True,
        "rems": False,
    },
}


class DrugDatabaseTool:
    """Simulated drug reference database for agent lookups."""

    def lookup_drug(self, drug_name: str) -> dict[str, Any] | None:
        """Look up a drug by generic or brand name."""
        name_lower = drug_name.lower().strip()
        logger.info("database.lookup", drug=name_lower)

        # Direct match on generic name
        if name_lower in _DRUG_DATABASE:
            return _DRUG_DATABASE[name_lower]

        # Search by brand name
        for _key, drug in _DRUG_DATABASE.items():
            brand = drug.get("brand_name", "").lower()
            if name_lower in brand or brand in name_lower:
                return drug

        # Partial match on generic name
        for key, drug in _DRUG_DATABASE.items():
            if name_lower in key or key in name_lower:
                return drug

        logger.warning("database.lookup.not_found", drug=name_lower)
        return None

    def search_by_class(self, drug_class: str) -> list[dict[str, Any]]:
        """Search drugs by their class (small_molecule, biologic, etc.)."""
        class_lower = drug_class.lower().strip()
        results = [
            drug
            for drug in _DRUG_DATABASE.values()
            if drug.get("drug_class", "").lower() == class_lower
        ]
        logger.info("database.search_class", drug_class=class_lower, results=len(results))
        return results

    def search_by_therapeutic_area(self, area: str) -> list[dict[str, Any]]:
        """Search drugs by therapeutic area."""
        area_lower = area.lower().strip()
        results = [
            drug
            for drug in _DRUG_DATABASE.values()
            if area_lower in drug.get("therapeutic_area", "").lower()
        ]
        logger.info("database.search_area", area=area_lower, results=len(results))
        return results

    def search_by_mechanism(self, mechanism: str) -> list[dict[str, Any]]:
        """Search drugs by mechanism of action."""
        mech_lower = mechanism.lower().strip()
        results = [
            drug
            for drug in _DRUG_DATABASE.values()
            if mech_lower in drug.get("mechanism", "").lower()
        ]
        logger.info("database.search_mechanism", mechanism=mech_lower, results=len(results))
        return results

    def get_all_drugs(self) -> list[dict[str, Any]]:
        """Return all drugs in the database."""
        return list(_DRUG_DATABASE.values())

    def get_drug_names(self) -> list[str]:
        """Return all generic drug names in the database."""
        return list(_DRUG_DATABASE.keys())
