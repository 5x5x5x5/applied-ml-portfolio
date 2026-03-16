"""Unit tests for PharmaAgents tools."""

from __future__ import annotations

import pytest

from pharma_agents.tools.database_tool import DrugDatabaseTool
from pharma_agents.tools.molecule_tool import MoleculeTool
from pharma_agents.tools.pubmed_tool import PubMedTool

# ---------------------------------------------------------------------------
# PubMed Tool
# ---------------------------------------------------------------------------


class TestPubMedTool:
    """Test the simulated PubMed search tool."""

    @pytest.fixture()
    def pubmed(self) -> PubMedTool:
        return PubMedTool()

    def test_search_returns_papers(self, pubmed: PubMedTool) -> None:
        result = pubmed.search("imatinib CML", max_results=5)
        assert result["query"] == "imatinib CML"
        assert result["returned"] == 5
        assert len(result["papers"]) == 5

    def test_search_paper_structure(self, pubmed: PubMedTool) -> None:
        result = pubmed.search("pembrolizumab", max_results=1)
        paper = result["papers"][0]
        required_keys = {
            "pmid",
            "title",
            "authors",
            "journal",
            "year",
            "study_type",
            "evidence_level",
            "abstract",
        }
        assert required_keys.issubset(paper.keys())

    def test_search_deterministic(self, pubmed: PubMedTool) -> None:
        r1 = pubmed.search("test query", max_results=3)
        r2 = pubmed.search("test query", max_results=3)
        assert r1["papers"][0]["pmid"] == r2["papers"][0]["pmid"]

    def test_get_paper_details(self, pubmed: PubMedTool) -> None:
        details = pubmed.get_paper_details("12345678")
        assert details["pmid"] == "12345678"
        assert "methods_summary" in details
        assert "conclusions" in details
        assert "references_count" in details

    def test_cross_reference(self, pubmed: PubMedTool) -> None:
        refs = pubmed.cross_reference("12345678", direction="cited_by")
        assert refs["source_pmid"] == "12345678"
        assert refs["direction"] == "cited_by"
        assert len(refs["related_papers"]) > 0

    def test_summarize_evidence(self, pubmed: PubMedTool) -> None:
        summary = pubmed.summarize_evidence(
            pmids=["11111111", "22222222", "33333333"],
            focus="efficacy of drug X",
        )
        assert summary["papers_analysed"] == 3
        assert summary["focus"] == "efficacy of drug X"
        assert "overall_evidence_quality" in summary
        assert summary["overall_evidence_quality"] in {"HIGH", "MODERATE", "LOW"}


# ---------------------------------------------------------------------------
# Molecule Tool
# ---------------------------------------------------------------------------


class TestMoleculeTool:
    """Test the molecular analysis tool."""

    @pytest.fixture()
    def mol_tool(self) -> MoleculeTool:
        return MoleculeTool()

    # Imatinib SMILES
    SMILES = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"

    def test_analyze_returns_properties(self, mol_tool: MoleculeTool) -> None:
        result = mol_tool.analyze(self.SMILES)
        assert "molecular_weight" in result
        assert "logp" in result
        assert "hbd" in result
        assert "hba" in result
        assert "tpsa" in result
        assert "rotatable_bonds" in result

    def test_analyze_molecular_weight_reasonable(self, mol_tool: MoleculeTool) -> None:
        result = mol_tool.analyze(self.SMILES)
        # Imatinib MW is ~493, our approximation should be in the right ballpark
        assert 200 < result["molecular_weight"] < 800

    def test_drug_likeness_assessment(self, mol_tool: MoleculeTool) -> None:
        result = mol_tool.assess_drug_likeness(self.SMILES)
        assert "lipinski" in result
        assert "veber" in result
        assert "overall_assessment" in result
        assert result["overall_assessment"] in {"DRUG-LIKE", "BORDERLINE", "NOT DRUG-LIKE"}

    def test_drug_likeness_lipinski_structure(self, mol_tool: MoleculeTool) -> None:
        result = mol_tool.assess_drug_likeness(self.SMILES)
        lipinski = result["lipinski"]
        assert "violations" in lipinski
        assert "pass" in lipinski
        assert isinstance(lipinski["violations"], int)

    def test_predict_admet(self, mol_tool: MoleculeTool) -> None:
        result = mol_tool.predict_admet(self.SMILES)
        assert "absorption" in result
        assert "distribution" in result
        assert "metabolism" in result
        assert "excretion" in result
        assert "toxicity" in result

    def test_admet_absorption_fields(self, mol_tool: MoleculeTool) -> None:
        result = mol_tool.predict_admet(self.SMILES)
        absorption = result["absorption"]
        assert "oral_bioavailability" in absorption
        assert absorption["oral_bioavailability"] in {"HIGH", "LOW"}

    def test_suggest_modifications(self, mol_tool: MoleculeTool) -> None:
        result = mol_tool.suggest_modifications(self.SMILES, "solubility")
        assert "suggestions" in result
        assert len(result["suggestions"]) > 0
        assert "modification" in result["suggestions"][0]
        assert "rationale" in result["suggestions"][0]

    def test_compare_to_known_drugs(self, mol_tool: MoleculeTool) -> None:
        result = mol_tool.compare_to_known_drugs(self.SMILES)
        assert "comparisons" in result
        assert len(result["comparisons"]) > 0
        assert "most_similar" in result
        assert result["most_similar"] is not None

    def test_compare_filtered_by_class(self, mol_tool: MoleculeTool) -> None:
        result = mol_tool.compare_to_known_drugs(self.SMILES, drug_class="kinase_inhibitor")
        for comp in result["comparisons"]:
            assert comp["drug_class"] == "kinase_inhibitor"

    def test_simple_molecule(self, mol_tool: MoleculeTool) -> None:
        """Test with a very simple SMILES (ethanol)."""
        result = mol_tool.analyze("CCO")
        assert result["molecular_weight"] > 0
        assert result["hba"] >= 1  # Oxygen is an HBA


# ---------------------------------------------------------------------------
# Drug Database Tool
# ---------------------------------------------------------------------------


class TestDrugDatabaseTool:
    """Test the drug database lookup tool."""

    @pytest.fixture()
    def db(self) -> DrugDatabaseTool:
        return DrugDatabaseTool()

    def test_lookup_by_generic_name(self, db: DrugDatabaseTool) -> None:
        result = db.lookup_drug("imatinib")
        assert result is not None
        assert result["generic_name"] == "imatinib mesylate"
        assert result["brand_name"] == "Gleevec"

    def test_lookup_by_brand_name(self, db: DrugDatabaseTool) -> None:
        result = db.lookup_drug("Keytruda")
        assert result is not None
        assert result["generic_name"] == "pembrolizumab"

    def test_lookup_case_insensitive(self, db: DrugDatabaseTool) -> None:
        result = db.lookup_drug("IMATINIB")
        assert result is not None

    def test_lookup_not_found(self, db: DrugDatabaseTool) -> None:
        result = db.lookup_drug("nonexistent_drug_xyz")
        assert result is None

    def test_search_by_class(self, db: DrugDatabaseTool) -> None:
        results = db.search_by_class("biologic")
        assert len(results) >= 2
        assert all(r["drug_class"] == "biologic" for r in results)

    def test_search_by_therapeutic_area(self, db: DrugDatabaseTool) -> None:
        results = db.search_by_therapeutic_area("oncology")
        assert len(results) >= 2

    def test_search_by_mechanism(self, db: DrugDatabaseTool) -> None:
        results = db.search_by_mechanism("kinase")
        assert len(results) >= 1

    def test_get_all_drugs(self, db: DrugDatabaseTool) -> None:
        all_drugs = db.get_all_drugs()
        assert len(all_drugs) >= 5

    def test_get_drug_names(self, db: DrugDatabaseTool) -> None:
        names = db.get_drug_names()
        assert "imatinib" in names
        assert "pembrolizumab" in names

    def test_drug_structure_complete(self, db: DrugDatabaseTool) -> None:
        drug = db.lookup_drug("imatinib")
        assert drug is not None
        required_keys = {
            "generic_name",
            "brand_name",
            "drug_class",
            "mechanism",
            "therapeutic_area",
            "indications",
            "approval_year",
            "route",
            "molecular_weight",
            "half_life_hours",
        }
        assert required_keys.issubset(drug.keys())
