"""FastAPI application for ProteinExplorer.

Provides REST API endpoints for protein sequence analysis,
pairwise alignment, and structure feature prediction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from protein_explorer.analysis.alignment import (
    format_alignment,
    needleman_wunsch,
    optimize_gap_penalties,
)
from protein_explorer.analysis.sequence_analyzer import (
    analyze_sequence,
    predict_secondary_structure,
    validate_sequence,
)
from protein_explorer.analysis.structure_predictor import (
    detect_domain_boundaries,
    predict_contact_map,
    predict_disorder,
    predict_solvent_accessibility,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ProteinExplorer API",
    description="Interactive Protein Structure Analysis Tool",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# --- Request/Response models ---


class SequenceRequest(BaseModel):
    """Request body for sequence analysis."""

    sequence: str = Field(..., min_length=1, description="Protein sequence (single-letter codes)")


class AlignmentRequest(BaseModel):
    """Request body for pairwise alignment."""

    sequence1: str = Field(..., min_length=1, description="First protein sequence")
    sequence2: str = Field(..., min_length=1, description="Second protein sequence")
    gap_open: float = Field(default=-10.0, description="Gap opening penalty")
    gap_extend: float = Field(default=-0.5, description="Gap extension penalty")
    optimize_gaps: bool = Field(default=False, description="Auto-optimize gap penalties")


class StructurePredictionRequest(BaseModel):
    """Request body for structure prediction."""

    sequence: str = Field(..., min_length=1, description="Protein sequence")
    include_contact_map: bool = Field(default=True, description="Include contact map prediction")
    include_accessibility: bool = Field(default=True, description="Include solvent accessibility")
    include_disorder: bool = Field(default=True, description="Include disorder prediction")
    include_domains: bool = Field(default=True, description="Include domain detection")


class AnalysisResponse(BaseModel):
    """Response for sequence analysis."""

    sequence: str
    length: int
    amino_acid_composition: dict[str, float]
    amino_acid_counts: dict[str, int]
    molecular_weight: float
    isoelectric_point: float
    hydrophobicity_profile: list[float]
    secondary_structure: list[str]
    secondary_structure_summary: dict[str, float]
    has_signal_peptide: bool
    signal_peptide_length: int
    disulfide_bonds: list[list[int]]
    charge_at_ph7: float
    gravy: float
    aromaticity: float
    instability_index: float
    is_stable: bool


class AlignmentResponse(BaseModel):
    """Response for alignment."""

    aligned_seq1: str
    aligned_seq2: str
    score: float
    identity: float
    similarity: float
    gaps: int
    gap_opens: int
    alignment_length: int
    midline: str
    formatted: str
    gap_open_used: float
    gap_extend_used: float


class StructurePredictionResponse(BaseModel):
    """Response for structure prediction."""

    sequence: str
    length: int
    contact_map: dict[str, Any] | None = None
    solvent_accessibility: dict[str, Any] | None = None
    disorder: dict[str, Any] | None = None
    domains: dict[str, Any] | None = None


class UniProtResponse(BaseModel):
    """Response for UniProt protein lookup."""

    uniprot_id: str
    name: str
    organism: str
    sequence: str
    length: int
    function: str
    gene_name: str


# --- Mock UniProt data ---

MOCK_UNIPROT: dict[str, dict[str, str]] = {
    "P69905": {
        "name": "Hemoglobin subunit alpha",
        "organism": "Homo sapiens",
        "sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
        "GSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKL"
        "LSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
        "function": "Oxygen transport from the lung to peripheral tissues",
        "gene_name": "HBA1",
    },
    "P68871": {
        "name": "Hemoglobin subunit beta",
        "organism": "Homo sapiens",
        "sequence": "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLST"
        "PDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDP"
        "ENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH",
        "function": "Oxygen transport from the lung to peripheral tissues",
        "gene_name": "HBB",
    },
    "P00533": {
        "name": "Epidermal growth factor receptor",
        "organism": "Homo sapiens",
        "sequence": "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSL"
        "QRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQI"
        "IRGKLFHPGG",
        "function": "Receptor tyrosine kinase binding EGF family ligands",
        "gene_name": "EGFR",
    },
    "P01308": {
        "name": "Insulin",
        "organism": "Homo sapiens",
        "sequence": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYT"
        "PKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSL"
        "YQLENYCN",
        "function": "Regulation of glucose metabolism",
        "gene_name": "INS",
    },
}


# --- Endpoints ---


@app.get("/")
async def root() -> FileResponse:
    """Serve the frontend dashboard."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    raise HTTPException(status_code=404, detail="Frontend not found")


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_protein(request: SequenceRequest) -> AnalysisResponse:
    """Perform full sequence analysis.

    Returns amino acid composition, molecular weight, pI,
    hydrophobicity profile, secondary structure prediction, and more.
    """
    try:
        result = analyze_sequence(request.sequence)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    ss_pred = predict_secondary_structure(result.sequence)

    return AnalysisResponse(
        sequence=result.sequence,
        length=result.length,
        amino_acid_composition=result.amino_acid_composition,
        amino_acid_counts=result.amino_acid_counts,
        molecular_weight=result.molecular_weight,
        isoelectric_point=result.isoelectric_point,
        hydrophobicity_profile=result.hydrophobicity_profile,
        secondary_structure=result.secondary_structure,
        secondary_structure_summary={
            "helix": ss_pred.helix_fraction,
            "sheet": ss_pred.sheet_fraction,
            "coil": ss_pred.coil_fraction,
        },
        has_signal_peptide=result.has_signal_peptide,
        signal_peptide_length=result.signal_peptide_length,
        disulfide_bonds=[list(bond) for bond in result.disulfide_bonds],
        charge_at_ph7=result.charge_at_ph7,
        gravy=result.gravy,
        aromaticity=result.aromaticity,
        instability_index=result.instability_index,
        is_stable=result.instability_index < 40,
    )


@app.post("/align", response_model=AlignmentResponse)
async def align_sequences(request: AlignmentRequest) -> AlignmentResponse:
    """Perform pairwise sequence alignment using Needleman-Wunsch."""
    try:
        gap_open = request.gap_open
        gap_extend = request.gap_extend

        if request.optimize_gaps:
            gap_open, gap_extend, _ = optimize_gap_penalties(request.sequence1, request.sequence2)

        result = needleman_wunsch(
            request.sequence1,
            request.sequence2,
            gap_open=gap_open,
            gap_extend=gap_extend,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    formatted = format_alignment(result)

    return AlignmentResponse(
        aligned_seq1=result.aligned_seq1,
        aligned_seq2=result.aligned_seq2,
        score=result.score,
        identity=result.identity,
        similarity=result.similarity,
        gaps=result.gaps,
        gap_opens=result.gap_opens,
        alignment_length=result.alignment_length,
        midline=result.midline,
        formatted=formatted,
        gap_open_used=gap_open,
        gap_extend_used=gap_extend,
    )


@app.post("/predict-structure", response_model=StructurePredictionResponse)
async def predict_structure(
    request: StructurePredictionRequest,
) -> StructurePredictionResponse:
    """Predict structural features from sequence."""
    try:
        seq = validate_sequence(request.sequence)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    response: dict[str, Any] = {
        "sequence": seq,
        "length": len(seq),
    }

    if request.include_contact_map:
        cm = predict_contact_map(seq)
        response["contact_map"] = {
            "predicted_contacts": [
                {"residue_i": c[0], "residue_j": c[1], "probability": c[2]}
                for c in cm.predicted_contacts[:50]  # Limit for response size
            ],
            "num_contacts": len(cm.predicted_contacts),
        }

    if request.include_accessibility:
        sa = predict_solvent_accessibility(seq)
        response["solvent_accessibility"] = {
            "accessibility": sa.accessibility,
            "burial_state": sa.burial_state,
            "fraction_buried": sa.fraction_buried,
            "fraction_exposed": sa.fraction_exposed,
        }

    if request.include_disorder:
        dis = predict_disorder(seq)
        response["disorder"] = {
            "scores": dis.disorder_scores,
            "disordered_regions": [{"start": r[0], "end": r[1]} for r in dis.disordered_regions],
            "fraction_disordered": dis.fraction_disordered,
        }

    if request.include_domains:
        dom = detect_domain_boundaries(seq)
        response["domains"] = {
            "num_domains": dom.num_domains,
            "boundaries": dom.domain_boundaries,
            "domains": [{"start": d[0], "end": d[1]} for d in dom.domains],
            "boundary_scores": dom.boundary_scores,
        }

    return StructurePredictionResponse(**response)


@app.get("/protein/{uniprot_id}", response_model=UniProtResponse)
async def get_protein(uniprot_id: str) -> UniProtResponse:
    """Fetch protein data by UniProt ID (mock data)."""
    uid = uniprot_id.upper()

    if uid not in MOCK_UNIPROT:
        raise HTTPException(
            status_code=404,
            detail=f"Protein {uid} not found. Available: {', '.join(MOCK_UNIPROT.keys())}",
        )

    data = MOCK_UNIPROT[uid]
    return UniProtResponse(
        uniprot_id=uid,
        name=data["name"],
        organism=data["organism"],
        sequence=data["sequence"],
        length=len(data["sequence"]),
        function=data["function"],
        gene_name=data["gene_name"],
    )


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "ProteinExplorer"}
