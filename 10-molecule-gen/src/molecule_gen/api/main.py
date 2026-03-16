"""FastAPI service for AI-powered drug molecule generation.

Exposes the MoleculeGen pipeline as REST endpoints for:
    - Generating novel drug-like molecules with property constraints
    - Optimizing existing molecules via latent space search
    - Predicting physicochemical and drug-likeness properties
    - Visualizing molecular structures as SVG images
"""

from __future__ import annotations

import logging

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from molecule_gen.chemistry.molecular_descriptors import (
    check_pains_alerts,
    compute_descriptors,
)
from molecule_gen.chemistry.smiles_processor import (
    SMILESVocabulary,
    canonicalize_smiles,
    is_valid_smiles,
)
from molecule_gen.generation.generator import (
    GeneratedMolecule,
    MoleculeGenerator,
    PropertyConstraints,
)
from molecule_gen.models.mol_vae import MolecularVAE, VAEConfig

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MoleculeGen API",
    description=(
        "AI-powered Drug Molecule Generator using Variational Autoencoders. "
        "Generate, optimize, and analyze novel drug-like molecules with "
        "property-conditioned latent space sampling."
    ),
    version="0.1.0",
)


# ---- Request/Response Schemas ----


class GenerateRequest(BaseModel):
    """Request body for molecule generation endpoint."""

    num_molecules: int = Field(
        default=10, ge=1, le=1000, description="Number of molecules to generate"
    )
    temperature: float = Field(
        default=1.0, ge=0.1, le=3.0, description="Sampling temperature (higher = more diverse)"
    )
    logp_min: float | None = Field(default=None, description="Minimum LogP (lipophilicity)")
    logp_max: float | None = Field(default=5.0, description="Maximum LogP")
    mw_min: float | None = Field(default=150.0, description="Minimum molecular weight (Da)")
    mw_max: float | None = Field(default=500.0, description="Maximum molecular weight (Da)")
    qed_min: float | None = Field(default=0.3, description="Minimum QED drug-likeness score")
    max_pains_alerts: int = Field(default=0, description="Maximum allowed PAINS alerts")
    diversity_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Minimum Tanimoto distance between molecules"
    )


class OptimizeRequest(BaseModel):
    """Request body for molecule optimization endpoint."""

    smiles: str = Field(description="SMILES of the seed molecule to optimize")
    num_candidates: int = Field(default=20, ge=1, le=500)
    search_radius: float = Field(
        default=1.0, ge=0.1, le=5.0, description="Latent space search radius"
    )
    num_steps: int = Field(default=5, ge=1, le=20, description="Iterative refinement steps")
    logp_max: float | None = Field(default=5.0)
    mw_max: float | None = Field(default=500.0)


class PredictPropertiesRequest(BaseModel):
    """Request body for property prediction endpoint."""

    smiles: str = Field(description="SMILES string of the molecule")


class MoleculeResponse(BaseModel):
    """Response schema for a single generated molecule."""

    smiles: str
    canonical_smiles: str | None = None
    is_valid: bool = False
    is_novel: bool = True
    molecular_weight: float | None = None
    logp: float | None = None
    tpsa: float | None = None
    num_hbd: int | None = None
    num_hba: int | None = None
    num_rotatable_bonds: int | None = None
    num_rings: int | None = None
    num_aromatic_rings: int | None = None
    fraction_sp3: float | None = None
    lipinski_violations: int | None = None
    is_lipinski_compliant: bool | None = None
    pains_alerts: list[str] = []

    @classmethod
    def from_generated(cls, mol: GeneratedMolecule) -> MoleculeResponse:
        """Construct response from a GeneratedMolecule object."""
        desc = mol.descriptors
        return cls(
            smiles=mol.smiles,
            canonical_smiles=mol.canonical_smiles,
            is_valid=mol.is_valid,
            is_novel=mol.is_novel,
            molecular_weight=desc.molecular_weight if desc else None,
            logp=desc.logp if desc else None,
            tpsa=desc.tpsa if desc else None,
            num_hbd=desc.num_hbd if desc else None,
            num_hba=desc.num_hba if desc else None,
            num_rotatable_bonds=desc.num_rotatable_bonds if desc else None,
            num_rings=desc.num_rings if desc else None,
            num_aromatic_rings=desc.num_aromatic_rings if desc else None,
            fraction_sp3=desc.fraction_sp3 if desc else None,
            lipinski_violations=desc.lipinski_violations if desc else None,
            is_lipinski_compliant=desc.is_lipinski_compliant if desc else None,
            pains_alerts=mol.pains_alerts,
        )


class GenerateResponse(BaseModel):
    """Response schema for molecule generation."""

    molecules: list[MoleculeResponse]
    num_generated: int
    num_valid: int
    validity_rate: float


class PropertyResponse(BaseModel):
    """Response schema for property prediction."""

    smiles: str
    canonical_smiles: str | None = None
    is_valid: bool
    molecular_weight: float | None = None
    logp: float | None = None
    tpsa: float | None = None
    num_hbd: int | None = None
    num_hba: int | None = None
    num_rotatable_bonds: int | None = None
    num_rings: int | None = None
    num_aromatic_rings: int | None = None
    num_heteroatoms: int | None = None
    fraction_sp3: float | None = None
    bertz_complexity: float | None = None
    lipinski_violations: int | None = None
    is_lipinski_compliant: bool | None = None
    veber_compliant: bool | None = None
    ghose_compliant: bool | None = None
    is_lead_like: bool | None = None
    pains_alerts: list[str] = []


# ---- Global model state (loaded on startup) ----

_generator: MoleculeGenerator | None = None


def _get_generator() -> MoleculeGenerator:
    """Get or lazily initialize the molecule generator.

    In production, the model would be loaded from a checkpoint. Here we
    initialize with random weights as a demonstration scaffold.
    """
    global _generator

    if _generator is None:
        logger.info("Initializing MoleculeGen model (untrained demo weights)")
        config = VAEConfig(
            vocab_size=SMILESVocabulary().size,
            latent_dim=128,
            max_sequence_length=120,
        )
        model = MolecularVAE(config)
        vocab = SMILESVocabulary()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _generator = MoleculeGenerator(model=model, vocab=vocab, device=device)

    return _generator


# ---- Endpoints ----


@app.post("/generate", response_model=GenerateResponse)
async def generate_molecules(request: GenerateRequest) -> GenerateResponse:
    """Generate novel drug-like molecules with desired properties.

    Samples from the VAE latent space and filters for validity, property
    constraints, diversity, and PAINS alerts. Returns molecules with
    computed physicochemical descriptors.
    """
    generator = _get_generator()

    constraints = PropertyConstraints(
        logp_min=request.logp_min,
        logp_max=request.logp_max,
        mw_min=request.mw_min,
        mw_max=request.mw_max,
        qed_min=request.qed_min,
        max_pains_alerts=request.max_pains_alerts,
    )

    molecules = generator.generate_random(
        num_molecules=request.num_molecules,
        temperature=request.temperature,
        constraints=constraints,
        diversity_threshold=request.diversity_threshold,
    )

    mol_responses = [MoleculeResponse.from_generated(m) for m in molecules]
    num_valid = sum(1 for m in mol_responses if m.is_valid)

    return GenerateResponse(
        molecules=mol_responses,
        num_generated=len(mol_responses),
        num_valid=num_valid,
        validity_rate=num_valid / max(len(mol_responses), 1),
    )


@app.post("/optimize", response_model=GenerateResponse)
async def optimize_molecule(request: OptimizeRequest) -> GenerateResponse:
    """Optimize an existing molecule by local latent space search.

    Takes a seed molecule SMILES, encodes it to the VAE latent space,
    and generates structurally similar candidates with improved properties.
    Mimics medicinal chemistry lead optimization.
    """
    if not is_valid_smiles(request.smiles):
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {request.smiles}")

    generator = _get_generator()

    constraints = PropertyConstraints(
        logp_max=request.logp_max,
        mw_max=request.mw_max,
    )

    molecules = generator.optimize_molecule(
        seed_smiles=request.smiles,
        num_candidates=request.num_candidates,
        search_radius=request.search_radius,
        constraints=constraints,
        num_steps=request.num_steps,
    )

    mol_responses = [MoleculeResponse.from_generated(m) for m in molecules]
    num_valid = sum(1 for m in mol_responses if m.is_valid)

    return GenerateResponse(
        molecules=mol_responses,
        num_generated=len(mol_responses),
        num_valid=num_valid,
        validity_rate=num_valid / max(len(mol_responses), 1),
    )


@app.post("/predict-properties", response_model=PropertyResponse)
async def predict_properties(request: PredictPropertiesRequest) -> PropertyResponse:
    """Predict physicochemical properties and drug-likeness of a molecule.

    Computes a comprehensive panel of molecular descriptors including:
    LogP, TPSA, H-bond donors/acceptors, rotatable bonds, Lipinski Ro5,
    Veber rules, Ghose filter, lead-likeness, and PAINS alerts.
    """
    valid = is_valid_smiles(request.smiles)
    canonical = canonicalize_smiles(request.smiles) if valid else None

    if not valid:
        return PropertyResponse(
            smiles=request.smiles,
            is_valid=False,
        )

    desc = compute_descriptors(canonical or request.smiles)
    alerts = check_pains_alerts(canonical or request.smiles)

    return PropertyResponse(
        smiles=request.smiles,
        canonical_smiles=desc.canonical_smiles,
        is_valid=True,
        molecular_weight=desc.molecular_weight,
        logp=desc.logp,
        tpsa=desc.tpsa,
        num_hbd=desc.num_hbd,
        num_hba=desc.num_hba,
        num_rotatable_bonds=desc.num_rotatable_bonds,
        num_rings=desc.num_rings,
        num_aromatic_rings=desc.num_aromatic_rings,
        num_heteroatoms=desc.num_heteroatoms,
        fraction_sp3=desc.fraction_sp3,
        bertz_complexity=desc.bertz_complexity,
        lipinski_violations=desc.lipinski_violations,
        is_lipinski_compliant=desc.is_lipinski_compliant,
        veber_compliant=desc.veber_compliant,
        ghose_compliant=desc.ghose_compliant,
        is_lead_like=desc.is_lead_like,
        pains_alerts=alerts,
    )


@app.get("/molecule/{smiles}/visualize")
async def visualize_molecule(
    smiles: str,
    width: int = Query(default=400, ge=100, le=1200),
    height: int = Query(default=300, ge=100, le=1200),
) -> Response:
    """Render a 2D depiction of the molecule as an SVG image.

    Uses RDKit's 2D coordinate generation and SVG rendering to produce
    a publication-quality structure diagram.

    Args:
        smiles: URL-encoded SMILES string of the molecule.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        SVG image response.
    """
    if not is_valid_smiles(smiles):
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {smiles}")

    try:
        from rdkit import Chem
        from rdkit.Chem import Draw

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise HTTPException(status_code=400, detail="Could not parse molecule")

        # Generate 2D coordinates
        from rdkit.Chem import AllChem

        AllChem.Compute2DCoords(mol)

        # Render to SVG
        drawer = Draw.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg_text = drawer.GetDrawingText()

        return Response(content=svg_text, media_type="image/svg+xml")

    except ImportError:
        # Fallback: return a simple SVG placeholder
        svg_placeholder = _generate_placeholder_svg(smiles, width, height)
        return Response(content=svg_placeholder, media_type="image/svg+xml")


def _generate_placeholder_svg(smiles: str, width: int, height: int) -> str:
    """Generate a placeholder SVG when RDKit drawing is unavailable."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect width="{width}" height="{height}" fill="#f8f9fa" stroke="#dee2e6" rx="8"/>
  <text x="{width // 2}" y="{height // 2 - 15}" text-anchor="middle"
        font-family="monospace" font-size="14" fill="#495057">
    {smiles[:60]}
  </text>
  <text x="{width // 2}" y="{height // 2 + 15}" text-anchor="middle"
        font-family="sans-serif" font-size="11" fill="#868e96">
    (RDKit required for structure rendering)
  </text>
</svg>"""


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}
