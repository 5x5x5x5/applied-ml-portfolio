"""Molecule generation pipeline with property optimization and diversity filtering.

Orchestrates the full generative workflow:
    1. Sample latent vectors from the prior or around a seed molecule
    2. Decode to SMILES via the VAE decoder
    3. Validate chemical structures
    4. Filter for desired property ranges (LogP, MW, QED, SA)
    5. Enforce structural diversity via Tanimoto distance thresholds
    6. Rank candidates by multi-objective desirability

Supports three generation modes:
    - Random: Pure sampling from the latent prior N(0, I)
    - Targeted: Constrained generation toward desired property profiles
    - Optimization: Gradient-free local search around an existing molecule
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from molecule_gen.chemistry.molecular_descriptors import (
    MolecularDescriptors,
    check_pains_alerts,
    compute_descriptors,
)
from molecule_gen.chemistry.smiles_processor import (
    SMILESVocabulary,
    canonicalize_smiles,
    compute_morgan_fingerprint,
    is_valid_smiles,
    tanimoto_similarity,
)
from molecule_gen.models.mol_vae import MolecularVAE

logger = logging.getLogger(__name__)


@dataclass
class PropertyConstraints:
    """Desired ranges for molecular properties during targeted generation.

    Each property can have a min/max bound. Set to None to leave unconstrained.
    """

    logp_min: float | None = None
    logp_max: float | None = 5.0
    mw_min: float | None = 150.0
    mw_max: float | None = 500.0
    qed_min: float | None = 0.3
    hbd_max: int | None = 5
    hba_max: int | None = 10
    tpsa_max: float | None = 140.0
    rotatable_bonds_max: int | None = 10
    sa_max: float | None = 6.0  # SA score <= 6 is reasonably synthesizable
    max_pains_alerts: int = 0  # Zero PAINS alerts by default


@dataclass
class GeneratedMolecule:
    """Container for a generated molecule with its computed properties."""

    smiles: str
    canonical_smiles: str | None = None
    descriptors: MolecularDescriptors | None = None
    pains_alerts: list[str] = field(default_factory=list)
    latent_vector: list[float] | None = None
    is_valid: bool = False
    is_novel: bool = True  # Not in training set
    passes_constraints: bool = False


class MoleculeGenerator:
    """Full molecule generation pipeline combining VAE with chemical filters.

    Integrates the molecular VAE with RDKit-based validation, property
    computation, diversity enforcement, and drug-likeness filtering to
    produce chemically valid and pharmacologically relevant novel molecules.
    """

    def __init__(
        self,
        model: MolecularVAE,
        vocab: SMILESVocabulary,
        device: torch.device | str = "cpu",
        known_smiles: set[str] | None = None,
    ) -> None:
        """Initialize the molecule generator.

        Args:
            model: Trained MolecularVAE model.
            vocab: SMILES vocabulary for encoding/decoding.
            device: Computation device (cpu or cuda).
            known_smiles: Set of training SMILES for novelty checking.
        """
        self.model = model
        self.vocab = vocab
        self.device = torch.device(device) if isinstance(device, str) else device
        self.known_smiles = known_smiles or set()
        self.model.to(self.device)
        self.model.eval()

    def generate_random(
        self,
        num_molecules: int = 100,
        temperature: float = 1.0,
        constraints: PropertyConstraints | None = None,
        max_attempts: int | None = None,
        diversity_threshold: float = 0.4,
    ) -> list[GeneratedMolecule]:
        """Generate molecules by random sampling from the latent prior.

        Samples z ~ N(0, temperature * I) and decodes to SMILES, then
        filters for validity, property constraints, and diversity.

        Args:
            num_molecules: Desired number of valid output molecules.
            temperature: Sampling temperature (higher = more diverse/risky).
            constraints: Property ranges to enforce.
            max_attempts: Max sampling attempts before giving up.
                          Defaults to 10 * num_molecules.
            diversity_threshold: Minimum Tanimoto distance between molecules.

        Returns:
            List of GeneratedMolecule objects that pass all filters.
        """
        if max_attempts is None:
            max_attempts = num_molecules * 10

        if constraints is None:
            constraints = PropertyConstraints()

        generated: list[GeneratedMolecule] = []
        generated_fps: list[Any] = []
        attempts = 0

        logger.info("Starting random generation: target=%d, temp=%.2f", num_molecules, temperature)

        while len(generated) < num_molecules and attempts < max_attempts:
            batch_size = min(64, max_attempts - attempts)
            attempts += batch_size

            with torch.no_grad():
                logits = self.model.sample(
                    num_samples=batch_size,
                    device=self.device,
                    temperature=temperature,
                )
                token_indices = logits.argmax(dim=-1)  # (batch, seq_len)

            for i in range(batch_size):
                indices = token_indices[i].cpu().tolist()
                smiles = self.vocab.decode_indices(indices)

                mol = self._process_candidate(smiles, constraints)
                if mol is None:
                    continue

                # Diversity check
                if not self._passes_diversity_filter(mol, generated_fps, diversity_threshold):
                    continue

                fp = compute_morgan_fingerprint(mol.canonical_smiles or smiles)
                if fp is not None:
                    generated_fps.append(fp)

                generated.append(mol)

                if len(generated) >= num_molecules:
                    break

        logger.info(
            "Generated %d valid molecules from %d attempts (%.1f%% success rate)",
            len(generated),
            attempts,
            100.0 * len(generated) / max(1, attempts),
        )

        return generated

    def generate_targeted(
        self,
        num_molecules: int = 50,
        constraints: PropertyConstraints | None = None,
        target_properties: dict[str, float] | None = None,
        num_latent_samples: int = 1000,
        temperature: float = 0.8,
    ) -> list[GeneratedMolecule]:
        """Generate molecules targeting specific property profiles.

        Uses rejection sampling with property scoring: generates a large
        pool of candidates and selects those closest to the target profile.

        Args:
            num_molecules: Number of molecules to return.
            constraints: Hard property constraints (pass/fail).
            target_properties: Soft targets to rank by proximity.
                               Keys: 'logp', 'mw', 'qed'.
            num_latent_samples: Size of candidate pool to generate.
            temperature: Sampling temperature.

        Returns:
            Top molecules ranked by proximity to target properties.
        """
        if constraints is None:
            constraints = PropertyConstraints()

        # Generate a large candidate pool
        candidates = self.generate_random(
            num_molecules=num_latent_samples,
            temperature=temperature,
            constraints=constraints,
            max_attempts=num_latent_samples * 5,
            diversity_threshold=0.3,
        )

        if not target_properties or not candidates:
            return candidates[:num_molecules]

        # Score candidates by proximity to target properties
        scored = []
        for mol in candidates:
            if mol.descriptors is None:
                continue
            score = self._compute_target_score(mol.descriptors, target_properties)
            scored.append((score, mol))

        # Sort by score (lower = closer to target)
        scored.sort(key=lambda x: x[0])

        return [mol for _, mol in scored[:num_molecules]]

    def optimize_molecule(
        self,
        seed_smiles: str,
        num_candidates: int = 50,
        search_radius: float = 1.0,
        constraints: PropertyConstraints | None = None,
        num_steps: int = 5,
        candidates_per_step: int = 100,
    ) -> list[GeneratedMolecule]:
        """Optimize an existing molecule by local search in latent space.

        Encodes the seed molecule to latent space, then generates candidates
        in a neighborhood defined by the search radius. Iteratively narrows
        the search around the best candidates (evolutionary strategy).

        This mimics medicinal chemistry lead optimization, where small
        structural modifications are made to improve a compound's profile.

        Args:
            seed_smiles: SMILES of the starting molecule.
            num_candidates: Number of optimized molecules to return.
            search_radius: Standard deviation of the Gaussian perturbation.
            constraints: Property constraints for filtering.
            num_steps: Number of iterative refinement steps.
            candidates_per_step: Candidates to evaluate per step.

        Returns:
            List of optimized molecules sorted by desirability.
        """
        if constraints is None:
            constraints = PropertyConstraints()

        # Encode seed molecule
        encoded = self.vocab.encode_smiles(seed_smiles)
        input_tensor = torch.tensor([encoded], dtype=torch.long, device=self.device)

        with torch.no_grad():
            mu, logvar = self.model.encode(input_tensor)
            z_seed = mu  # Use mean for deterministic starting point

        all_candidates: list[GeneratedMolecule] = []
        current_center = z_seed

        for step in range(num_steps):
            # Generate perturbations around current center
            noise = (
                torch.randn(
                    candidates_per_step,
                    self.model.config.latent_dim,
                    device=self.device,
                )
                * search_radius
            )

            z_candidates = current_center + noise

            with torch.no_grad():
                logits = self.model.decode(z_candidates)
                token_indices = logits.argmax(dim=-1)

            step_candidates: list[GeneratedMolecule] = []
            for i in range(candidates_per_step):
                indices = token_indices[i].cpu().tolist()
                smiles = self.vocab.decode_indices(indices)

                mol = self._process_candidate(smiles, constraints)
                if mol is not None:
                    step_candidates.append(mol)

            all_candidates.extend(step_candidates)

            # Narrow search around best candidates for next step
            if step_candidates and step < num_steps - 1:
                # Re-encode best candidates to update search center
                best = step_candidates[0]
                best_smiles = best.canonical_smiles or best.smiles
                best_encoded = self.vocab.encode_smiles(best_smiles)
                best_tensor = torch.tensor([best_encoded], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    mu_new, _ = self.model.encode(best_tensor)
                    current_center = mu_new
                search_radius *= 0.8  # Shrink search radius

        # Deduplicate and sort by overall drug-likeness
        unique = self._deduplicate(all_candidates)
        return unique[:num_candidates]

    def interpolate_molecules(
        self,
        smiles_start: str,
        smiles_end: str,
        num_steps: int = 10,
    ) -> list[GeneratedMolecule]:
        """Generate intermediate molecules by latent space interpolation.

        Encodes two molecules and linearly interpolates between their
        latent representations, decoding each intermediate point. This
        produces a "molecular morphing" trajectory showing how one
        chemical structure can smoothly transform into another.

        Args:
            smiles_start: Starting molecule SMILES.
            smiles_end: Ending molecule SMILES.
            num_steps: Number of interpolation points (including endpoints).

        Returns:
            List of GeneratedMolecule at each interpolation step.
        """
        # Encode both molecules
        enc_start = self.vocab.encode_smiles(smiles_start)
        enc_end = self.vocab.encode_smiles(smiles_end)

        t_start = torch.tensor([enc_start], dtype=torch.long, device=self.device)
        t_end = torch.tensor([enc_end], dtype=torch.long, device=self.device)

        with torch.no_grad():
            mu_start, _ = self.model.encode(t_start)
            mu_end, _ = self.model.encode(t_end)

            logits = self.model.interpolate(mu_start, mu_end, num_steps=num_steps)
            token_indices = logits.argmax(dim=-1)

        results: list[GeneratedMolecule] = []
        for i in range(num_steps):
            indices = token_indices[i].cpu().tolist()
            smiles = self.vocab.decode_indices(indices)
            mol = GeneratedMolecule(smiles=smiles)
            mol.canonical_smiles = canonicalize_smiles(smiles)
            mol.is_valid = is_valid_smiles(smiles)
            if mol.is_valid:
                mol.descriptors = compute_descriptors(mol.canonical_smiles or smiles)
            results.append(mol)

        return results

    def _process_candidate(
        self,
        smiles: str,
        constraints: PropertyConstraints,
    ) -> GeneratedMolecule | None:
        """Validate and score a candidate SMILES string.

        Returns None if the molecule fails validity or constraint checks.
        """
        # Skip empty or trivially short SMILES
        if not smiles or len(smiles) < 3:
            return None

        # Chemical validity
        if not is_valid_smiles(smiles):
            return None

        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            return None

        mol = GeneratedMolecule(
            smiles=smiles,
            canonical_smiles=canonical,
            is_valid=True,
        )

        # Novelty check
        mol.is_novel = canonical not in self.known_smiles

        # Compute descriptors
        mol.descriptors = compute_descriptors(canonical)

        # PAINS alerts
        mol.pains_alerts = check_pains_alerts(canonical)

        # Check property constraints
        mol.passes_constraints = self._check_constraints(mol, constraints)

        if not mol.passes_constraints:
            return None

        return mol

    def _check_constraints(
        self,
        mol: GeneratedMolecule,
        constraints: PropertyConstraints,
    ) -> bool:
        """Check whether a molecule satisfies all property constraints."""
        desc = mol.descriptors
        if desc is None:
            return False

        # LogP bounds
        if constraints.logp_min is not None and desc.logp is not None:
            if desc.logp < constraints.logp_min:
                return False
        if constraints.logp_max is not None and desc.logp is not None:
            if desc.logp > constraints.logp_max:
                return False

        # Molecular weight bounds
        if constraints.mw_min is not None and desc.molecular_weight is not None:
            if desc.molecular_weight < constraints.mw_min:
                return False
        if constraints.mw_max is not None and desc.molecular_weight is not None:
            if desc.molecular_weight > constraints.mw_max:
                return False

        # QED lower bound
        if constraints.qed_min is not None:
            # QED not computed in descriptors, so skip if unavailable
            pass

        # Hydrogen bond limits (Lipinski)
        if constraints.hbd_max is not None and desc.num_hbd is not None:
            if desc.num_hbd > constraints.hbd_max:
                return False
        if constraints.hba_max is not None and desc.num_hba is not None:
            if desc.num_hba > constraints.hba_max:
                return False

        # TPSA for oral absorption
        if constraints.tpsa_max is not None and desc.tpsa is not None:
            if desc.tpsa > constraints.tpsa_max:
                return False

        # Rotatable bonds (flexibility)
        if constraints.rotatable_bonds_max is not None and desc.num_rotatable_bonds is not None:
            if desc.num_rotatable_bonds > constraints.rotatable_bonds_max:
                return False

        # PAINS alerts
        if len(mol.pains_alerts) > constraints.max_pains_alerts:
            return False

        return True

    def _passes_diversity_filter(
        self,
        candidate: GeneratedMolecule,
        existing_fps: list[Any],
        threshold: float,
    ) -> bool:
        """Check that a candidate is sufficiently dissimilar to existing molecules.

        Uses Tanimoto distance (1 - Tanimoto similarity) to ensure structural
        diversity in the generated set.

        Args:
            candidate: Candidate molecule to check.
            existing_fps: List of Morgan fingerprints of already-accepted molecules.
            threshold: Minimum Tanimoto distance required.

        Returns:
            True if the candidate is sufficiently novel/diverse.
        """
        if not existing_fps:
            return True

        smiles = candidate.canonical_smiles or candidate.smiles
        fp = compute_morgan_fingerprint(smiles)
        if fp is None:
            return True  # Can't compute, allow through

        for existing_fp in existing_fps:
            similarity = tanimoto_similarity(fp, existing_fp)
            if similarity > (1.0 - threshold):
                return False

        return True

    def _compute_target_score(
        self,
        descriptors: MolecularDescriptors,
        targets: dict[str, float],
    ) -> float:
        """Compute distance from target property profile (lower is better).

        Uses normalized squared differences weighted equally across
        all specified target properties.
        """
        score = 0.0
        count = 0

        # Normalization constants (typical property ranges)
        normalizers = {
            "logp": 10.0,
            "mw": 500.0,
            "qed": 1.0,
            "tpsa": 200.0,
        }

        for prop, target_val in targets.items():
            actual = None
            if prop == "logp":
                actual = descriptors.logp
            elif prop == "mw":
                actual = descriptors.molecular_weight
            elif prop == "tpsa":
                actual = descriptors.tpsa

            if actual is not None:
                norm = normalizers.get(prop, 1.0)
                score += ((actual - target_val) / norm) ** 2
                count += 1

        return score / max(count, 1)

    def _deduplicate(self, molecules: list[GeneratedMolecule]) -> list[GeneratedMolecule]:
        """Remove duplicate molecules based on canonical SMILES."""
        seen: set[str] = set()
        unique: list[GeneratedMolecule] = []

        for mol in molecules:
            key = mol.canonical_smiles or mol.smiles
            if key not in seen:
                seen.add(key)
                unique.append(mol)

        return unique
