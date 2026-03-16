"""Molecular property prediction models for drug-likeness assessment.

Provides neural network-based predictors for key pharmacokinetic and
physicochemical properties used in drug design:

    - LogP (Crippen): Octanol-water partition coefficient measuring lipophilicity.
      Optimal range for oral drugs is typically 1-3 (Lipinski guideline < 5).

    - Molecular Weight: Total molecular weight in Daltons. Lipinski's guideline
      suggests MW < 500 Da for oral bioavailability.

    - QED (Quantitative Estimate of Drug-likeness): Bickerton et al. (2012)
      composite score integrating multiple desirability functions. Range [0, 1].

    - Synthetic Accessibility (SA): Ertl & Schuffenhauer (2009) score estimating
      ease of synthesis. Range [1, 10] where 1 = easy, 10 = difficult.

    - Lipinski's Rule of Five: Binary filter for oral bioavailability based on
      MW, LogP, HBD, and HBA thresholds.

These predictors can be used standalone for virtual screening or integrated
into the VAE's property-conditioned generation pipeline for targeted molecular
optimization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class PropertyPredictorConfig:
    """Configuration for molecular property predictor network."""

    input_dim: int = 128  # latent_dim from VAE or fingerprint dimension
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    num_properties: int = 4  # logP, MW, QED, SA
    dropout: float = 0.2
    use_batch_norm: bool = True


class MultiTaskPropertyPredictor(nn.Module):
    """Multi-task feedforward network for simultaneous molecular property prediction.

    Uses a shared trunk with property-specific heads to leverage correlations
    between related physicochemical descriptors. Shared representations learn
    general molecular features while specialized heads capture property-specific
    patterns.

    Predicted properties:
        0: LogP (lipophilicity, Crippen method)
        1: Molecular weight (Da)
        2: QED score (drug-likeness, 0-1)
        3: SA score (synthetic accessibility, 1-10)
    """

    PROPERTY_NAMES: list[str] = ["logP", "molecular_weight", "qed", "sa_score"]

    def __init__(self, config: PropertyPredictorConfig) -> None:
        super().__init__()
        self.config = config

        # Shared trunk
        trunk_layers: list[nn.Module] = []
        in_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            trunk_layers.append(nn.Linear(in_dim, hidden_dim))
            if config.use_batch_norm:
                trunk_layers.append(nn.BatchNorm1d(hidden_dim))
            trunk_layers.append(nn.ReLU())
            trunk_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim

        self.shared_trunk = nn.Sequential(*trunk_layers)

        # Property-specific prediction heads
        final_hidden = config.hidden_dims[-1]
        self.property_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(final_hidden, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                )
                for _ in range(config.num_properties)
            ]
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Predict molecular properties from latent vectors or fingerprints.

        Args:
            x: Input features of shape (batch, input_dim). Can be latent
               vectors from the VAE encoder or molecular fingerprints.

        Returns:
            Dictionary mapping property names to predicted values,
            each of shape (batch, 1).
        """
        shared_features = self.shared_trunk(x)

        predictions: dict[str, Tensor] = {}
        for i, (name, head) in enumerate(
            zip(self.PROPERTY_NAMES, self.property_heads, strict=True)
        ):
            predictions[name] = head(shared_features)

        return predictions

    def predict_all(self, x: Tensor) -> Tensor:
        """Predict all properties as a single tensor.

        Args:
            x: Input features of shape (batch, input_dim).

        Returns:
            Tensor of shape (batch, num_properties) with all predictions.
        """
        preds = self.forward(x)
        return torch.cat([preds[name] for name in self.PROPERTY_NAMES], dim=-1)


@dataclass
class LipinskiResult:
    """Results of Lipinski's Rule of Five assessment.

    A compound is considered drug-like if it satisfies at least 3 of the 4 rules.
    This is a critical first-pass filter in pharmaceutical lead optimization.
    """

    mw_pass: bool  # MW <= 500 Da
    logp_pass: bool  # LogP <= 5
    hbd_pass: bool  # Hydrogen bond donors <= 5
    hba_pass: bool  # Hydrogen bond acceptors <= 10
    num_violations: int  # Count of rule violations (0-4)
    is_drug_like: bool  # True if num_violations <= 1

    @property
    def passes(self) -> bool:
        """Whether the molecule passes the Ro5 filter (at most 1 violation)."""
        return self.is_drug_like


def apply_lipinski_filter(
    mw: float,
    logp: float,
    hbd: int,
    hba: int,
) -> LipinskiResult:
    """Apply Lipinski's Rule of Five to assess oral drug-likeness.

    Lipinski's rules (Lipinski et al., 1997) provide simple physicochemical
    guidelines for predicting oral bioavailability:

        1. Molecular weight <= 500 Da
        2. Calculated LogP <= 5
        3. Number of hydrogen bond donors (NH + OH) <= 5
        4. Number of hydrogen bond acceptors (N + O) <= 10

    A compound with more than one violation is unlikely to be orally bioavailable.

    Args:
        mw: Molecular weight in Daltons.
        logp: Calculated octanol-water partition coefficient.
        hbd: Count of hydrogen bond donor groups (OH, NH).
        hba: Count of hydrogen bond acceptor atoms (N, O).

    Returns:
        LipinskiResult with per-rule pass/fail and overall assessment.
    """
    mw_pass = mw <= 500.0
    logp_pass = logp <= 5.0
    hbd_pass = hbd <= 5
    hba_pass = hba <= 10

    violations = sum(not p for p in [mw_pass, logp_pass, hbd_pass, hba_pass])

    return LipinskiResult(
        mw_pass=mw_pass,
        logp_pass=logp_pass,
        hbd_pass=hbd_pass,
        hba_pass=hba_pass,
        num_violations=violations,
        is_drug_like=violations <= 1,
    )


def compute_qed_score(
    mw: float,
    logp: float,
    hba: int,
    hbd: int,
    psa: float,
    rotatable_bonds: int,
    num_aromatic_rings: int,
    num_alerts: int,
) -> float:
    """Compute Quantitative Estimate of Drug-likeness (QED).

    Implements a simplified version of the QED metric from Bickerton et al.
    (2012, Nature Chemistry). QED integrates individual desirability functions
    for eight molecular properties into a single composite score using the
    geometric mean of weighted desirability scores.

    Each desirability function is a Gaussian centered on the "ideal" value
    for oral drug candidates.

    Args:
        mw: Molecular weight in Daltons.
        logp: Calculated LogP.
        hba: Number of hydrogen bond acceptors.
        hbd: Number of hydrogen bond donors.
        psa: Polar surface area in Angstrom^2.
        rotatable_bonds: Count of freely rotatable bonds.
        num_aromatic_rings: Number of aromatic ring systems.
        num_alerts: Number of structural alerts (PAINS, toxicophores).

    Returns:
        QED score in [0, 1], where 1 = maximally drug-like.
    """
    # Desirability function parameters (mu, sigma, ideal range)
    # Approximated from the QED publication
    desirability_params: dict[str, tuple[float, float]] = {
        "mw": (300.0, 120.0),
        "logp": (2.5, 1.5),
        "hba": (4.0, 2.5),
        "hbd": (1.5, 1.5),
        "psa": (75.0, 40.0),
        "rotb": (3.0, 2.5),
        "arom": (2.0, 1.5),
        "alerts": (0.0, 0.5),
    }

    values = {
        "mw": mw,
        "logp": logp,
        "hba": float(hba),
        "hbd": float(hbd),
        "psa": psa,
        "rotb": float(rotatable_bonds),
        "arom": float(num_aromatic_rings),
        "alerts": float(num_alerts),
    }

    # QED weights from Bickerton et al. (mean weighting scheme)
    weights = {
        "mw": 0.66,
        "logp": 0.46,
        "hba": 0.05,
        "hbd": 0.26,
        "psa": 0.06,
        "rotb": 0.65,
        "arom": 0.48,
        "alerts": 0.95,
    }

    # Compute weighted desirability scores using Gaussian functions
    desirability_scores: list[float] = []
    for key in desirability_params:
        mu, sigma = desirability_params[key]
        x = values[key]
        w = weights[key]
        # Gaussian desirability: exp(-0.5 * ((x - mu) / sigma)^2)
        d = float(np.exp(-0.5 * ((x - mu) / sigma) ** 2))
        desirability_scores.append(d**w)

    # Geometric mean of weighted desirability scores
    if not desirability_scores:
        return 0.0

    product = 1.0
    for score in desirability_scores:
        product *= score

    qed = product ** (1.0 / len(desirability_scores))
    return float(np.clip(qed, 0.0, 1.0))


def compute_sa_score_estimate(
    ring_count: int,
    stereo_centers: int,
    sp3_fraction: float,
    num_bridged_atoms: int = 0,
    macrocycle: bool = False,
) -> float:
    """Estimate synthetic accessibility from molecular topology features.

    Simplified proxy for the full Ertl & Schuffenhauer SA score. The full
    algorithm uses fragment contributions from a trained model, but this
    heuristic captures the main structural complexity factors:

        - Ring system complexity (spiro, fused, bridged)
        - Stereocenters (each adds synthetic difficulty)
        - sp3 fraction (higher = harder, but also more 3D drug-like)
        - Macrocycles (>= 12-membered rings are synthetically challenging)

    Args:
        ring_count: Total number of rings in the molecule.
        stereo_centers: Number of defined stereocenters.
        sp3_fraction: Fraction of sp3-hybridized carbons.
        num_bridged_atoms: Number of bridgehead atoms.
        macrocycle: Whether the molecule contains a macrocyclic ring.

    Returns:
        SA score estimate in [1, 10] where 1 = easy, 10 = hard.
    """
    score = 1.0

    # Ring complexity contribution
    score += min(ring_count * 0.5, 3.0)

    # Stereochemistry burden
    score += stereo_centers * 0.4

    # sp3 carbon fraction (moderate penalty for high values)
    score += sp3_fraction * 1.5

    # Bridged systems are synthetically challenging
    score += num_bridged_atoms * 0.8

    # Macrocycles are notoriously hard to synthesize
    if macrocycle:
        score += 2.0

    return float(np.clip(score, 1.0, 10.0))


class PropertyLoss(nn.Module):
    """Multi-task loss for property prediction with per-property weighting.

    Supports heteroscedastic uncertainty weighting (Kendall et al., 2018)
    where the loss for each property is weighted by a learned log-variance,
    automatically balancing the multi-task objective.
    """

    def __init__(
        self,
        num_properties: int = 4,
        use_uncertainty_weighting: bool = True,
    ) -> None:
        super().__init__()
        self.use_uncertainty_weighting = use_uncertainty_weighting

        if use_uncertainty_weighting:
            # Learnable log-variance parameters (one per task)
            self.log_vars = nn.Parameter(torch.zeros(num_properties))

    def forward(
        self,
        predictions: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute weighted multi-task property prediction loss.

        Args:
            predictions: Dict of predicted values per property, each (batch, 1).
            targets: Dict of target values per property, each (batch, 1).

        Returns:
            Tuple of (total_loss, per_property_losses).
        """
        property_names = list(predictions.keys())
        per_property_losses: dict[str, Tensor] = {}
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)

        for i, name in enumerate(property_names):
            mse = nn.functional.mse_loss(predictions[name], targets[name])
            per_property_losses[name] = mse

            if self.use_uncertainty_weighting:
                # Homoscedastic uncertainty weighting
                precision = torch.exp(-self.log_vars[i])
                total_loss = total_loss + precision * mse + self.log_vars[i]
            else:
                total_loss = total_loss + mse

        return total_loss, per_property_losses
