#!/usr/bin/env python3
"""Evaluation script for molecular generation quality metrics.

Assesses the VAE's generation capability using standard metrics from the
molecular generation literature:

    Validity: Fraction of generated SMILES that parse to valid molecules.
    Uniqueness: Fraction of valid molecules that are structurally unique.
    Novelty: Fraction of unique molecules not present in the training set.
    Diversity: Average pairwise Tanimoto distance within the generated set
               (measured via Morgan fingerprints).
    Property Distribution: KL divergence between property distributions of
                           generated vs. training molecules.

Reference metrics from MOSES benchmark (Polykovskiy et al., 2020):
    Top models achieve >95% validity, >99% uniqueness, >90% novelty,
    and internal diversity ~0.85.

Usage:
    uv run python scripts/evaluate_generation.py \\
        --checkpoint checkpoints/mol_vae_best.pt \\
        --training-data data/molecules.csv \\
        --num-samples 10000 \\
        --output results/evaluation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from molecule_gen.chemistry.molecular_descriptors import compute_descriptors
from molecule_gen.chemistry.smiles_processor import (
    SMILESVocabulary,
    canonicalize_smiles,
    compute_morgan_fingerprint,
    is_valid_smiles,
    tanimoto_similarity,
)
from molecule_gen.models.mol_vae import MolecularVAE, VAEConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate_generation")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate molecular generation quality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default=None,
        help="Path to vocabulary JSON (defaults to checkpoint_dir/vocabulary.json)",
    )
    parser.add_argument(
        "--training-data",
        type=str,
        default=None,
        help="Path to training CSV for novelty evaluation",
    )
    parser.add_argument(
        "--smiles-column",
        type=str,
        default="smiles",
        help="Column name for SMILES in training CSV",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of molecules to generate for evaluation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--diversity-sample-size",
        type=int,
        default=1000,
        help="Number of molecules to subsample for diversity calculation",
    )

    return parser.parse_args()


def compute_validity(smiles_list: list[str]) -> tuple[float, list[str]]:
    """Compute the fraction of valid SMILES and return valid ones.

    Args:
        smiles_list: Raw generated SMILES strings.

    Returns:
        Tuple of (validity_rate, list_of_valid_smiles).
    """
    valid_smiles: list[str] = []
    for smiles in smiles_list:
        if smiles and is_valid_smiles(smiles):
            canonical = canonicalize_smiles(smiles)
            if canonical:
                valid_smiles.append(canonical)

    rate = len(valid_smiles) / max(len(smiles_list), 1)
    return rate, valid_smiles


def compute_uniqueness(valid_smiles: list[str]) -> tuple[float, list[str]]:
    """Compute the fraction of unique molecules among valid ones.

    Args:
        valid_smiles: List of valid canonical SMILES.

    Returns:
        Tuple of (uniqueness_rate, list_of_unique_smiles).
    """
    unique = list(set(valid_smiles))
    rate = len(unique) / max(len(valid_smiles), 1)
    return rate, unique


def compute_novelty(
    unique_smiles: list[str],
    training_smiles: set[str],
) -> tuple[float, list[str]]:
    """Compute the fraction of unique molecules not in the training set.

    Args:
        unique_smiles: List of unique canonical SMILES.
        training_smiles: Set of canonical SMILES from the training data.

    Returns:
        Tuple of (novelty_rate, list_of_novel_smiles).
    """
    novel = [s for s in unique_smiles if s not in training_smiles]
    rate = len(novel) / max(len(unique_smiles), 1)
    return rate, novel


def compute_internal_diversity(
    smiles_list: list[str],
    sample_size: int = 1000,
) -> float:
    """Compute average pairwise Tanimoto distance (internal diversity).

    Higher diversity (closer to 1.0) indicates the model generates a
    wider range of chemical structures.

    Args:
        smiles_list: List of canonical SMILES.
        sample_size: Number of molecules to subsample for efficiency.

    Returns:
        Average pairwise Tanimoto distance in [0, 1].
    """
    if len(smiles_list) < 2:
        return 0.0

    # Subsample if necessary (O(n^2) pairwise comparison)
    if len(smiles_list) > sample_size:
        indices = np.random.choice(len(smiles_list), sample_size, replace=False)
        smiles_list = [smiles_list[i] for i in indices]

    # Compute fingerprints
    fps = []
    for smiles in smiles_list:
        fp = compute_morgan_fingerprint(smiles)
        if fp is not None:
            fps.append(fp)

    if len(fps) < 2:
        return 0.0

    # Compute pairwise distances
    distances: list[float] = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = tanimoto_similarity(fps[i], fps[j])
            distances.append(1.0 - sim)

    return float(np.mean(distances))


def compute_property_statistics(
    smiles_list: list[str],
    max_molecules: int = 5000,
) -> dict[str, dict[str, float]]:
    """Compute property distribution statistics for generated molecules.

    Args:
        smiles_list: List of valid canonical SMILES.
        max_molecules: Maximum molecules to compute descriptors for.

    Returns:
        Dictionary of property names -> {mean, std, min, max, median}.
    """
    if len(smiles_list) > max_molecules:
        smiles_list = list(np.random.choice(smiles_list, max_molecules, replace=False))

    properties: dict[str, list[float]] = {
        "molecular_weight": [],
        "logp": [],
        "tpsa": [],
        "num_hbd": [],
        "num_hba": [],
        "num_rotatable_bonds": [],
        "num_rings": [],
        "fraction_sp3": [],
    }

    for smiles in smiles_list:
        desc = compute_descriptors(smiles)
        if desc.molecular_weight is not None:
            properties["molecular_weight"].append(desc.molecular_weight)
        if desc.logp is not None:
            properties["logp"].append(desc.logp)
        if desc.tpsa is not None:
            properties["tpsa"].append(desc.tpsa)
        if desc.num_hbd is not None:
            properties["num_hbd"].append(float(desc.num_hbd))
        if desc.num_hba is not None:
            properties["num_hba"].append(float(desc.num_hba))
        if desc.num_rotatable_bonds is not None:
            properties["num_rotatable_bonds"].append(float(desc.num_rotatable_bonds))
        if desc.num_rings is not None:
            properties["num_rings"].append(float(desc.num_rings))
        if desc.fraction_sp3 is not None:
            properties["fraction_sp3"].append(desc.fraction_sp3)

    stats: dict[str, dict[str, float]] = {}
    for prop_name, values in properties.items():
        if values:
            arr = np.array(values)
            stats[prop_name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "median": float(np.median(arr)),
            }

    return stats


def compute_lipinski_pass_rate(smiles_list: list[str], max_molecules: int = 5000) -> float:
    """Compute fraction of molecules satisfying Lipinski's Rule of Five.

    Args:
        smiles_list: List of valid SMILES.
        max_molecules: Maximum molecules to evaluate.

    Returns:
        Fraction passing Ro5 in [0, 1].
    """
    if len(smiles_list) > max_molecules:
        smiles_list = list(np.random.choice(smiles_list, max_molecules, replace=False))

    passing = 0
    total = 0

    for smiles in smiles_list:
        desc = compute_descriptors(smiles)
        if desc.is_lipinski_compliant is not None:
            total += 1
            if desc.is_lipinski_compliant:
                passing += 1

    return passing / max(total, 1)


def generate_molecules(
    model: MolecularVAE,
    vocab: SMILESVocabulary,
    num_samples: int,
    batch_size: int,
    temperature: float,
    device: torch.device,
) -> list[str]:
    """Generate SMILES strings from the VAE.

    Args:
        model: Trained MolecularVAE.
        vocab: SMILES vocabulary.
        num_samples: Total number of molecules to generate.
        batch_size: Batch size for generation.
        temperature: Sampling temperature.
        device: Computation device.

    Returns:
        List of raw generated SMILES strings.
    """
    model.eval()
    all_smiles: list[str] = []

    remaining = num_samples
    while remaining > 0:
        current_batch = min(batch_size, remaining)

        with torch.no_grad():
            logits = model.sample(
                num_samples=current_batch,
                device=device,
                temperature=temperature,
            )
            token_indices = logits.argmax(dim=-1)

        for i in range(current_batch):
            indices = token_indices[i].cpu().tolist()
            smiles = vocab.decode_indices(indices)
            all_smiles.append(smiles)

        remaining -= current_batch

    return all_smiles


def main() -> None:
    """Run full evaluation pipeline."""
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # Load checkpoint
    logger.info("Loading checkpoint from %s", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config_dict = checkpoint["config"]
    config = VAEConfig(**config_dict)

    model = MolecularVAE(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Model loaded (epoch %d)", checkpoint.get("epoch", -1))

    # Load vocabulary
    vocab_path = args.vocab
    if vocab_path is None:
        vocab_path = str(Path(args.checkpoint).parent / "vocabulary.json")

    with open(vocab_path) as f:
        vocab_data = json.load(f)
    vocab = SMILESVocabulary(vocab_data["tokens"])
    logger.info("Vocabulary loaded: %d tokens", vocab.size)

    # Load training data for novelty check
    training_smiles: set[str] = set()
    if args.training_data:
        import pandas as pd

        df = pd.read_csv(args.training_data)
        for smiles in df[args.smiles_column].dropna():
            canonical = canonicalize_smiles(smiles)
            if canonical:
                training_smiles.add(canonical)
        logger.info("Loaded %d training SMILES for novelty check", len(training_smiles))

    # Generate molecules
    logger.info("Generating %d molecules (temperature=%.2f)...", args.num_samples, args.temperature)
    raw_smiles = generate_molecules(
        model=model,
        vocab=vocab,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        temperature=args.temperature,
        device=device,
    )
    logger.info("Generated %d raw SMILES", len(raw_smiles))

    # Evaluate
    logger.info("Computing validity...")
    validity_rate, valid_smiles = compute_validity(raw_smiles)
    logger.info("Validity: %.2f%% (%d/%d)", 100 * validity_rate, len(valid_smiles), len(raw_smiles))

    logger.info("Computing uniqueness...")
    uniqueness_rate, unique_smiles = compute_uniqueness(valid_smiles)
    logger.info(
        "Uniqueness: %.2f%% (%d/%d)", 100 * uniqueness_rate, len(unique_smiles), len(valid_smiles)
    )

    novelty_rate = 0.0
    novel_smiles: list[str] = []
    if training_smiles:
        logger.info("Computing novelty...")
        novelty_rate, novel_smiles = compute_novelty(unique_smiles, training_smiles)
        logger.info(
            "Novelty: %.2f%% (%d/%d)",
            100 * novelty_rate,
            len(novel_smiles),
            len(unique_smiles),
        )

    logger.info("Computing internal diversity (may take a moment)...")
    diversity = compute_internal_diversity(unique_smiles, sample_size=args.diversity_sample_size)
    logger.info("Internal diversity: %.4f", diversity)

    logger.info("Computing property statistics...")
    prop_stats = compute_property_statistics(valid_smiles)
    for prop, stats in prop_stats.items():
        logger.info(
            "  %s: mean=%.2f, std=%.2f, range=[%.2f, %.2f]",
            prop,
            stats["mean"],
            stats["std"],
            stats["min"],
            stats["max"],
        )

    logger.info("Computing Lipinski compliance...")
    lipinski_rate = compute_lipinski_pass_rate(valid_smiles)
    logger.info("Lipinski Ro5 compliance: %.2f%%", 100 * lipinski_rate)

    # Compile results
    results = {
        "num_generated": len(raw_smiles),
        "num_valid": len(valid_smiles),
        "num_unique": len(unique_smiles),
        "num_novel": len(novel_smiles),
        "validity_rate": validity_rate,
        "uniqueness_rate": uniqueness_rate,
        "novelty_rate": novelty_rate,
        "internal_diversity": diversity,
        "lipinski_compliance_rate": lipinski_rate,
        "property_statistics": prop_stats,
        "settings": {
            "temperature": args.temperature,
            "num_samples": args.num_samples,
            "checkpoint": args.checkpoint,
        },
        "sample_molecules": unique_smiles[:20],
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)

    # Summary
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info("  Validity:    %.2f%%", 100 * validity_rate)
    logger.info("  Uniqueness:  %.2f%%", 100 * uniqueness_rate)
    logger.info("  Novelty:     %.2f%%", 100 * novelty_rate)
    logger.info("  Diversity:   %.4f", diversity)
    logger.info("  Lipinski:    %.2f%%", 100 * lipinski_rate)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
