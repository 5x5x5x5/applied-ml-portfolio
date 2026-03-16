"""Shared fixtures for MoleculeGen test suite."""

from __future__ import annotations

import pytest
import torch

from molecule_gen.chemistry.smiles_processor import SMILESVocabulary
from molecule_gen.models.mol_vae import MolecularVAE, VAEConfig

# ---- Well-known drug molecule SMILES for testing ----

# Aspirin (acetylsalicylic acid)
ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"

# Ibuprofen
IBUPROFEN_SMILES = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"

# Caffeine
CAFFEINE_SMILES = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"

# Paracetamol (acetaminophen)
PARACETAMOL_SMILES = "CC(=O)Nc1ccc(O)cc1"

# Benzene (simplest aromatic)
BENZENE_SMILES = "c1ccccc1"

# Ethanol (simplest valid drug-like molecule for testing)
ETHANOL_SMILES = "CCO"

# A set of common drug SMILES for corpus-based tests
DRUG_SMILES_CORPUS: list[str] = [
    ASPIRIN_SMILES,
    IBUPROFEN_SMILES,
    CAFFEINE_SMILES,
    PARACETAMOL_SMILES,
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",  # Pyrene
    "CC(=O)O",  # Acetic acid
    "c1ccc(cc1)O",  # Phenol
    "CC(C)(C)c1ccc(O)cc1",  # 4-tert-Butylphenol
    "O=C(O)c1ccccc1",  # Benzoic acid
    "CCN(CC)CC",  # Triethylamine
    "c1ccncc1",  # Pyridine
    "c1ccc(NC(=O)C)cc1",  # Acetanilide
    "OC(=O)CC(O)(CC(=O)O)C(=O)O",  # Citric acid
    "c1ccc(c(c1)O)O",  # Catechol
    "CC(=O)c1ccccc1",  # Acetophenone
]


@pytest.fixture
def vocab() -> SMILESVocabulary:
    """Provide a default SMILES vocabulary."""
    return SMILESVocabulary()


@pytest.fixture
def corpus_vocab() -> SMILESVocabulary:
    """Provide a vocabulary built from the drug SMILES corpus."""
    return SMILESVocabulary.build_from_corpus(DRUG_SMILES_CORPUS, min_frequency=1)


@pytest.fixture
def vae_config(vocab: SMILESVocabulary) -> VAEConfig:
    """Provide a small VAE config suitable for fast unit tests."""
    return VAEConfig(
        vocab_size=vocab.size,
        embedding_dim=16,
        encoder_hidden_dim=32,
        decoder_hidden_dim=32,
        latent_dim=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
        max_sequence_length=60,
        dropout=0.0,
    )


@pytest.fixture
def vae_model(vae_config: VAEConfig) -> MolecularVAE:
    """Provide an untrained MolecularVAE with small dimensions for testing."""
    model = MolecularVAE(vae_config)
    model.eval()
    return model


@pytest.fixture
def device() -> torch.device:
    """Provide the appropriate device for testing."""
    return torch.device("cpu")
