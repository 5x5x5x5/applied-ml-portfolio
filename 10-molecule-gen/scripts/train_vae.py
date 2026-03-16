#!/usr/bin/env python3
"""Training script for the Molecular VAE.

Trains the SMILES-based Variational Autoencoder on a dataset of drug-like
molecules. Supports KL annealing, learning rate scheduling, gradient clipping,
and periodic checkpoint saving.

Usage:
    uv run python scripts/train_vae.py \\
        --data data/molecules.csv \\
        --epochs 100 \\
        --batch-size 256 \\
        --latent-dim 128 \\
        --lr 3e-4 \\
        --kl-annealing-steps 5000 \\
        --checkpoint-dir checkpoints/

Example data format (CSV with 'smiles' column):
    smiles
    CC(=O)Oc1ccccc1C(=O)O
    c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34
    CC(C)Cc1ccc(C(C)C(=O)O)cc1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from molecule_gen.chemistry.smiles_processor import SMILESVocabulary
from molecule_gen.models.mol_vae import MolecularVAE, VAEConfig, kl_annealing_schedule, vae_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_vae")


class SMILESDataset(Dataset):
    """PyTorch Dataset for tokenized SMILES strings.

    Reads SMILES from a list, encodes them using the provided vocabulary,
    and returns padded token index tensors.
    """

    def __init__(
        self,
        smiles_list: list[str],
        vocab: SMILESVocabulary,
        max_length: int = 120,
    ) -> None:
        self.smiles_list = smiles_list
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        smiles = self.smiles_list[idx]
        encoded = self.vocab.encode_smiles(smiles, max_length=self.max_length)
        tokens = torch.tensor(encoded, dtype=torch.long)

        # Compute true sequence length (excluding padding)
        length = (tokens != self.vocab.PAD_IDX).sum()

        return {"tokens": tokens, "length": length}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for VAE training."""
    parser = argparse.ArgumentParser(
        description="Train a Molecular VAE on SMILES data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to CSV file with a 'smiles' column",
    )
    parser.add_argument(
        "--smiles-column",
        type=str,
        default="smiles",
        help="Name of the SMILES column in the CSV",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=120,
        help="Maximum SMILES sequence length (longer sequences are truncated)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation",
    )

    # Model architecture
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--encoder-hidden-dim", type=int, default=256)
    parser.add_argument("--decoder-hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-encoder-layers", type=int, default=2)
    parser.add_argument("--num-decoder-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Gradient norm clipping")
    parser.add_argument(
        "--kl-annealing-steps",
        type=int,
        default=5000,
        help="Steps to linearly anneal KL weight from 0 to 1",
    )
    parser.add_argument(
        "--kl-strategy",
        type=str,
        default="linear",
        choices=["linear", "cyclical"],
        help="KL annealing strategy",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving model checkpoints",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for training",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def train_epoch(
    model: MolecularVAE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    global_step: int,
    kl_annealing_steps: int,
    kl_strategy: str,
    grad_clip: float,
) -> tuple[float, float, float, int]:
    """Train for one epoch.

    Returns:
        Tuple of (avg_total_loss, avg_recon_loss, avg_kl_loss, updated_global_step).
    """
    model.train()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    num_batches = 0

    for batch in dataloader:
        tokens = batch["tokens"].to(device)
        lengths = batch["length"].to(device)

        # Forward pass
        logits, mu, logvar = model(tokens, lengths=lengths)

        # Compute loss with KL annealing
        beta = kl_annealing_schedule(global_step, kl_annealing_steps, strategy=kl_strategy)
        total_loss, recon_loss, kl_loss = vae_loss(
            logits=logits,
            targets=tokens,
            mu=mu,
            logvar=logvar,
            beta=beta,
        )

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss_sum += total_loss.item()
        recon_loss_sum += recon_loss.item()
        kl_loss_sum += kl_loss.item()
        num_batches += 1
        global_step += 1

    avg_total = total_loss_sum / max(num_batches, 1)
    avg_recon = recon_loss_sum / max(num_batches, 1)
    avg_kl = kl_loss_sum / max(num_batches, 1)

    return avg_total, avg_recon, avg_kl, global_step


def validate(
    model: MolecularVAE,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate on validation set.

    Returns:
        Tuple of (avg_total_loss, avg_recon_loss, avg_kl_loss).
    """
    model.eval()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            tokens = batch["tokens"].to(device)
            lengths = batch["length"].to(device)

            logits, mu, logvar = model(tokens, lengths=lengths)
            total_loss, recon_loss, kl_loss = vae_loss(
                logits=logits,
                targets=tokens,
                mu=mu,
                logvar=logvar,
                beta=1.0,  # Full KL for validation
            )

            total_loss_sum += total_loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            num_batches += 1

    avg_total = total_loss_sum / max(num_batches, 1)
    avg_recon = recon_loss_sum / max(num_batches, 1)
    avg_kl = kl_loss_sum / max(num_batches, 1)

    return avg_total, avg_recon, avg_kl


def save_checkpoint(
    model: MolecularVAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    val_loss: float,
    config: VAEConfig,
    path: Path,
) -> None:
    """Save model checkpoint with training state."""
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "config": {
            "vocab_size": config.vocab_size,
            "embedding_dim": config.embedding_dim,
            "encoder_hidden_dim": config.encoder_hidden_dim,
            "decoder_hidden_dim": config.decoder_hidden_dim,
            "latent_dim": config.latent_dim,
            "num_encoder_layers": config.num_encoder_layers,
            "num_decoder_layers": config.num_decoder_layers,
            "max_sequence_length": config.max_sequence_length,
            "dropout": config.dropout,
        },
    }
    torch.save(checkpoint, path)
    logger.info("Saved checkpoint to %s (epoch %d, val_loss=%.4f)", path, epoch, val_loss)


def main() -> None:
    """Main training loop."""
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # Load data
    logger.info("Loading SMILES data from %s", args.data)
    df = pd.read_csv(args.data)
    smiles_list = df[args.smiles_column].dropna().tolist()
    logger.info("Loaded %d SMILES strings", len(smiles_list))

    # Filter by length
    smiles_list = [s for s in smiles_list if len(s) <= args.max_length * 2]
    logger.info("After length filtering: %d SMILES", len(smiles_list))

    # Build vocabulary from training data
    vocab = SMILESVocabulary.build_from_corpus(smiles_list, min_frequency=2)
    logger.info("Vocabulary size: %d tokens", vocab.size)

    # Train/validation split
    n_val = int(len(smiles_list) * args.val_split)
    indices = np.random.permutation(len(smiles_list))
    val_smiles = [smiles_list[i] for i in indices[:n_val]]
    train_smiles = [smiles_list[i] for i in indices[n_val:]]
    logger.info("Train: %d, Validation: %d", len(train_smiles), len(val_smiles))

    # Datasets and DataLoaders
    train_dataset = SMILESDataset(train_smiles, vocab, max_length=args.max_length)
    val_dataset = SMILESDataset(val_smiles, vocab, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model
    config = VAEConfig(
        vocab_size=vocab.size,
        embedding_dim=args.embedding_dim,
        encoder_hidden_dim=args.encoder_hidden_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        latent_dim=args.latent_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        max_sequence_length=args.max_length,
        dropout=args.dropout,
    )
    model = MolecularVAE(config).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s", f"{num_params:,}")

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        logger.info("Resumed from checkpoint: epoch %d, step %d", start_epoch, global_step)

    # Checkpoint directory
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save vocabulary
    vocab_path = ckpt_dir / "vocabulary.json"
    with open(vocab_path, "w") as f:
        json.dump(
            {"tokens": [vocab.idx_to_token[i] for i in range(vocab.size)]},
            f,
            indent=2,
        )
    logger.info("Saved vocabulary to %s", vocab_path)

    # Training loop
    logger.info("Starting training for %d epochs", args.epochs)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_total, train_recon, train_kl, global_step = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            global_step=global_step,
            kl_annealing_steps=args.kl_annealing_steps,
            kl_strategy=args.kl_strategy,
            grad_clip=args.grad_clip,
        )

        # Validate
        val_total, val_recon, val_kl = validate(model, val_loader, device)

        # Learning rate scheduling
        scheduler.step(val_total)

        epoch_time = time.time() - epoch_start
        beta = kl_annealing_schedule(global_step, args.kl_annealing_steps, args.kl_strategy)
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %d/%d (%.1fs) | Train: total=%.4f recon=%.4f kl=%.4f | "
            "Val: total=%.4f recon=%.4f kl=%.4f | beta=%.3f lr=%.2e",
            epoch + 1,
            args.epochs,
            epoch_time,
            train_total,
            train_recon,
            train_kl,
            val_total,
            val_recon,
            val_kl,
            beta,
            current_lr,
        )

        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = ckpt_dir / f"mol_vae_epoch_{epoch + 1:03d}.pt"
            save_checkpoint(model, optimizer, epoch, global_step, val_total, config, ckpt_path)

        # Save best model
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_path = ckpt_dir / "mol_vae_best.pt"
            save_checkpoint(model, optimizer, epoch, global_step, val_total, config, best_path)
            logger.info("New best validation loss: %.4f", best_val_loss)

    # Save final checkpoint
    final_path = ckpt_dir / "mol_vae_final.pt"
    save_checkpoint(model, optimizer, args.epochs - 1, global_step, val_total, config, final_path)

    logger.info("Training complete. Best validation loss: %.4f", best_val_loss)


if __name__ == "__main__":
    main()
