"""CellVision training script with full training loop.

Features:
    - Configurable via argparse
    - Logging with structured output
    - Checkpoint saving (best and periodic)
    - Early stopping with patience
    - Cosine annealing learning rate schedule
    - Mixed precision training (AMP) for GPU efficiency
    - Training metrics visualization

Usage:
    uv run python scripts/train_model.py \
        --data-dir data/cells \
        --model-type cellnet \
        --epochs 100 \
        --batch-size 32 \
        --lr 1e-3 \
        --output-dir models/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from cell_vision import CELL_TYPES, NUM_CLASSES
from cell_vision.data.dataset import create_data_loaders
from cell_vision.models.cell_classifier import CellNet, CellNetResNet

logger = logging.getLogger("cell_vision.training")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CellVision cell type classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory with per-class image subdirectories",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Input image size (square)")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loading workers")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split fraction")
    parser.add_argument("--test-split", type=float, default=0.15, help="Test split fraction")

    # Model
    parser.add_argument(
        "--model-type",
        choices=["cellnet", "resnet18"],
        default="cellnet",
        help="Model architecture",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use pretrained weights (ResNet only)",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        default=False,
        help="Freeze backbone for transfer learning (ResNet only)",
    )
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate")

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument(
        "--min-delta", type=float, default=1e-4, help="Min improvement for early stopping"
    )
    parser.add_argument("--use-amp", action="store_true", help="Enable mixed precision training")

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory for checkpoints and logs",
    )
    parser.add_argument(
        "--save-every", type=int, default=10, help="Save checkpoint every N epochs"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


class EarlyStopping:
    """Monitors validation loss and stops training if no improvement.

    Args:
        patience: Number of epochs to wait for improvement before stopping.
        min_delta: Minimum decrease in loss to qualify as improvement.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop.

        Args:
            val_loss: Current validation loss.

        Returns:
            True if training should stop.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered after %d epochs without improvement",
                    self.patience,
                )
                return True

        return False


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        model: The neural network.
        train_loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Compute device.
        scaler: GradScaler for mixed precision.
        use_amp: Whether to use automatic mixed precision.

    Returns:
        Tuple of (average loss, accuracy) for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with torch.amp.autocast(device_type=str(device)):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    """Evaluate model on a dataset.

    Args:
        model: The neural network.
        data_loader: Evaluation data loader.
        criterion: Loss function.
        device: Compute device.

    Returns:
        Tuple of (avg loss, accuracy, all true labels, all predicted labels).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels: list[int] = []
    all_preds: list[int] = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(predicted.cpu().tolist())

    return running_loss / total, correct / total, all_labels, all_preds


def plot_training_history(history: dict[str, list[float]], output_path: Path) -> None:
    """Plot and save training curves.

    Args:
        history: Dictionary with loss and accuracy lists per epoch.
        output_path: Path to save the figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    if history["val_loss"]:
        ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    if history["val_acc"]:
        ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Training curves saved to %s", output_path)


def plot_confusion_matrix(
    true_labels: list[int],
    pred_labels: list[int],
    class_names: list[str],
    output_path: Path,
) -> None:
    """Plot and save confusion matrix.

    Args:
        true_labels: Ground truth class indices.
        pred_labels: Predicted class indices.
        class_names: Human-readable class names.
        output_path: Path to save the figure.
    """
    cm = confusion_matrix(true_labels, pred_labels)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_normalized, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix (Normalized)")
    fig.colorbar(im, ax=ax)

    tick_labels = [name.replace("_", " ").title() for name in class_names]
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)

    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text_color = "white" if cm_normalized[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{cm_normalized[i, j]:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Confusion matrix saved to %s", output_path)


def main() -> None:
    """Main training entrypoint."""
    args = parse_args()

    # ---- Setup ----

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.output_dir / "training.log"),
        ],
    )

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)
    logger.info("Configuration: %s", vars(args))

    # ---- Data ----

    logger.info("Loading dataset from %s", args.data_dir)
    data = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        val_split=args.val_split,
        test_split=args.test_split,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    train_loader = data["train"]
    val_loader = data["val"]
    test_loader = data["test"]
    class_weights = data["class_weights"].to(device)

    logger.info("Class weights: %s", class_weights.tolist())

    # ---- Model ----

    if args.model_type == "cellnet":
        model = CellNet(num_classes=NUM_CLASSES, dropout_rate=args.dropout).to(device)
    else:
        model = CellNetResNet(
            num_classes=NUM_CLASSES,
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone,
            dropout_rate=args.dropout,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model: %s | Total params: %d | Trainable: %d",
        args.model_type,
        total_params,
        trainable_params,
    )

    # ---- Loss, optimizer, scheduler ----

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Mixed precision
    scaler = torch.amp.GradScaler() if args.use_amp and device.type == "cuda" else None

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    # ---- Training loop ----

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    start_time = time.time()

    logger.info("Starting training for %d epochs", args.epochs)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            use_amp=args.use_amp,
        )

        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            "Epoch %03d/%03d | Train Loss: %.4f | Train Acc: %.4f | "
            "Val Loss: %.4f | Val Acc: %.4f | LR: %.2e | Time: %.1fs",
            epoch,
            args.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            current_lr,
            epoch_time,
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = args.output_dir / "cellnet_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_type": args.model_type,
                    "num_classes": NUM_CLASSES,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                best_path,
            )
            logger.info("New best model saved (Val Acc: %.4f) -> %s", val_acc, best_path)

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = args.output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                },
                ckpt_path,
            )
            logger.info("Checkpoint saved -> %s", ckpt_path)

        # Early stopping
        if early_stopping(val_loss):
            logger.info("Stopping early at epoch %d", epoch)
            break

    total_time = time.time() - start_time
    logger.info("Training complete in %.1f seconds. Best Val Acc: %.4f", total_time, best_val_acc)

    # ---- Test evaluation ----

    logger.info("Evaluating on test set...")
    best_checkpoint = torch.load(args.output_dir / "cellnet_best.pt", weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_loss, test_acc, true_labels, pred_labels = evaluate(model, test_loader, criterion, device)
    logger.info("Test Loss: %.4f | Test Acc: %.4f", test_loss, test_acc)

    # Classification report
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=CELL_TYPES,
        output_dict=True,
    )
    logger.info(
        "Classification Report:\n%s",
        classification_report(true_labels, pred_labels, target_names=CELL_TYPES),
    )

    # Save report
    report_path = args.output_dir / "classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Classification report saved to %s", report_path)

    # ---- Plots ----

    plot_training_history(history, args.output_dir / "training_curves.png")
    plot_confusion_matrix(
        true_labels,
        pred_labels,
        CELL_TYPES,
        args.output_dir / "confusion_matrix.png",
    )

    # Save final training summary
    summary = {
        "model_type": args.model_type,
        "epochs_trained": len(history["train_loss"]),
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "total_training_time_seconds": total_time,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }

    summary_path = args.output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Training summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
