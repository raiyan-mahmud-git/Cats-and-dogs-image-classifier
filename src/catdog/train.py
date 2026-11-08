"""Model training script with history tracking and diagnostic visualization."""

import json
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import tensorflow as tf

from catdog.config import Config
from catdog.data import load_train_val_datasets, set_seed
from catdog.model import build_baseline_model


def plot_and_save_training_curves(history: Dict[str, Any], output_path: Path) -> None:
    """Plot and save training and validation loss and accuracy curves.
    
    Design explanation:
    Comparing train vs validation curves across all epochs visually diagnoses
    overfitting, learning plateaus, and training stability without guessing.
    """
    acc = history["accuracy"]
    val_acc = history["val_accuracy"]
    loss = history["loss"]
    val_loss = history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs_range, acc, "o-", label="Training Accuracy", color="#2563eb")
    ax1.plot(epochs_range, val_acc, "s-", label="Validation Accuracy", color="#16a34a")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Training & Validation Accuracy")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    ax2.plot(epochs_range, loss, "o-", label="Training Loss", color="#dc2626")
    ax2.plot(epochs_range, val_loss, "s-", label="Validation Loss", color="#ea580c")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss (Binary Crossentropy)")
    ax2.set_title("Training & Validation Loss")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[Training] Saved training curves to: {output_path}")


def train_baseline(config: Config = None) -> tf.keras.Model:
    """Execute baseline model training workflow."""
    if config is None:
        config = Config()

    print("=" * 60)
    print("STAGE 1: TRAINING BASELINE CNN")
    print(f"Batch size: {config.batch_size} | Epochs: {config.epochs} | Seed: {config.seed}")
    print("=" * 60)

    # 1. Reproducibility
    set_seed(config.seed)

    # 2. Data loading
    train_ds, val_ds = load_train_val_datasets(config)

    # 3. Model construction
    model = build_baseline_model(config)
    model.summary()

    # 4. Training (NO manual steps_per_epoch to prevent wasted 1-step epochs)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        verbose=1,
    )

    # 5. Save model in modern Keras v3 format (.keras instead of legacy .h5)
    model_save_path = config.models_dir / "baseline_model.keras"
    model.save(model_save_path)
    print(f"[Training] Model successfully saved to: {model_save_path}")

    # 6. Save raw history to JSON
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_path = config.reports_dir / "baseline_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)
    print(f"[Training] Training history saved to: {history_path}")

    # 7. Diagnostic plot
    plot_path = config.reports_dir / "baseline_training_curves.png"
    plot_and_save_training_curves(history_dict, plot_path)

    return model


if __name__ == "__main__":
    train_baseline()
