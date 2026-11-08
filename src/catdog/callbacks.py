"""Callback definitions for training control, checkpointing, and logging."""

from pathlib import Path
from typing import List

import tensorflow as tf

from catdog.config import Config


def get_training_callbacks(
    config: Config,
    experiment_name: str = "improved_cnn",
    early_stopping_patience: int = 10,
    reduce_lr_patience: int = 4,
) -> List[tf.keras.callbacks.Callback]:
    """Build callbacks for adaptive training, checkpointing, and telemetry.
    
    Design explanation:
    - `EarlyStopping`: Halts training when validation metric stops improving. We use a generous
      patience (10 epochs) because networks with BatchNormalization require 5-8 epochs for running
      mean and variance statistics to stabilize on small datasets before validation loss steadily drops.
    - `ModelCheckpoint`: Automatically saves only the best performing `.keras` model based on validation accuracy.
    - `ReduceLROnPlateau`: Cuts learning rate by 50% when progress stalls, allowing fine-grained convergence.
    - `TensorBoard`: Logs scalar metrics, learning rates, and computational graphs for visual inspection.
    """
    checkpoint_path = config.models_dir / f"{experiment_name}.keras"
    tensorboard_log_dir = config.reports_dir / "tensorboard" / experiment_name

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_log_dir),
            histogram_freq=0,
            write_graph=True,
        ),
    ]

    return callbacks
