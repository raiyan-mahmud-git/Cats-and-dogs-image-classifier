"""Callback definitions for training control, checkpointing, and logging."""

from typing import List

import tensorflow as tf

from catdog.config import Config


def get_training_callbacks(
    config: Config,
    experiment_name: str = "improved_cnn",
    early_stopping_patience: int = 10,
    reduce_lr_patience: int = 4,
) -> List[tf.keras.callbacks.Callback]:
    """Build callbacks for adaptive training, checkpointing, and telemetry."""
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
