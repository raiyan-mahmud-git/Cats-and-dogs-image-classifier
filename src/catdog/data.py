"""Data loading, caching, prefetching, and preprocessing pipeline."""

import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from catdog.config import Config


def set_seed(seed: int = 42) -> None:
    """Set global seeds for Python, NumPy, and TensorFlow for reproducibility.
    
    Design explanation:
    Machine learning experiments must be reproducible. Setting random seeds across
    Python's internal random, NumPy, and TensorFlow guarantees deterministic data
    shuffling, weight initialization, and dataset batching.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_train_val_datasets(
    config: Config,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load train and validation datasets with cache and prefetch optimizations.
    
    Design explanation:
    `image_dataset_from_directory` replaces the deprecated `ImageDataGenerator`,
    returning a modern `tf.data.Dataset`. Chaining `.cache()` stores decoded images
    in memory after the first epoch, and `.prefetch(AUTOTUNE)` overlaps data
    loading on CPU while the model computes on GPU/CPU, preventing I/O bottlenecks.
    """
    if not config.train_dir.exists():
        raise FileNotFoundError(f"Training directory not found at: {config.train_dir}")
    if not config.validation_dir.exists():
        raise FileNotFoundError(f"Validation directory not found at: {config.validation_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(config.train_dir),
        image_size=(config.img_height, config.img_width),
        batch_size=config.batch_size,
        label_mode="binary",
        shuffle=True,
        seed=config.seed,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(config.validation_dir),
        image_size=(config.img_height, config.img_width),
        batch_size=config.batch_size,
        label_mode="binary",
        shuffle=False,
    )

    # In-memory caching and background prefetching
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds


def load_test_dataset(config: Config) -> Tuple[tf.data.Dataset, List[Path]]:
    """Load the flat unlabelled test images deterministically.
    
    Design explanation:
    The test set contains 50 flat image files without class subfolders.
    Instead of brittle workarounds, we directly read files using `tf.io`
    sorted in lexicographical order (to match freeCodeCamp's ground truth answer key)
    and assert exact count equality to avoid silent zip truncations.
    """
    if not config.test_dir.exists():
        raise FileNotFoundError(f"Test directory not found at: {config.test_dir}")

    valid_extensions = {".jpg", ".jpeg", ".png"}
    test_files = sorted(
        [p for p in config.test_dir.iterdir() if p.suffix.lower() in valid_extensions],
        key=lambda p: p.name,
    )

    assert len(test_files) == 50, (
        f"Expected exactly 50 test images for freeCodeCamp benchmark, found {len(test_files)}"
    )

    def _parse_image(file_path: tf.Tensor) -> tf.Tensor:
        raw = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(raw, channels=config.channels)
        img = tf.image.resize(img, [config.img_height, config.img_width])
        return img

    file_paths_str = [str(p) for p in test_files]
    test_ds = tf.data.Dataset.from_tensor_slices(file_paths_str)
    test_ds = test_ds.map(_parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(config.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    return test_ds, test_files
