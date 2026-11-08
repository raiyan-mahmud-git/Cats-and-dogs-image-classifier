"""Unit tests for Stage 2 architectural improvements and callbacks."""

import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import tensorflow as tf

from catdog.callbacks import get_training_callbacks
from catdog.config import Config
from catdog.model import build_gap_model, build_improved_cnn


def test_gap_model_architecture():
    config = Config()
    model = build_gap_model(config)
    assert model.input_shape == (None, 150, 150, 3)
    assert model.output_shape == (None, 1)
    # Check that parameters are drastically lower than the 19M baseline
    assert model.count_params() < 1_000_000


def test_improved_cnn_architecture():
    config = Config()
    model = build_improved_cnn(config)
    assert model.input_shape == (None, 150, 150, 3)
    assert model.output_shape == (None, 1)

    # Check forward pass
    dummy = np.random.randint(0, 256, size=(2, 150, 150, 3)).astype(np.float32)
    out = model(dummy, training=False)
    assert out.shape == (2, 1)
    assert np.all(out.numpy() >= 0.0) and np.all(out.numpy() <= 1.0)


def test_callbacks_builder():
    config = Config()
    callbacks = get_training_callbacks(config, experiment_name="test_exp")
    assert len(callbacks) == 4
    types = [type(c).__name__ for c in callbacks]
    assert "EarlyStopping" in types
    assert "ModelCheckpoint" in types
    assert "ReduceLROnPlateau" in types
    assert "TensorBoard" in types
