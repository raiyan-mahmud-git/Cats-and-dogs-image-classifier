"""Sanity checks for configuration, data pipeline, and model architecture."""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import tensorflow as tf
from catdog.config import Config
from catdog.model import build_baseline_model


def test_config_paths():
    config = Config()
    assert config.img_height == 150
    assert config.img_width == 150
    assert config.epochs == 20
    assert len(config.fcc_test_answers) == 50


def test_model_build_and_shapes():
    config = Config()
    model = build_baseline_model(config)
    assert model.input_shape == (None, 150, 150, 3)
    assert model.output_shape == (None, 1)

    # Test dummy forward pass
    dummy_input = np.random.randint(0, 256, size=(2, 150, 150, 3)).astype(np.float32)
    output = model(dummy_input, training=False)
    assert output.shape == (2, 1)
    # Output should be valid probabilities [0, 1]
    assert np.all(output.numpy() >= 0.0)
    assert np.all(output.numpy() <= 1.0)
