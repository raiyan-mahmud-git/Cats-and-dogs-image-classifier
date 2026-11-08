"""Unit tests for Stage 3 transfer learning architecture and fine-tuning logic."""

import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import tensorflow as tf

from catdog.config import Config
from catdog.model import build_transfer_model


def test_transfer_model_shapes_and_freezing():
    config = Config()
    model = build_transfer_model(config)
    assert model.input_shape == (None, 150, 150, 3)
    assert model.output_shape == (None, 1)

    base_model = model.get_layer("mobilenetv2_base")
    # Base model should be frozen in initial phase
    assert base_model.trainable is False
    assert sum(p.numpy().size for p in base_model.trainable_weights) == 0

    # Total trainable params should only be the classifier head (~164k)
    trainable_params = sum(p.numpy().size for p in model.trainable_weights)
    assert trainable_params < 200_000


def test_transfer_preprocessing_rescaling():
    config = Config()
    model = build_transfer_model(config)
    rescaling_layer = model.get_layer("mobilenet_rescaling")

    # Test [0, 255] -> [-1, 1] transformation
    test_tensor = tf.constant([[[[0.0, 127.5, 255.0]]]], dtype=tf.float32)
    rescaled = rescaling_layer(test_tensor).numpy()

    np.testing.assert_allclose(rescaled, [[[[-1.0, 0.0, 1.0]]]], atol=1e-5)


def test_fine_tuning_unfreezing():
    config = Config()
    model = build_transfer_model(config)
    base_model = model.get_layer("mobilenetv2_base")
    base_model.trainable = True

    # Freeze up to layer 100
    for layer in base_model.layers[:100]:
        layer.trainable = False

    # Freeze all BN layers
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # Layers after 100 that are not BatchNorm should be trainable
    trainable_base_layers = [lyr for lyr in base_model.layers[100:] if lyr.trainable]
    assert len(trainable_base_layers) > 0
    # No BatchNorm layer in the entire base should be trainable
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            assert layer.trainable is False
