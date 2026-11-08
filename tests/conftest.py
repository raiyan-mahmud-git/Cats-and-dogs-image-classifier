"""
tests/conftest.py
Shared test fixtures and configuration.

Design decisions:
1. The `tiny_model` fixture builds a minimal 2-layer CNN (~1K params) that follows
   the same interface as our production models (Input → Rescaling → Conv → GAP → Dense(1)).
   This lets tests verify prediction pipelines WITHOUT the 25 MB trained model.
2. The `tmp_images` fixture creates a reusable set of edge-case images (RGB, gray,
   RGBA, corrupt, empty) used across multiple test modules.
3. We register `requires_model` and `slow` markers so tests can be filtered:
   `pytest -m "not slow"` runs the fast unit-only suite in ~3 seconds.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure src/ is on the Python path (for non-editable-install workflows)
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------------------------
# Shared image fixture (edge-case gallery)
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_images(tmp_path):
    """Create a set of test images covering common edge cases.

    Returns a dict mapping descriptive keys to file paths:
    - rgb:       normal 200×200 RGB JPEG
    - gray:      100×100 grayscale PNG
    - rgba:      100×100 RGBA PNG (with transparency)
    - tiny:      1×1 degenerate RGB JPEG
    - corrupt:   text file masquerading as .jpg
    - wrong_ext: plain .txt file
    - empty:     0-byte file with .jpg extension
    """
    from PIL import Image as PILImage

    images = {}

    # Normal RGB JPEG
    rgb = PILImage.new("RGB", (200, 200), color=(128, 64, 32))
    rgb_path = tmp_path / "normal.jpg"
    rgb.save(rgb_path)
    images["rgb"] = rgb_path

    # Grayscale PNG
    gray = PILImage.new("L", (100, 100), color=128)
    gray_path = tmp_path / "grayscale.png"
    gray.save(gray_path)
    images["gray"] = gray_path

    # RGBA PNG (with transparency)
    rgba = PILImage.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    rgba_path = tmp_path / "transparent.png"
    rgba.save(rgba_path)
    images["rgba"] = rgba_path

    # Tiny (1×1) image — valid but degenerate
    tiny = PILImage.new("RGB", (1, 1), color=(0, 0, 0))
    tiny_path = tmp_path / "tiny.jpg"
    tiny.save(tiny_path)
    images["tiny"] = tiny_path

    # Non-image file with image extension
    bad_path = tmp_path / "corrupt.jpg"
    bad_path.write_text("this is not a JPEG file")
    images["corrupt"] = bad_path

    # Valid file but wrong extension
    txt_path = tmp_path / "readme.txt"
    txt_path.write_text("not an image")
    images["wrong_ext"] = txt_path

    # Empty file
    empty_path = tmp_path / "empty.jpg"
    empty_path.write_bytes(b"")
    images["empty"] = empty_path

    return images


# ---------------------------------------------------------------------------
# Tiny model fixture (CI-safe, no trained weights needed)
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_model(tmp_path):
    """Build and save a tiny (~1K params) model for CI testing.

    Architecture mirrors production: Input → Rescaling → Conv2D → GAP → Dense(1, sigmoid).
    Weights are random (untrained), but the model predicts valid [0, 1] probabilities
    and has the correct input/output shapes.

    Returns (model, model_path) where model_path is a saved .keras file.
    """
    import tensorflow as tf

    inp = tf.keras.layers.Input(shape=(150, 150, 3), name="image_input")
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_rescaling")(inp)
    x = tf.keras.layers.Conv2D(4, (3, 3), activation="relu", name="conv_tiny")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid", name="predictions")(x)

    model = tf.keras.Model(inputs=inp, outputs=x, name="tiny_test_model")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model_path = tmp_path / "tiny_model.keras"
    model.save(model_path)

    return model, model_path


@pytest.fixture
def tiny_tflite(tiny_model, tmp_path):
    """Convert the tiny model to TFLite format for testing.

    Uses intermediate SavedModel export for reliable cross-version compatibility
    (avoids MLIR direct Keras conversion crashes on Windows TF 2.16).

    Returns the path to the saved .tflite file.
    """
    import tensorflow as tf

    model, _ = tiny_model

    # Use intermediate SavedModel directory to avoid Keras 3 -> TFLite MLIR crash
    saved_model_dir = tmp_path / "tiny_saved_model"
    model.export(str(saved_model_dir))

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    tflite_bytes = converter.convert()

    tflite_path = tmp_path / "tiny_model.tflite"
    tflite_path.write_bytes(tflite_bytes)

    return tflite_path


# ---------------------------------------------------------------------------
# Helper: check if trained models exist
# ---------------------------------------------------------------------------
def _trained_model_available() -> bool:
    """Check if any trained model file exists in the models/ directory."""
    from catdog.config import Config

    config = Config()
    model_names = [
        "transfer_model_float16.tflite",
        "transfer_model_float32.tflite",
        "transfer_model_inference.keras",
        "transfer_model.keras",
        "improved_cnn.keras",
        "baseline_model.keras",
    ]
    return any((config.models_dir / n).exists() for n in model_names)


requires_model = pytest.mark.skipif(
    not _trained_model_available(),
    reason="No trained model files found — skipping integration test",
)
