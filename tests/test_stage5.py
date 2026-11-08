"""
tests/test_stage5.py
Tests for Stage 5: Packaging & Inference.

Covers:
- Image validation (good files, bad extensions, corrupt files, RGBA, grayscale)
- CatDogPredictor with the real transfer model
- TFLite export and inference
- CLI argument parsing
"""

import csv
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image as PILImage

from catdog.predict import (
    CatDogPredictor,
    ImageValidationError,
    _load_and_validate_image,
    _build_parser,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_images(tmp_path):
    """Create a set of test images covering edge cases."""
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

    # Tiny (1x1) image — valid but degenerate
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
# Image validation tests
# ---------------------------------------------------------------------------
class TestImageValidation:
    def test_load_valid_rgb_image(self, tmp_images):
        """A normal RGB JPEG loads successfully and has the right shape."""
        arr = _load_and_validate_image(tmp_images["rgb"], target_size=(150, 150))
        assert arr.shape == (150, 150, 3)
        assert arr.dtype == np.float32

    def test_load_grayscale_converts_to_rgb(self, tmp_images):
        """Grayscale images get replicated to 3 channels."""
        arr = _load_and_validate_image(tmp_images["gray"], target_size=(150, 150))
        assert arr.shape == (150, 150, 3)
        # All three channels should be identical for a grayscale conversion
        assert np.allclose(arr[:, :, 0], arr[:, :, 1])

    def test_load_rgba_strips_alpha(self, tmp_images):
        """RGBA images are composited to RGB (alpha dropped)."""
        arr = _load_and_validate_image(tmp_images["rgba"], target_size=(150, 150))
        assert arr.shape == (150, 150, 3)  # No alpha channel

    def test_load_tiny_image_resizes(self, tmp_images):
        """Even a 1x1 image is resized to the target dimensions."""
        arr = _load_and_validate_image(tmp_images["tiny"], target_size=(150, 150))
        assert arr.shape == (150, 150, 3)

    def test_reject_corrupt_file(self, tmp_images):
        """A corrupt file raises ImageValidationError."""
        with pytest.raises(ImageValidationError, match="Cannot decode"):
            _load_and_validate_image(tmp_images["corrupt"])

    def test_reject_wrong_extension(self, tmp_images):
        """A .txt file is rejected before any decode attempt."""
        with pytest.raises(ImageValidationError, match="Unsupported file type"):
            _load_and_validate_image(tmp_images["wrong_ext"])

    def test_reject_empty_file(self, tmp_images):
        """An empty file is rejected."""
        with pytest.raises(ImageValidationError, match="empty"):
            _load_and_validate_image(tmp_images["empty"])

    def test_reject_missing_file(self, tmp_path):
        """A nonexistent path raises ImageValidationError."""
        with pytest.raises(ImageValidationError, match="not found"):
            _load_and_validate_image(tmp_path / "nonexistent.jpg")


# ---------------------------------------------------------------------------
# Predictor tests (require a trained model)
# ---------------------------------------------------------------------------
class TestPredictor:
    @pytest.fixture(autouse=True)
    def _check_model(self):
        """Skip predictor tests if no trained model is available."""
        from catdog.config import Config

        config = Config()
        model_exists = any(
            (config.models_dir / n).exists()
            for n in (
                "transfer_model.keras",
                "improved_cnn.keras",
                "baseline_model.keras",
            )
        )
        if not model_exists:
            pytest.skip("No trained model available")

    def test_predict_single_image(self, tmp_images):
        """Predictor returns a valid result dict on a normal image."""
        predictor = CatDogPredictor()
        result = predictor.predict_image(tmp_images["rgb"])

        assert result["label"] in ("Cat", "Dog")
        assert 0.0 <= result["confidence_pct"] <= 100.0
        assert 0.0 <= result["dog_probability"] <= 1.0
        assert result["time_ms"] > 0

    def test_predict_grayscale_image(self, tmp_images):
        """Predictor handles grayscale input without crashing."""
        predictor = CatDogPredictor()
        result = predictor.predict_image(tmp_images["gray"])
        assert result["label"] in ("Cat", "Dog")

    def test_predict_rgba_image(self, tmp_images):
        """Predictor handles RGBA input without crashing."""
        predictor = CatDogPredictor()
        result = predictor.predict_image(tmp_images["rgba"])
        assert result["label"] in ("Cat", "Dog")

    def test_predict_folder(self, tmp_images):
        """Folder mode classifies all valid images and marks corrupt ones."""
        predictor = CatDogPredictor()
        folder = tmp_images["rgb"].parent
        results = predictor.predict_folder(folder)

        # Should have at least the valid images
        valid = [r for r in results if "error" not in r]
        errors = [r for r in results if "error" in r]
        assert len(valid) >= 3  # rgb, gray, rgba, tiny
        # The corrupt.jpg should produce an error entry
        corrupt_errors = [r for r in errors if "corrupt" in r["image"]]
        assert len(corrupt_errors) == 1

    def test_predict_bad_image_raises(self, tmp_images):
        """Predictor raises ImageValidationError on corrupt input."""
        predictor = CatDogPredictor()
        with pytest.raises(ImageValidationError):
            predictor.predict_image(tmp_images["corrupt"])


# ---------------------------------------------------------------------------
# CLI argument parsing tests
# ---------------------------------------------------------------------------
class TestCLI:
    def test_parser_single_image(self):
        """Parser accepts a single image path."""
        parser = _build_parser()
        args = parser.parse_args(["photo.jpg"])
        assert args.input == "photo.jpg"
        assert args.model is None
        assert args.csv is None

    def test_parser_folder_with_csv(self):
        """Parser accepts folder + --csv flag."""
        parser = _build_parser()
        args = parser.parse_args(["photos/", "--csv", "out.csv"])
        assert args.input == "photos/"
        assert args.csv == "out.csv"

    def test_parser_custom_model(self):
        """Parser accepts --model flag."""
        parser = _build_parser()
        args = parser.parse_args(["img.jpg", "--model", "my.tflite"])
        assert args.model == "my.tflite"


# ---------------------------------------------------------------------------
# TFLite export test (if model exists)
# ---------------------------------------------------------------------------
class TestTFLiteExport:
    def test_tflite_inference_from_existing(self):
        """Verify TFLite inference works on an existing .tflite file.

        TFLite conversion uses MLIR internally and can crash the process
        on some Windows + TF 2.16 configurations. We test conversion
        separately via `python -m catdog.export`, and here we only verify
        that TFLite *inference* works if a converted model exists.
        """
        import tensorflow as tf
        from catdog.config import Config

        config = Config()
        tflite_path = config.models_dir / "transfer_model_float32.tflite"

        if not tflite_path.exists():
            pytest.skip("No TFLite model found — run `python -m catdog.export` first")

        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()
        out = interpreter.get_output_details()

        dummy = np.random.rand(1, 150, 150, 3).astype(np.float32)
        interpreter.set_tensor(inp[0]["index"], dummy)
        interpreter.invoke()
        prediction = interpreter.get_tensor(out[0]["index"])

        assert prediction.shape == (1, 1)
        assert 0.0 <= float(prediction[0, 0]) <= 1.0

    def test_predictor_with_tflite(self):
        """CatDogPredictor correctly loads and infers from a .tflite model."""
        from catdog.config import Config

        config = Config()
        tflite_path = config.models_dir / "transfer_model_float32.tflite"

        if not tflite_path.exists():
            pytest.skip("No TFLite model found — run `python -m catdog.export` first")

        predictor = CatDogPredictor(model_path=tflite_path)
        assert predictor._is_tflite is True

        # Predict on a real test image
        test_img = config.test_dir / "1.jpg"
        if not test_img.exists():
            pytest.skip("No test images available")

        result = predictor.predict_image(test_img)
        assert result["label"] in ("Cat", "Dog")
        assert 0.0 <= result["dog_probability"] <= 1.0
