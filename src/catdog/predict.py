"""Production inference module and CLI entry point."""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Defer heavy TensorFlow import until actually needed (keeps --help fast)
_tf = None
_keras = None


def _lazy_imports():
    """Import TensorFlow lazily so `--help` and argument validation are instant."""
    global _tf, _keras
    if _tf is None:
        import tensorflow as tf
        from tensorflow import keras

        _tf = tf
        _keras = keras


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
DEFAULT_IMG_SIZE = (150, 150)
CLASS_NAMES = {0: "Cat", 1: "Dog"}


# ---------------------------------------------------------------------------
# Image loading and validation
# ---------------------------------------------------------------------------
class ImageValidationError(Exception):
    """Raised when an input file cannot be processed as a valid image."""

    pass


def _load_and_validate_image(
    image_path: Path,
    target_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
) -> np.ndarray:
    """Load, validate, and preprocess a single image file.

    Validation steps (in order):
    1. File existence check.
    2. Extension whitelist check (prevents feeding .zip, .mp4, etc.).
    3. File size cap (rejects files > 20 MB).
    4. Decode attempt with Pillow (catches truncated / corrupt files).
    5. Channel normalization: grayscale → RGB, RGBA → RGB (drop alpha).
    6. Resize to target dimensions.

    Returns:
        np.ndarray of shape (H, W, 3) with float32 values in [0, 255].
    """
    image_path = Path(image_path)

    # 1. Existence
    if not image_path.exists():
        raise ImageValidationError(f"File not found: {image_path}")

    if not image_path.is_file():
        raise ImageValidationError(f"Not a file: {image_path}")

    # 2. Extension
    ext = image_path.suffix.lower()
    if ext not in VALID_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file type '{ext}'. Accepted: {', '.join(sorted(VALID_EXTENSIONS))}"
        )

    # 3. File size
    file_size = image_path.stat().st_size
    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        raise ImageValidationError(
            f"File too large ({size_mb:.1f} MB). Maximum is {MAX_FILE_SIZE_MB} MB."
        )

    if file_size == 0:
        raise ImageValidationError(f"File is empty (0 bytes): {image_path}")

    # 4. Decode with Pillow (catches corrupt / truncated data)
    try:
        from PIL import Image as PILImage

        pil_img = PILImage.open(image_path)
        pil_img.load()  # Force full decode; truncated files fail here
    except Exception as e:
        raise ImageValidationError(f"Cannot decode image '{image_path.name}': {e}")

    # 5. Channel normalization
    if pil_img.mode == "L":
        # Grayscale → RGB by replicating the single channel 3 times
        pil_img = pil_img.convert("RGB")
    elif pil_img.mode == "RGBA":
        # RGBA → RGB by compositing onto a white background
        background = PILImage.new("RGB", pil_img.size, (255, 255, 255))
        background.paste(pil_img, mask=pil_img.split()[3])  # alpha channel
        pil_img = background
    elif pil_img.mode == "P":
        # Palette mode → RGB
        pil_img = pil_img.convert("RGB")
    elif pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # 6. Resize
    pil_img = pil_img.resize(target_size, PILImage.BILINEAR)

    return np.array(pil_img, dtype=np.float32)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
class CatDogPredictor:
    """Unified predictor supporting both .keras and .tflite models."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
    ):
        _lazy_imports()
        from catdog.config import Config

        config = Config()

        if model_path is None:
            # Default to the best available model
            for name in (
                "transfer_model_float16.tflite",
                "transfer_model_float32.tflite",
                "transfer_model_inference.keras",
                "transfer_model.keras",
                "improved_cnn.keras",
                "baseline_model.keras",
            ):
                candidate = config.models_dir / name
                if candidate.exists():
                    model_path = candidate
                    break

        if model_path is None:
            raise FileNotFoundError("No trained model found in models/. Run training first.")

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        self._is_tflite = self.model_path.suffix.lower() == ".tflite"

        if self._is_tflite:
            self._interpreter = _tf.lite.Interpreter(model_path=str(self.model_path))
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            self._model = None
        else:
            self._model = _keras.models.load_model(self.model_path)
            self._interpreter = None

        self.img_size = (config.img_height, config.img_width)

    def _predict_keras(self, img_array: np.ndarray) -> float:
        """Run inference through the full Keras model."""
        batch = np.expand_dims(img_array, axis=0)
        return float(self._model.predict(batch, verbose=0)[0][0])

    def _predict_tflite(self, img_array: np.ndarray) -> float:
        """Run inference through the TFLite interpreter."""
        batch = np.expand_dims(img_array, axis=0).astype(np.float32)
        self._interpreter.set_tensor(self._input_details[0]["index"], batch)
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output_details[0]["index"])
        return float(output.ravel()[0])

    def predict_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Predict whether an image is a cat or a dog.

        Returns a dict with keys:
            image, label, confidence_pct, dog_probability, time_ms
        """
        image_path = Path(image_path)

        # Load and validate (may raise ImageValidationError)
        img_array = _load_and_validate_image(image_path, self.img_size)

        # Inference with timing
        t0 = time.perf_counter()
        if self._is_tflite:
            prob = self._predict_tflite(img_array)
        else:
            prob = self._predict_keras(img_array)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        is_dog = prob >= 0.5
        label = CLASS_NAMES[1] if is_dog else CLASS_NAMES[0]
        confidence = prob if is_dog else (1.0 - prob)

        return {
            "image": image_path.name,
            "label": label,
            "confidence_pct": round(confidence * 100, 2),
            "dog_probability": round(prob, 4),
            "time_ms": round(elapsed_ms, 1),
        }

    def predict_folder(self, folder_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Predict all valid images in a folder, skipping invalid files.

        Returns a list of result dicts (same schema as predict_image),
        plus an 'error' key for files that failed validation.
        """
        folder = Path(folder_path)
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder}")

        results: List[Dict[str, Any]] = []
        files = sorted(
            [f for f in folder.iterdir() if f.is_file()],
            key=lambda p: p.name,
        )

        for f in files:
            if f.suffix.lower() not in VALID_EXTENSIONS:
                continue  # Silently skip non-image files (READMEs, etc.)
            try:
                result = self.predict_image(f)
                results.append(result)
            except ImageValidationError as e:
                results.append({"image": f.name, "label": "ERROR", "error": str(e)})

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m catdog.predict",
        description=(
            "Cat vs Dog Image Classifier — predict a single image or "
            "batch-classify every image in a folder."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m catdog.predict photo.jpg\n"
            "  python -m catdog.predict photos/ --csv results.csv\n"
            "  python -m catdog.predict photo.jpg --model models/transfer_model.tflite\n"
        ),
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to a single image file or a folder of images.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Path to a .keras or .tflite model file. "
            "Defaults to the best available model in models/."
        ),
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        metavar="OUTPUT.csv",
        help="(Folder mode) Write results to a CSV file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the table output (useful with --csv).",
    )
    return parser


def _print_single_result(result: Dict[str, Any]) -> None:
    """Pretty-print a single prediction result."""
    print()
    print("┌─────────────────────────────────────┐")
    print("│    Cat vs Dog Prediction Result      │")
    print("├─────────────────────────────────────┤")
    print(f"│  Image:      {result['image']:<23s}│")
    print(f"│  Prediction: {result['label']:<23s}│")
    print(f"│  Confidence: {result['confidence_pct']:<22.1f}%│")
    print(f"│  P(Dog):     {result['dog_probability']:<23.4f}│")
    print(f"│  Latency:    {result['time_ms']:<22.1f}ms│")
    print("└─────────────────────────────────────┘")


def _print_folder_results(results: List[Dict[str, Any]]) -> None:
    """Pretty-print batch results as a table."""
    print()
    header = f"{'Image':<20s} {'Label':<6s} {'Confidence':>10s} {'P(Dog)':>8s} {'Time':>8s}"
    print(header)
    print("─" * len(header))

    for r in results:
        if "error" in r:
            print(f"{r['image']:<20s} {'ERROR':<6s} {r['error']}")
        else:
            print(
                f"{r['image']:<20s} {r['label']:<6s} "
                f"{r['confidence_pct']:>9.1f}% "
                f"{r['dog_probability']:>8.4f} "
                f"{r['time_ms']:>6.1f}ms"
            )

    # Summary
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    cats = sum(1 for r in valid if r["label"] == "Cat")
    dogs = sum(1 for r in valid if r["label"] == "Dog")
    print("─" * len(header))
    print(
        f"Total: {len(valid)} classified ({cats} cats, {dogs} dogs)"
        + (f", {len(errors)} errors" if errors else "")
    )

    if valid:
        avg_time = sum(r["time_ms"] for r in valid) / len(valid)
        print(f"Average latency: {avg_time:.1f} ms/image")


def _write_csv(results: List[Dict[str, Any]], csv_path: str) -> None:
    """Write prediction results to a CSV file."""
    fieldnames = ["image", "label", "confidence_pct", "dog_probability", "time_ms", "error"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nResults saved to: {csv_path}")


def main():
    parser = _build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)

    try:
        predictor = CatDogPredictor(model_path=args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if input_path.is_dir():
        # --- Folder mode ---
        results = predictor.predict_folder(input_path)
        if not results:
            print(f"No valid images found in {input_path}")
            sys.exit(0)

        if not args.quiet:
            _print_folder_results(results)

        if args.csv:
            _write_csv(results, args.csv)
    else:
        # --- Single image mode ---
        try:
            result = predictor.predict_image(input_path)
            _print_single_result(result)
        except ImageValidationError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
