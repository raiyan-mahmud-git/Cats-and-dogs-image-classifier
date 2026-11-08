"""Evaluation module for honest validation and test benchmark reporting."""

import json
from typing import Dict, Any, Optional

import tensorflow as tf

from catdog.config import Config
from catdog.data import load_test_dataset, load_train_val_datasets


def evaluate_baseline(
    config: Optional[Config] = None,
    model: Optional[tf.keras.Model] = None,
) -> Dict[str, Any]:
    """Evaluate baseline model on both validation data and the 50-image test set."""
    if config is None:
        config = Config()

    if model is None:
        model_path = config.models_dir / "baseline_model.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}. Run training first.")
        print(f"[Evaluation] Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)

    # 1. Validation evaluation
    _, val_ds = load_train_val_datasets(config)
    val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
    print("\n" + "=" * 60)
    print("VALIDATION SET RESULTS")
    print(f"Validation Loss:     {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    print("=" * 60)

    # 2. Test set evaluation
    test_ds, test_files = load_test_dataset(config)
    raw_predictions = model.predict(test_ds, verbose=0)

    # Flatten (N, 1) to (N,) to avoid iterating over single-element arrays
    probabilities = raw_predictions.ravel().tolist()
    binary_predictions = [1 if p > 0.5 else 0 for p in probabilities]
    ground_truth = config.fcc_test_answers

    assert len(probabilities) == len(ground_truth) == 50, (
        f"Mismatch: {len(probabilities)} predictions vs {len(ground_truth)} answers"
    )

    # 3. Filename order and alignment verification
    print("\n50-IMAGE TEST SET PREDICTIONS & GROUND TRUTH ALIGNMENT:")
    print("-" * 75)
    print(
        f"{'Idx':<4} | {'Filename':<10} | {'Prob(Dog)':<10} | {'Pred':<5} | {'Truth':<5} | {'Match':<5}"
    )
    print("-" * 75)

    test_results = []
    correct_count = 0
    for idx, (p, pred, truth, file_path) in enumerate(
        zip(probabilities, binary_predictions, ground_truth, test_files)
    ):
        is_match = pred == truth
        if is_match:
            correct_count += 1
        label_str = "Dog" if pred == 1 else "Cat"
        truth_str = "Dog" if truth == 1 else "Cat"
        print(
            f"{idx + 1:<4} | {file_path.name:<10} | {p:<10.4f} | {label_str:<5} | {truth_str:<5} | {str(is_match):<5}"
        )
        test_results.append(
            {
                "index": idx + 1,
                "filename": file_path.name,
                "dog_probability": float(p),
                "predicted_class": int(pred),
                "predicted_label": label_str,
                "ground_truth": int(truth),
                "ground_truth_label": truth_str,
                "correct": bool(is_match),
            }
        )

    test_accuracy = correct_count / len(ground_truth)
    print("-" * 75)
    print(f"Total Test Images: {len(ground_truth)}")
    print(f"Correct:           {correct_count} / {len(ground_truth)}")
    print(f"Test Accuracy:     {test_accuracy * 100:.2f}%")
    print("=" * 75)

    # 4. Save evaluation report
    report = {
        "validation_loss": float(val_loss),
        "validation_accuracy": float(val_accuracy),
        "test_total": len(ground_truth),
        "test_correct": int(correct_count),
        "test_accuracy": float(test_accuracy),
        "freecodecamp_pass_mark": 0.63,
        "passed": bool(test_accuracy >= 0.63),
        "per_image_results": test_results,
    }

    report_path = config.reports_dir / "baseline_evaluation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[Evaluation] Report saved to: {report_path}")

    return report


if __name__ == "__main__":
    evaluate_baseline()
