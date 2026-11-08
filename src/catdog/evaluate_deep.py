"""Comprehensive Stage 4 evaluation suite: metrics, ROC-AUC, calibration, and Grad-CAM gallery."""

import json
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from catdog.config import Config
from catdog.data import load_test_dataset, load_train_val_datasets
from catdog.explainability import (
    generate_gradcam_heatmap,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_failure_gallery,
    plot_roc_curve,
    superimpose_gradcam,
)
from catdog.metrics import (
    compute_binary_metrics,
    compute_calibration,
    compute_roc_auc,
    compute_wilson_confidence_interval,
)


def run_comprehensive_evaluation(config: Config = None) -> Dict[str, Any]:
    """Execute complete Stage 4 diagnostic evaluation suite."""
    if config is None:
        config = Config()

    model_path = config.models_dir / "transfer_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Transfer model not found at {model_path}. Run Stage 3 first.")

    print("=" * 65)
    print("STAGE 4: COMPREHENSIVE EVALUATION & EXPLAINABILITY")
    print("=" * 65)

    model = tf.keras.models.load_model(model_path)

    # -----------------------------------------------------------------
    # 1. EVALUATION ON 1,000 VALIDATION SAMPLES (Statistically Robust)
    # -----------------------------------------------------------------
    print("\n[Evaluation] Evaluating on 1,000 validation images...")
    _, val_ds = load_train_val_datasets(config)

    val_images_list: List[np.ndarray] = []
    val_labels_list: List[int] = []
    for images, labels in val_ds:
        val_images_list.append(images.numpy())
        val_labels_list.append(labels.numpy())

    val_images = np.concatenate(val_images_list, axis=0)
    val_true = np.concatenate(val_labels_list, axis=0).astype(int)

    val_preds = model.predict(val_images, verbose=0).ravel()

    # Metrics
    val_metrics = compute_binary_metrics(val_true, val_preds)
    fpr, tpr, roc_auc = compute_roc_auc(val_true, val_preds)
    bin_confs, bin_accs, ece = compute_calibration(val_true, val_preds)

    # 95% Wilson Confidence Interval
    val_ci_low, val_ci_high, val_ci_margin = compute_wilson_confidence_interval(
        val_metrics["accuracy"],
        len(val_true),
    )

    # -----------------------------------------------------------------
    # 2. EVALUATION ON 50-IMAGE TEST SET & SIZING COMPARISON
    # -----------------------------------------------------------------
    print("[Evaluation] Evaluating on 50-image test set...")
    test_ds, test_files = load_test_dataset(config)

    test_images_list = []
    for batch in test_ds:
        test_images_list.append(batch.numpy())
    test_images = np.concatenate(test_images_list, axis=0)

    test_true = np.array(config.fcc_test_answers, dtype=int)
    test_preds = model.predict(test_images, verbose=0).ravel()
    test_metrics = compute_binary_metrics(test_true, test_preds)

    test_ci_low, test_ci_high, test_ci_margin = compute_wilson_confidence_interval(
        test_metrics["accuracy"],
        len(test_true),
    )

    # Print Statistical comparison
    print("\n" + "=" * 65)
    print("STATISTICAL RIGOR & SAMPLE SIZE ANALYSIS")
    print("=" * 65)
    print(f"Validation Set (N = {len(val_true)}):")
    print(f"  Accuracy:      {val_metrics['accuracy'] * 100:.2f}%")
    print(f"  95% CI:        [{val_ci_low * 100:.2f}%, {val_ci_high * 100:.2f}%] (+/- {val_ci_margin * 100:.2f}%)")
    print(f"  Precision:     {val_metrics['precision'] * 100:.2f}%")
    print(f"  Recall:        {val_metrics['recall'] * 100:.2f}%")
    print(f"  Specificity:   {val_metrics['specificity'] * 100:.2f}%")
    print(f"  F1-Score:      {val_metrics['f1_score'] * 100:.2f}%")
    print(f"  ROC-AUC:       {roc_auc:.4f}")
    print(f"  Expected Calibration Error (ECE): {ece:.4f}")

    print(f"\nTest Set (N = {len(test_true)}):")
    print(f"  Accuracy:      {test_metrics['accuracy'] * 100:.2f}% ({int(test_metrics['accuracy'] * len(test_true))}/{len(test_true)})")
    print(f"  95% CI:        [{test_ci_low * 100:.2f}%, {test_ci_high * 100:.2f}%] (+/- {test_ci_margin * 100:.2f}%)")
    print(f"  Notice: The 95% CI is {test_ci_margin / val_ci_margin:.1f}x WIDER on 50 images than on 1,000 images!")
    print("=" * 65)

    # -----------------------------------------------------------------
    # 3. SAVE DIAGNOSTIC PLOTS
    # -----------------------------------------------------------------
    cm_path = config.reports_dir / "confusion_matrix.png"
    plot_confusion_matrix(val_metrics["confusion_matrix"], cm_path)

    roc_path = config.reports_dir / "roc_curve.png"
    plot_roc_curve(fpr, tpr, roc_auc, roc_path)

    calib_path = config.reports_dir / "calibration_curve.png"
    plot_calibration_curve(bin_confs, bin_accs, ece, calib_path)

    # -----------------------------------------------------------------
    # 4. FAILURE ANALYSIS & GRAD-CAM GALLERY
    # -----------------------------------------------------------------
    print("\n[Explainability] Generating Grad-CAM failure case analysis...")
    test_binary = (test_preds >= 0.5).astype(int)
    wrong_indices = np.where(test_binary != test_true)[0]

    wrong_cases = []
    for idx in wrong_indices:
        raw_img = test_images[idx].astype(np.uint8)
        heatmap = generate_gradcam_heatmap(model, test_images[idx])
        overlay = superimpose_gradcam(raw_img, heatmap)
        p = float(test_preds[idx])
        conf = p if p > 0.5 else (1.0 - p)

        wrong_cases.append({
            "index": int(idx + 1),
            "filename": test_files[idx].name,
            "true_label": "Dog" if test_true[idx] == 1 else "Cat",
            "pred_label": "Dog" if test_binary[idx] == 1 else "Cat",
            "probability": round(p, 4),
            "confidence": round(conf * 100, 2),
            "original_image": raw_img,
            "gradcam_overlay": overlay,
        })

    gallery_path = config.reports_dir / "failure_analysis_gallery.png"
    plot_failure_gallery(wrong_cases, gallery_path)

    # Also generate a representative success Grad-CAM comparison
    sample_success_path = config.reports_dir / "gradcam_sample_analysis.png"
    plot_sample_gradcam_comparisons(model, test_images, test_true, test_files, sample_success_path)

    # -----------------------------------------------------------------
    # 5. CONSOLIDATED JSON REPORT
    # -----------------------------------------------------------------
    report = {
        "validation_set": {
            "samples": len(val_true),
            "metrics": val_metrics,
            "confidence_interval_95": {
                "low": val_ci_low,
                "high": val_ci_high,
                "margin_of_error": val_ci_margin,
            },
            "roc_auc": roc_auc,
            "expected_calibration_error": ece,
        },
        "test_set": {
            "samples": len(test_true),
            "metrics": test_metrics,
            "confidence_interval_95": {
                "low": test_ci_low,
                "high": test_ci_high,
                "margin_of_error": test_ci_margin,
            },
        },
        "failure_analysis": [
            {
                "index": c["index"],
                "filename": c["filename"],
                "true_label": c["true_label"],
                "pred_label": c["pred_label"],
                "probability": c["probability"],
                "confidence": c["confidence"],
            }
            for c in wrong_cases
        ],
    }

    report_path = config.reports_dir / "stage4_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Report] Stage 4 Full Diagnostic Report saved to: {report_path}")

    return report


def plot_sample_gradcam_comparisons(
    model: tf.keras.Model,
    images: np.ndarray,
    labels: np.ndarray,
    files: List[Path],
    output_path: Path,
) -> None:
    """Plot sample true Cat and true Dog Grad-CAM overlays demonstrating correct focus."""
    # Find one confident cat (true=0) and one confident dog (true=1)
    cat_idx = np.where(labels == 0)[0][0]
    dog_idx = np.where(labels == 1)[0][0]

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))

    for row, (idx, name) in enumerate([(cat_idx, "Cat"), (dog_idx, "Dog")]):
        raw_img = images[idx].astype(np.uint8)
        heatmap = generate_gradcam_heatmap(model, images[idx])
        overlay = superimpose_gradcam(raw_img, heatmap)

        axes[row, 0].imshow(raw_img)
        axes[row, 0].set_title(f"Original {name} ({files[idx].name})", fontsize=11)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(f"Grad-CAM Attention ({name})", fontsize=11)
        axes[row, 1].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_comprehensive_evaluation()
