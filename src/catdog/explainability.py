"""Explainability module: Grad-CAM heatmaps, failure analysis, and diagnostic visualizations."""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def generate_gradcam_heatmap(
    model: tf.keras.Model,
    img_array: np.ndarray,
) -> np.ndarray:
    """Generate Grad-CAM heatmap for an image using MobileNetV2's final convolutional layer.
    
    Design explanation:
    Grad-CAM uses the gradients of the target class flowing into the final convolutional
    layer (`out_relu`) to produce a coarse 2D localization map highlighting the salient
    regions of the image that drove the decision.
    """
    if len(img_array.shape) == 3:
        img_batch = np.expand_dims(img_array, axis=0)
    else:
        img_batch = img_array

    base_model = model.get_layer("mobilenetv2_base")
    last_conv_layer = base_model.get_layer("out_relu")

    # Build head model mapping base output to final predictions
    head_input = tf.keras.layers.Input(shape=base_model.output_shape[1:])
    x = head_input
    for layer_name in ["gap", "dense_head", "dropout", "predictions"]:
        x = model.get_layer(layer_name)(x)
    head_model = tf.keras.Model(head_input, x)

    # Feature extractor mapping base input to [conv_output, base_output]
    conv_model = tf.keras.Model(
        base_model.inputs,
        [last_conv_layer.output, base_model.output],
    )

    # Apply in-graph rescaling
    rescaling = model.get_layer("mobilenet_rescaling")
    rescaled = rescaling(img_batch)

    # Compute gradients of class probability w.r.t. conv outputs
    with tf.GradientTape() as tape:
        conv_outs, _ = conv_model(rescaled)
        tape.watch(conv_outs)
        preds = head_model(conv_outs)
        prob = preds[0][0]

    grads = tape.gradient(prob, conv_outs)

    # Pool gradients over spatial dimensions (Global Average Pooling of gradients)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight channels by their gradient importance
    conv_outs = conv_outs[0]
    heatmap = conv_outs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU activation on weighted feature map (only positive influence features)
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def superimpose_gradcam(
    img_array: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap_name: str = "jet",
) -> np.ndarray:
    """Superimpose Grad-CAM heatmap over an RGB image.
    
    Design explanation:
    We upsample the low-resolution conv heatmap to image dimensions, apply a perceptually
    vibrant colormap, and alpha-blend it with the original pixels to reveal exact feature focus.
    """
    img_uint8 = np.asarray(img_array, dtype=np.uint8)
    h, w = img_uint8.shape[:2]

    # Resize heatmap to match image dimensions
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        [h, w],
        method="bilinear",
    ).numpy()
    heatmap_resized = np.squeeze(heatmap_resized)

    # Apply colormap
    cmap = plt.colormaps[colormap_name]
    colored_heatmap = cmap(heatmap_resized)[:, :, :3]  # Drop alpha
    colored_heatmap = np.uint8(255 * colored_heatmap)

    # Blend
    superimposed = (1.0 - alpha) * img_uint8 + alpha * colored_heatmap
    return np.clip(superimposed, 0, 255).astype(np.uint8)


def plot_confusion_matrix(
    cm: Dict[str, int],
    output_path: Path,
) -> None:
    """Plot an annotated confusion matrix visualization."""
    matrix = np.array([
        [cm["tn"], cm["fp"]],
        [cm["fn"], cm["tp"]],
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(matrix, cmap="Blues", alpha=0.8)

    classes = ["Cat (0)", "Dog (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)
    ax.set_title("Confusion Matrix", fontsize=14, pad=15)

    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            label = f"{val}\n({(val / np.sum(matrix)) * 100:.1f}%)"
            color = "white" if val > np.max(matrix) / 2 else "black"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    fig.colorbar(cax)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(
    fpr: List[float],
    tpr: List[float],
    roc_auc: float,
    output_path: Path,
) -> None:
    """Plot ROC curve with chance line and area under the curve."""
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", lw=1.5, label="Random Guess (AUC = 0.50)")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#2563eb")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_calibration_curve(
    bin_confs: List[float],
    bin_accs: List[float],
    ece: float,
    output_path: Path,
) -> None:
    """Plot Reliability Diagram (Calibration Curve) with Expected Calibration Error."""
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.plot(bin_confs, bin_accs, "s-", color="#16a34a", lw=2, markersize=7, label=f"Model (ECE = {ece:.4f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Mean Predicted Probability (Confidence)", fontsize=11)
    ax.set_ylabel("Empirical Accuracy (Fraction of Positives)", fontsize=11)
    ax.set_title("Calibration Curve (Reliability Diagram)", fontsize=13)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_failure_gallery(
    wrong_cases: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot a gallery of the most confidently wrong predictions with Grad-CAM heatmaps.
    
    Design explanation:
    Pairing the original image with its Grad-CAM activation visually reveals why the model
    failed (e.g. background textures, cropped features, or confusing pose angles).
    """
    if not wrong_cases:
        return

    n_cases = min(len(wrong_cases), 4)
    fig, axes = plt.subplots(n_cases, 2, figsize=(8, 3.5 * n_cases))
    if n_cases == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n_cases):
        case = wrong_cases[i]
        orig_img = case["original_image"]
        overlay_img = case["gradcam_overlay"]

        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(
            f"Image: {case['filename']}\nTrue: {case['true_label']} | Pred: {case['pred_label']}",
            fontsize=10,
        )
        axes[i, 0].axis("off")

        axes[i, 1].imshow(overlay_img)
        axes[i, 1].set_title(
            f"Grad-CAM Heatmap\nConfidence: {case['confidence']:.2f}%",
            fontsize=10,
        )
        axes[i, 1].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
