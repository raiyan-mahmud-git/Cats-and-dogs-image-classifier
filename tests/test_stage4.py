"""Unit tests for evaluation metrics, confidence intervals, and Grad-CAM generation."""

import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import tensorflow as tf

from catdog.config import Config
from catdog.explainability import generate_gradcam_heatmap, superimpose_gradcam
from catdog.metrics import (
    compute_binary_metrics,
    compute_roc_auc,
    compute_wilson_confidence_interval,
)


def test_binary_metrics_exact_values():
    # 2 true positives, 1 false positive, 1 true negative, 1 false negative
    y_true = np.array([1, 1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.7, 0.1])  # preds at 0.5: 1, 1, 0, 1, 0

    m = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    cm = m["confusion_matrix"]
    assert cm["tp"] == 2
    assert cm["fp"] == 1
    assert cm["tn"] == 1
    assert cm["fn"] == 1
    assert m["accuracy"] == 0.6
    assert m["precision"] == round(2 / 3, 4)
    assert m["recall"] == round(2 / 3, 4)


def test_roc_auc_perfect_vs_reversed():
    y_true = np.array([1, 1, 0, 0])
    y_prob_perf = np.array([0.9, 0.8, 0.2, 0.1])
    _, _, auc_perf = compute_roc_auc(y_true, y_prob_perf)
    assert auc_perf == 1.0

    y_prob_rev = np.array([0.1, 0.2, 0.8, 0.9])
    _, _, auc_rev = compute_roc_auc(y_true, y_prob_rev)
    assert auc_rev == 0.0


def test_wilson_confidence_interval():
    # Margin of error on N=50 should be significantly larger than N=1000
    _, _, margin_50 = compute_wilson_confidence_interval(0.98, 50)
    _, _, margin_1000 = compute_wilson_confidence_interval(0.976, 1000)

    assert margin_50 > margin_1000
    # Statistical theory: ratio of margins should be ~ sqrt(1000 / 50) = sqrt(20) ~ 4.47
    assert 3.0 < (margin_50 / margin_1000) < 6.0


def test_gradcam_generation():
    config = Config()
    model_path = config.models_dir / "transfer_model_inference.keras"
    if not model_path.exists():
        model_path = config.models_dir / "transfer_model.keras"
    if not model_path.exists():
        return

    model = tf.keras.models.load_model(model_path)
    dummy_img = np.random.randint(0, 256, size=(150, 150, 3)).astype(np.float32)

    heatmap = generate_gradcam_heatmap(model, dummy_img)
    assert heatmap.ndim == 2
    assert np.min(heatmap) >= 0.0
    assert np.max(heatmap) <= 1.0

    overlay = superimpose_gradcam(dummy_img, heatmap)
    assert overlay.shape == (150, 150, 3)
    assert overlay.dtype == np.uint8
