"""Comprehensive statistical evaluation metrics, ROC-AUC, calibration, and confidence intervals."""

import math
from typing import Dict, Any, List, Tuple
import numpy as np


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute standard binary classification metrics from ground truth and probabilities.
    
    Design explanation:
    Beyond simple accuracy, balanced evaluation requires precision, recall, specificity,
    and F1-score to detect asymmetric class performance and false alarm rates.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
        "f1_score": round(float(f1), 4),
        "total_samples": total,
    }


def compute_roc_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_thresholds: int = 200,
) -> Tuple[List[float], List[float], float]:
    """Compute ROC curve coordinates and Area Under Curve (AUC) via trapezoidal integration.
    
    Design explanation:
    The ROC curve illustrates the diagnostic tradeoff between sensitivity and (1 - specificity)
    across all classification thresholds. AUC provides a threshold-independent measure of ranking quality.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    fpr_list: List[float] = []
    tpr_list: List[float] = []

    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)

    for thresh in sorted(thresholds, reverse=True):
        y_pred = (y_prob >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tpr = tp / positives if positives > 0 else 0.0
        fpr = fp / negatives if negatives > 0 else 0.0
        fpr_list.append(float(fpr))
        tpr_list.append(float(tpr))

    # Add endpoints (0, 0) and (1, 1)
    fpr_arr = np.array([0.0] + fpr_list + [1.0])
    tpr_arr = np.array([0.0] + tpr_list + [1.0])

    # Sort by FPR for monotonic integration
    sort_idx = np.argsort(fpr_arr)
    fpr_sorted = fpr_arr[sort_idx]
    tpr_sorted = tpr_arr[sort_idx]

    # Trapezoidal rule for AUC (compatible with numpy 1.x np.trapz and numpy 2.x np.trapezoid)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(trapz_fn(tpr_sorted, fpr_sorted))

    return fpr_sorted.tolist(), tpr_sorted.tolist(), round(auc, 4)


def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[List[float], List[float], float]:
    """Calculate reliability diagram data and Expected Calibration Error (ECE).
    
    Design explanation:
    A well-calibrated classifier outputs probabilities that match empirical frequencies
    (e.g., when the model predicts 0.9 confidence, it should be correct 90% of the time).
    ECE quantifies the weighted discrepancy between confidence and observed accuracy.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confs: List[float] = []
    bin_accs: List[float] = []
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= low) & (y_prob < high if i < n_bins - 1 else y_prob <= high)
        bin_count = np.sum(mask)

        if bin_count > 0:
            mean_conf = float(np.mean(y_prob[mask]))
            empirical_acc = float(np.mean(y_true[mask]))
            bin_confs.append(round(mean_conf, 4))
            bin_accs.append(round(empirical_acc, 4))
            ece += (bin_count / total) * abs(empirical_acc - mean_conf)
        else:
            bin_confs.append(round((low + high) / 2.0, 4))
            bin_accs.append(0.0)

    return bin_confs, bin_accs, round(float(ece), 4)


def compute_wilson_confidence_interval(
    accuracy: float,
    n_samples: int,
    confidence_level: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute the Wilson score confidence interval for a binomial proportion.
    
    Design explanation:
    Unlike the naive Wald interval which fails near 0 and 1, the Wilson score interval
    accurately reflects sample size uncertainty. Comparing 50 test images vs 1,000 validation
    images reveals that small test sets have wide uncertainty bands (+/- 4-6%), proving why
    a 2-image difference cannot validate model superiority.
    """
    # z = 1.96 for 95% confidence
    z = 1.95996 if confidence_level == 0.95 else 2.576
    p = accuracy
    n = n_samples

    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) / n) + (z**2) / (4 * n**2))) / denom

    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    half_width = (high - low) / 2.0

    return round(float(low), 4), round(float(high), 4), round(float(half_width), 4)
