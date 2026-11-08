"""Ablation study harness comparing architectural and optimization enhancements."""

import json
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import tensorflow as tf

from catdog.callbacks import get_training_callbacks
from catdog.config import Config
from catdog.data import load_test_dataset, load_train_val_datasets, set_seed
from catdog.model import (
    build_baseline_model,
    build_gap_model,
    build_improved_cnn,
)


def evaluate_model_on_test(model: tf.keras.Model, config: Config) -> float:
    """Evaluate model on the 50-image test set against FCC answer key."""
    test_ds, _ = load_test_dataset(config)
    preds = model.predict(test_ds, verbose=0).ravel()
    binary_preds = [1 if p > 0.5 else 0 for p in preds]
    truth = config.fcc_test_answers
    correct = sum(p == t for p, t in zip(binary_preds, truth))
    return float(correct / len(truth))


def run_ablation_experiment(
    variant_name: str,
    model_builder_fn,
    callbacks: List[tf.keras.callbacks.Callback] = None,
    epochs: int = 20,
    config: Config = None,
) -> Dict[str, Any]:
    """Train and evaluate an individual ablation variant."""
    if config is None:
        config = Config()

    print("\n" + "=" * 65)
    print(f"RUNNING ABLATION: {variant_name}")
    print("=" * 65)

    set_seed(config.seed)
    train_ds, val_ds = load_train_val_datasets(config)

    model = model_builder_fn(config)
    param_count = model.count_params()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks or [],
        verbose=1,
    )

    # Final validation evaluation
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    test_acc = evaluate_model_on_test(model, config)

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    result = {
        "variant": variant_name,
        "parameters": int(param_count),
        "epochs_trained": len(history_dict["loss"]),
        "final_val_loss": float(val_loss),
        "final_val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "history": history_dict,
    }

    print(
        f"[{variant_name}] Params: {param_count:,} | Val Acc: {val_acc * 100:.2f}% | Test Acc: {test_acc * 100:.2f}%"
    )
    return result, model


def run_all_ablations(config: Config = None) -> Dict[str, Any]:
    """Execute complete ablation study across baseline, GAP, BatchNorm, and Callbacks."""
    if config is None:
        config = Config()

    # Load baseline evaluation if available to avoid redundant re-training
    baseline_eval_path = config.reports_dir / "baseline_evaluation.json"
    baseline_history_path = config.reports_dir / "baseline_history.json"

    if baseline_eval_path.exists() and baseline_history_path.exists():
        with open(baseline_eval_path, "r", encoding="utf-8") as f:
            b_eval = json.load(f)
        with open(baseline_history_path, "r", encoding="utf-8") as f:
            b_hist = json.load(f)

        baseline_result = {
            "variant": "0. Baseline CNN",
            "parameters": 19034177,
            "epochs_trained": len(b_hist["loss"]),
            "final_val_loss": b_eval["validation_loss"],
            "final_val_accuracy": b_eval["validation_accuracy"],
            "test_accuracy": b_eval["test_accuracy"],
            "history": b_hist,
        }
    else:
        baseline_result, _ = run_ablation_experiment(
            "0. Baseline CNN",
            build_baseline_model,
            epochs=config.epochs,
            config=config,
        )

    # Variant 1: + GlobalAveragePooling2D
    gap_result, _ = run_ablation_experiment(
        "1. + GlobalAveragePooling2D",
        build_gap_model,
        epochs=config.epochs,
        config=config,
    )

    # Variant 2: + BatchNormalization (fixed LR)
    def build_bn_model(cfg):
        return build_improved_cnn(cfg, use_batchnorm=True, l2_reg=1e-4, dropout_rate=0.3)

    bn_result, _ = run_ablation_experiment(
        "2. + BatchNormalization",
        build_bn_model,
        epochs=config.epochs,
        config=config,
    )

    # Variant 3: + LR Schedule & Callbacks (Final Improved CNN)
    callbacks = get_training_callbacks(
        config=config,
        experiment_name="improved_cnn",
        early_stopping_patience=6,
        reduce_lr_patience=3,
    )

    improved_result, best_model = run_ablation_experiment(
        "3. + Callbacks & LR Scheduling",
        build_bn_model,
        callbacks=callbacks,
        epochs=25,
        config=config,
    )

    # Save best improved model to models/improved_cnn.keras
    save_path = config.models_dir / "improved_cnn.keras"
    best_model.save(save_path)
    print(f"\n[Ablation] Saved final improved CNN to: {save_path}")

    # Compile ablation comparison table
    all_results = [baseline_result, gap_result, bn_result, improved_result]
    summary = {
        "ablation_results": [
            {
                "variant": r["variant"],
                "parameters": r["parameters"],
                "epochs_trained": r["epochs_trained"],
                "val_loss": round(r["final_val_loss"], 4),
                "val_accuracy": round(r["final_val_accuracy"] * 100, 2),
                "test_accuracy": round(r["test_accuracy"] * 100, 2),
            }
            for r in all_results
        ]
    }

    report_path = config.reports_dir / "stage2_ablation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Ablation] Summary report saved to: {report_path}")

    # Plot comparison of training curves
    plot_ablation_comparison(all_results, config.reports_dir / "stage2_ablation_curves.png")

    return summary


def plot_ablation_comparison(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Plot validation accuracy and validation loss curves across all ablation variants."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#94a3b8", "#3b82f6", "#f59e0b", "#10b981"]

    for res, color in zip(results, colors):
        val_acc = res["history"]["val_accuracy"]
        val_loss = res["history"]["val_loss"]
        epochs = range(1, len(val_acc) + 1)
        ax1.plot(epochs, val_acc, "o-", label=res["variant"], color=color, markersize=3)
        ax2.plot(epochs, val_loss, "s-", label=res["variant"], color=color, markersize=3)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Accuracy")
    ax1.set_title("Ablation: Validation Accuracy by Epoch")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Ablation: Validation Loss by Epoch")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[Ablation] Saved comparison curves to: {output_path}")


if __name__ == "__main__":
    run_all_ablations()
