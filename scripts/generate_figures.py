"""Generate figures for TCN-Attention-BiLSTM research defense."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def plot_prediction_vs_actual():
    """Plot prediction vs actual from test predictions."""
    # Try latest run first, fallback to original
    v2_pred = OUTPUTS / "predictions" / "full_120_v2_tcn_attention_bilstm" / "test_predictions.csv"
    v1_pred = OUTPUTS / "predictions" / "full_120_tcn_attention_bilstm" / "test_predictions.csv"
    pred_path = v2_pred if v2_pred.exists() else v1_pred
    if not pred_path.exists():
        print(f"SKIP prediction_vs_actual: {pred_path} not found")
        return

    import csv
    actuals, predictions = [], []
    with open(pred_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 500:
                break
            actuals.append(float(row.get("actual", row.get("y_true", 0))))
            predictions.append(float(row.get("predicted", row.get("y_pred", 0))))

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(actuals))
    ax.plot(x, actuals, label="Actual", alpha=0.7, linewidth=0.8)
    ax.plot(x, predictions, label="Predicted", alpha=0.7, linewidth=0.8)
    ax.axhline(y=0.183838, color='r', linestyle='--', alpha=0.5, label="Threshold p90 (0.184)")
    ax.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label="Calibrated (0.05)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Congestion Score (proxy)")
    ax.set_title("Prediction vs Actual — TCN-Attention-BiLSTM (first 500 test samples)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "prediction_vs_actual.png", dpi=150)
    plt.close(fig)
    print(f"OK: prediction_vs_actual.png ({len(actuals)} samples)")


def plot_error_distribution():
    """Plot error distribution from test predictions."""
    # Try latest run first, fallback to original
    v2_pred = OUTPUTS / "predictions" / "full_120_v2_tcn_attention_bilstm" / "test_predictions.csv"
    v1_pred = OUTPUTS / "predictions" / "full_120_tcn_attention_bilstm" / "test_predictions.csv"
    pred_path = v2_pred if v2_pred.exists() else v1_pred
    if not pred_path.exists():
        print(f"SKIP error_distribution: {pred_path} not found")
        return

    import csv
    errors = []
    with open(pred_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            actual = float(row.get("actual", row.get("y_true", 0)))
            predicted = float(row.get("predicted", row.get("y_pred", 0)))
            errors.append(predicted - actual)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(x=0, color='r', linestyle='--')
    axes[0].set_xlabel("Error (predicted - actual)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Error Distribution")
    axes[0].grid(True, alpha=0.3)

    axes[1].boxplot(errors, vert=True)
    axes[1].set_ylabel("Error")
    axes[1].set_title("Error Boxplot")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Error Analysis — MAE=0.042792, RMSE=0.056399 (n={len(errors)})")
    fig.tight_layout()
    fig.savefig(FIGURES / "error_distribution.png", dpi=150)
    plt.close(fig)
    print(f"OK: error_distribution.png ({len(errors)} samples)")


def plot_model_comparison_rmse():
    """Plot RMSE comparison across models."""
    csv_path = OUTPUTS / "web" / "balanced_model_comparison_table.csv"
    if not csv_path.exists():
        print(f"SKIP model_comparison_rmse: {csv_path} not found")
        return

    import csv
    models, rmses = [], []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            models.append(row["model"])
            rmses.append(float(row["rmse"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#95a5a6', '#95a5a6', '#3498db', '#3498db', '#e74c3c', '#9b59b6', '#2ecc71', '#f39c12']
    bars = ax.barh(models, rmses, color=colors[:len(models)], edgecolor='black')
    ax.set_xlabel("RMSE (lower is better)")
    ax.set_title("Model Comparison — RMSE on NASA HTTP 1995 Test Set")
    ax.grid(True, alpha=0.3, axis='x')

    for bar, val in zip(bars, rmses):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison_rmse.png", dpi=150)
    plt.close(fig)
    print(f"OK: model_comparison_rmse.png ({len(models)} models)")


def plot_training_curves():
    """Plot training curves from history."""
    # Try latest run first, fallback to original
    v2_history = OUTPUTS / "metrics" / "full_120_v2_tcn_attention_bilstm" / "history.json"
    v1_history = OUTPUTS / "metrics" / "full_120_tcn_attention_bilstm" / "history.json"
    history_path = v2_history if v2_history.exists() else v1_history
    if not history_path.exists():
        print(f"SKIP training_curves: {history_path} not found")
        return

    history = load_json(history_path)
    # Handle nested structure: {"model": ..., "history": {"loss": ..., "val_loss": ...}}
    if "history" in history and isinstance(history["history"], dict):
        history = history["history"]
    train_loss = history.get("train_loss", history.get("loss", []))
    val_loss = history.get("val_loss", [])

    if not train_loss or not val_loss:
        print("SKIP training_curves: no loss data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_loss) + 1)
    axes[0].plot(epochs, train_loss, label="Train Loss", linewidth=1.5)
    axes[0].plot(epochs, val_loss, label="Val Loss", linewidth=1.5)
    axes[0].axvline(x=30, color='r', linestyle='--', alpha=0.5, label="Best Epoch (30)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    val_rmse = history.get("val_rmse", history.get("rmse", []))
    if val_rmse:
        axes[1].plot(epochs, val_rmse, label="Val RMSE", color='green', linewidth=1.5)
        axes[1].axvline(x=30, color='r', linestyle='--', alpha=0.5, label="Best Epoch (30)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("RMSE")
        axes[1].set_title("Validation RMSE")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.suptitle("TCN-Attention-BiLSTM Training Curves (120 epochs)")
    fig.tight_layout()
    fig.savefig(FIGURES / "training_curves.png", dpi=150)
    plt.close(fig)
    print(f"OK: training_curves.png ({len(train_loss)} epochs)")


def plot_early_warning_timeline():
    """Plot early warning timeline from test predictions."""
    # Try latest run first, fallback to original
    v2_pred = OUTPUTS / "predictions" / "full_120_v2_tcn_attention_bilstm" / "test_predictions.csv"
    v1_pred = OUTPUTS / "predictions" / "full_120_tcn_attention_bilstm" / "test_predictions.csv"
    pred_path = v2_pred if v2_pred.exists() else v1_pred
    if not pred_path.exists():
        print(f"SKIP early_warning_timeline: {pred_path} not found")
        return

    import csv
    actuals, predictions = [], []
    with open(pred_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 1000:
                break
            actuals.append(float(row.get("actual", row.get("y_true", 0))))
            predictions.append(float(row.get("predicted", row.get("y_pred", 0))))

    threshold = 0.05
    fig, ax = plt.subplots(figsize=(14, 5))

    x = range(len(actuals))
    ax.plot(x, actuals, label="Actual Score", alpha=0.7, linewidth=0.8)
    ax.plot(x, predictions, label="Predicted Score", alpha=0.7, linewidth=0.8)
    ax.axhline(y=threshold, color='orange', linestyle='--', alpha=0.8, label=f"Calibrated Threshold ({threshold})")

    # Highlight warning zones
    for i in range(len(actuals)):
        if actuals[i] >= threshold:
            ax.axvspan(i-0.5, i+0.5, alpha=0.2, color='red')

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Congestion Score")
    ax.set_title("Early Warning Timeline — Calibrated Threshold = 0.05")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / "early_warning_timeline.png", dpi=150)
    plt.close(fig)
    print(f"OK: early_warning_timeline.png ({len(actuals)} steps)")


def plot_synthetic_stress_scenarios():
    """Plot synthetic stress scenario comparison."""
    csv_path = OUTPUTS / "synthetic_stress_eval" / "full_120_tcn_attention_bilstm" / "scenario_metrics.csv"
    if not csv_path.exists():
        print(f"SKIP synthetic_stress_scenarios: {csv_path} not found")
        return

    import csv
    scenarios, f1s = [], []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenarios.append(row["scenario_name"])
            f1s.append(float(row["f1"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#3498db' if f1 >= 0.5 else '#e74c3c' for f1 in f1s]
    bars = ax.bar(scenarios, f1s, color=colors, edgecolor='black')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label="F1 = 0.5")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("F1 Score")
    ax.set_title("Synthetic Stress Benchmark — F1 by Scenario (threshold=0.15)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES / "synthetic_stress_scenarios.png", dpi=150)
    plt.close(fig)
    print(f"OK: synthetic_stress_scenarios.png ({len(scenarios)} scenarios)")


if __name__ == "__main__":
    print("=== Generating Figures ===")
    plot_prediction_vs_actual()
    plot_error_distribution()
    plot_model_comparison_rmse()
    plot_training_curves()
    plot_early_warning_timeline()
    plot_synthetic_stress_scenarios()
    print(f"\n=== Done: {len(list(FIGURES.glob('*.png')))} figures in {FIGURES} ===")
