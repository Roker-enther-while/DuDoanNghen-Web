"""Rolling-origin cross-validation for time series models.

Instead of a single chronological train/val/test split, this script:
1. Creates multiple overlapping train/val/test splits along the time axis
2. Trains and evaluates each model on each fold
3. Reports mean +/- std across folds
4. Performs statistical tests between proposed model and baselines

Usage:
    python scripts/run_rolling_cv.py --data data/processed/nasa_http_3m/windows/windows_fp16.npz --n-folds 5
    python scripts/run_rolling_cv.py --data data/processed/nasa_http_3m/windows/windows_fp16.npz --n-folds 5 --models tcn_attention_bilstm lstm gru tcn
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.data_loader import get_data_summary, load_window_data, validate_window_data
from src.training.trainer import load_training_config, train_and_evaluate


DEFAULT_MODELS = ["naive_last_value", "moving_average", "lstm", "gru", "tcn", "tcn_lstm", "tcn_attention_bilstm"]


def create_rolling_folds(n_total: int, n_folds: int, min_train: int = 1000, min_val: int = 500, min_test: int = 500) -> list[dict]:
    """Create rolling-origin folds with chronological ordering.

    Each fold has:
    - train: first `train_end` samples
    - val: next `val_end - train_end` samples
    - test: next `test_end - val_end` samples

    The origin shifts forward for each fold.
    """
    folds = []
    # Each fold: train uses increasing portions, val/test are the next windows
    # Fold 0: train=0..A, val=A..B, test=B..C
    # Fold 1: train=0..A+d, val=A+d..B+d, test=B+d..C+d
    # etc.

    # Calculate step size between origins
    available = n_total
    if available < min_train + min_val + min_test:
        raise ValueError(f"Not enough data for even 1 fold: {available} < {min_train + min_val + min_test}")

    # For rolling origin: train grows, val/test shift forward
    # Use equal-sized val and test windows, train grows from min_train to near the end
    test_window = min_test
    val_window = min_val
    total_val_test = val_window + test_window

    # Calculate train sizes for each fold
    max_train = n_total - total_val_test
    if max_train < min_train:
        max_train = n_total - total_val_test

    train_sizes = np.linspace(min_train, max_train, n_folds, dtype=int)

    for fold_idx, train_end in enumerate(train_sizes):
        val_start = train_end
        val_end = val_start + val_window
        test_start = val_end
        test_end = test_start + test_window

        if test_end > n_total:
            test_end = n_total
            test_start = max(test_end - test_window, val_end)
            if test_start <= val_start:
                continue

        folds.append({
            "fold": fold_idx,
            "train_indices": list(range(0, train_end)),
            "val_indices": list(range(val_start, val_end)),
            "test_indices": list(range(test_start, test_end)),
            "train_size": train_end,
            "val_size": val_end - val_start,
            "test_size": test_end - test_start,
        })

    return folds


def run_rolling_cv(
    data_path: str,
    models: list[str],
    n_folds: int = 5,
    epochs: int = 60,
    seed: int = 42,
    output_dir: str = "outputs/rolling_cv",
) -> dict:
    """Run rolling-origin cross-validation for all models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load full data
    raw_data = load_window_data(data_path)
    validate_window_data(raw_data)
    summary = get_data_summary(raw_data)

    # Stack all splits for rolling origin
    X_all = np.concatenate([raw_data["X_train"], raw_data["X_val"], raw_data["X_test"]], axis=0)
    y_all = np.concatenate([raw_data["y_train"], raw_data["y_val"], raw_data["y_test"]], axis=0)
    n_total = len(X_all)
    print(f"Total samples: {n_total}")
    print(f"Data shape: X={X_all.shape}, y={y_all.shape}")

    # Create folds
    folds = create_rolling_folds(n_total, n_folds)
    print(f"Created {len(folds)} folds")
    for f in folds:
        print(f"  Fold {f['fold']}: train={f['train_size']}, val={f['val_size']}, test={f['test_size']}")

    # Load default config
    base_config = load_training_config("configs/training/ablation/base.yaml")

    # Run all models on all folds
    all_results = []
    for model_name in models:
        for fold_info in folds:
            print(f"\n{'='*50}")
            print(f"Model: {model_name}, Fold: {fold_info['fold']}")
            print(f"{'='*50}")

            # Create fold-specific data
            fold_data = {
                "X_train": X_all[fold_info["train_indices"]],
                "y_train": y_all[fold_info["train_indices"]],
                "X_val": X_all[fold_info["val_indices"]],
                "y_val": y_all[fold_info["val_indices"]],
                "X_test": X_all[fold_info["test_indices"]],
                "y_test": y_all[fold_info["test_indices"]],
                "_path": data_path,
                "feature_columns": list(raw_data.get("feature_columns", [])),
                "target_column": str(raw_data.get("target_column", "target_next_congestion_score")),
            }

            # Save fold data as temporary NPZ
            fold_path = output_dir / f"fold_{fold_info['fold']}_data.npz"
            np.savez(fold_path, **{k: v for k, v in fold_data.items() if not k.startswith("_")})

            # Configure training
            config = copy.deepcopy(base_config)
            config["seed"] = seed
            config["epochs"] = epochs
            config["artifact_name"] = f"rcv_{model_name}_fold{fold_info['fold']}"
            config["output_tag"] = f"rcv_{model_name}_fold{fold_info['fold']}"

            try:
                result = train_and_evaluate(model_name, str(fold_path), config)
                all_results.append({
                    "model": model_name,
                    "fold": fold_info["fold"],
                    "status": result["status"],
                    "mae": result["metrics"]["mae"],
                    "rmse": result["metrics"]["rmse"],
                    "r2": result["metrics"]["r2"],
                    "f1": result["alert_metrics"]["f1"],
                    "train_time_seconds": result["train_time_seconds"],
                })
            except Exception as exc:
                print(f"  FAILED: {exc}")
                all_results.append({
                    "model": model_name,
                    "fold": fold_info["fold"],
                    "status": "failed",
                    "error": str(exc),
                })

            # Clean up fold data
            fold_path.unlink(missing_ok=True)

    # Save raw results
    (output_dir / "raw_results.json").write_text(
        json.dumps(all_results, indent=2, allow_nan=False), encoding="utf-8"
    )

    # Compute summary per model
    model_summary = {}
    for model_name in models:
        model_results = [r for r in all_results if r["model"] == model_name and r["status"] == "success"]
        if not model_results:
            model_summary[model_name] = {"status": "no_successful_folds"}
            continue
        metrics = {}
        for metric in ["mae", "rmse", "r2", "f1"]:
            values = [r[metric] for r in model_results]
            metrics[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": [float(x) for x in values],
            }
        model_summary[model_name] = {
            "n_folds": len(model_results),
            "metrics": metrics,
        }

    # Statistical tests: proposed vs each baseline
    proposed = "tcn_attention_bilstm"
    stat_tests = []
    if proposed in model_summary and model_summary[proposed].get("status") != "no_successful_folds":
        proposed_r2 = model_summary[proposed]["metrics"]["r2"]["values"]
        proposed_mae = model_summary[proposed]["metrics"]["mae"]["values"]
        for model_name in models:
            if model_name == proposed or model_name not in model_summary:
                continue
            if model_summary[model_name].get("status") == "no_successful_folds":
                continue
            other_r2 = model_summary[model_name]["metrics"]["r2"]["values"]
            other_mae = model_summary[model_name]["metrics"]["mae"]["values"]
            n = min(len(proposed_r2), len(other_r2))
            if n < 3:
                continue
            try:
                r2_stat, r2_p = sp_stats.wilcoxon(proposed_r2[:n], other_r2[:n])
            except ValueError:
                r2_p = 1.0
            try:
                mae_stat, mae_p = sp_stats.wilcoxon(proposed_mae[:n], other_mae[:n])
            except ValueError:
                mae_p = 1.0

            diff = np.array(proposed_r2[:n]) - np.array(other_r2[:n])
            cohens_d = float(np.mean(diff) / max(np.std(diff, ddof=1), 1e-10))

            stat_tests.append({
                "vs_model": model_name,
                "r2_proposed_mean": model_summary[proposed]["metrics"]["r2"]["mean"],
                "r2_other_mean": model_summary[model_name]["metrics"]["r2"]["mean"],
                "r2_diff": round(model_summary[proposed]["metrics"]["r2"]["mean"] - model_summary[model_name]["metrics"]["r2"]["mean"], 6),
                "wilcoxon_p": round(r2_p, 6),
                "cohens_d": round(cohens_d, 4),
                "significant": bool(r2_p < 0.05),
            })

    # Save summary
    (output_dir / "cv_summary.json").write_text(
        json.dumps({"model_summary": model_summary, "statistical_tests": stat_tests}, indent=2, allow_nan=False),
        encoding="utf-8"
    )

    # Generate markdown
    md_lines = [
        "# Rolling-Origin Cross-Validation Results",
        "",
        f"**Folds**: {len(folds)}",
        f"**Models**: {', '.join(models)}",
        f"**Target**: proxy_congestion_score (synthetic composite, NOT measured congestion).",
        "",
        "## Per-Model Results (mean +/- std across folds)",
        "",
        "| Model | MAE | RMSE | R² | F1 |",
        "|---|---|---|---|---|",
    ]

    sorted_models = sorted(
        [(m, s) for m, s in model_summary.items() if s.get("status") != "no_successful_folds"],
        key=lambda x: x[1]["metrics"]["r2"]["mean"],
        reverse=True,
    )
    for m, s in sorted_models:
        met = s["metrics"]
        md_lines.append(
            f"| {m} | "
            f"{met['mae']['mean']:.6f} +/- {met['mae']['std']:.6f} | "
            f"{met['rmse']['mean']:.6f} +/- {met['rmse']['std']:.6f} | "
            f"{met['r2']['mean']:.6f} +/- {met['r2']['std']:.6f} | "
            f"{met['f1']['mean']:.6f} +/- {met['f1']['std']:.6f} |"
        )

    if stat_tests:
        md_lines.extend(["", "## Statistical Tests (Proposed vs. Baselines)", ""])
        md_lines.append("| Baseline | R² Diff | Wilcoxon p | Cohen's d | Significant? |")
        md_lines.append("|---|---|---|---|---|")
        for t in stat_tests:
            sig = "Yes" if t["significant"] else "No"
            md_lines.append(
                f"| {t['vs_model']} | {t['r2_diff']:+.6f} | "
                f"{t['wilcoxon_p']:.6f} | {t['cohens_d']:.4f} | {sig} |"
            )

    md_lines.extend([
        "",
        "## Notes",
        "- Rolling-origin evaluation ensures temporal validity.",
        "- Statistical tests use Wilcoxon signed-rank (non-parametric).",
        "- Negative R² means the model is worse than predicting the mean.",
        "- All results use chronological splits (no future data leakage).",
    ])

    md_text = "\n".join(md_lines) + "\n"
    (output_dir / "rolling_cv_report.md").write_text(md_text, encoding="utf-8")
    print(f"\n{md_text}")
    print(f"\nResults saved to: {output_dir}")
    return {"model_summary": model_summary, "statistical_tests": stat_tests}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rolling-origin cross-validation")
    parser.add_argument("--data", required=True, help="Path to windows FP16 NPZ")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Models to evaluate")
    parser.add_argument("--epochs", type=int, default=60, help="Epochs per fold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="outputs/rolling_cv", help="Output directory")
    args = parser.parse_args(argv)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        print("Run the data pipeline first:")
        print(f"  python scripts/run_data_pipeline.py --config configs/data/nasa_http_3m.yaml")
        return 1

    result = run_rolling_cv(
        str(data_path), args.models, args.n_folds,
        args.epochs, args.seed, args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
