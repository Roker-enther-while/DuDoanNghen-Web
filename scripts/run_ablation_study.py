"""Run ablation study for TCN-Attention-BiLSTM architecture.

Trains each ablation variant with multiple seeds and produces a comparison table.

Variants:
  1. TCN only (no Attention, no BiLSTM)
  2. BiLSTM only (no TCN, no Attention)
  3. Attention + BiLSTM (no TCN)
  4. TCN + BiLSTM (no Attention)
  5. TCN + Attention (no BiLSTM)
  6. Full: TCN + Attention + BiLSTM (proposed)

Usage:
    # Requires data/processed/nasa_http_3m/windows/windows_fp16.npz
    python scripts/run_ablation_study.py --data data/processed/nasa_http_3m/windows/windows_fp16.npz --seeds 42 123 456
    python scripts/run_ablation_study.py --data data/processed/nasa_http_3m/windows/windows_fp16.npz --seeds 42 123 456 --epochs 60
"""

from __future__ import annotations

import argparse
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
from src.training.threshold_calibration import make_threshold_grid, sweep_thresholds, choose_best_threshold
from src.training.metrics import alert_metrics


ABLATION_VARIANTS = {
    "naive_last_value": {
        "model_name": "naive_last_value",
        "config_path": "configs/training/ablation/base.yaml",
        "description": "Naive baseline: predict last observed value.",
    },
    "moving_average": {
        "model_name": "moving_average",
        "config_path": "configs/training/ablation/base.yaml",
        "description": "Baseline: predict moving average of last values.",
    },
    "tcn_only": {
        "model_name": "tcn",
        "config_path": "configs/training/ablation/tcn_only.yaml",
        "description": "TCN only (causal convolutional, no attention, no recurrent)",
    },
    "bilstm_only": {
        "model_name": "bilstm",
        "config_path": "configs/training/ablation/bilstm.yaml",
        "description": "BiLSTM only (bidirectional recurrent, no convolutional, no attention)",
    },
    "attention_bilstm": {
        "model_name": "attention_bilstm",
        "config_path": "configs/training/ablation/attention_bilstm.yaml",
        "description": "Self-Attention + BiLSTM (no TCN)",
    },
    "tcn_bilstm": {
        "model_name": "tcn_lstm",
        "config_path": "configs/training/ablation/tcn_bilstm.yaml",
        "description": "TCN + BiLSTM (no self-attention)",
    },
    "tcn_attention": {
        "model_name": "tcn_attention",
        "config_path": "configs/training/ablation/tcn_attention.yaml",
        "description": "TCN + Self-Attention (no BiLSTM)",
    },
    "full": {
        "model_name": "tcn_attention_bilstm",
        "config_path": "configs/training/ablation/full.yaml",
        "description": "Full: TCN + Self-Attention + BiLSTM (proposed)",
    },
}


def run_single_ablation(variant_name: str, variant_info: dict, data_path: str, seed: int, epochs: int | None = None) -> dict:
    """Train one ablation variant with a specific seed."""
    config = load_training_config(variant_info["config_path"])
    config["seed"] = seed
    if epochs is not None:
        config["epochs"] = epochs
    # Ensure unique artifact name per seed
    config["artifact_name"] = f"ablation_{variant_name}_seed{seed}"
    config["output_tag"] = f"ablation_{variant_name}_seed{seed}"

    print(f"\n{'='*60}")
    print(f"Training: {variant_name} (seed={seed})")
    print(f"Model: {variant_info['model_name']}")
    print(f"Description: {variant_info['description']}")
    print(f"{'='*60}")

    try:
        result = train_and_evaluate(variant_info["model_name"], data_path, config)

        # Per-variant threshold calibration: load val predictions, find best threshold, apply to test
        val_pred_path = result.get("prediction_path", "").replace("test_predictions", "val_predictions")
        # The trainer saves test predictions but not val predictions separately.
        # Re-load data and compute val predictions for threshold calibration.
        raw_data = load_window_data(data_path)
        y_val = raw_data["y_val"].astype(np.float32)
        y_test = raw_data["y_test"].astype(np.float32)

        # Load test predictions from the saved CSV
        pred_path = result.get("prediction_path")
        if pred_path and Path(pred_path).exists():
            pred_df = pd.read_csv(pred_path)
            test_pred = pred_df["y_pred"].values.astype(np.float32)
        else:
            test_pred = np.zeros_like(y_test, dtype=np.float32)

        # For val predictions, we need to re-load from the model's history or re-predict
        # The trainer stores val predictions implicitly. Let's use the shared threshold first,
        # then calibrate per-variant using the test predictions' distribution as a proxy.
        # Better approach: re-compute val predictions from the model checkpoint.
        model_path = result.get("model_path")
        if model_path and Path(model_path).exists():
            try:
                import torch
                from torch.utils.data import DataLoader
                from src.training.torch_models import build_torch_model

                checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                model_name_ckpt = checkpoint.get("model_name", variant_info["model_name"])
                config_ckpt = checkpoint.get("config", config)
                input_shape = tuple(checkpoint.get("input_shape", (60, 19)))
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model = build_torch_model(model_name_ckpt, input_shape, config_ckpt).to(device)
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()

                val_x = torch.as_tensor(raw_data["X_val"].astype(np.float32))
                val_loader = DataLoader(val_x, batch_size=256, shuffle=False)
                val_preds = []
                with torch.no_grad():
                    for xb in val_loader:
                        val_preds.append(model(xb.to(device)).detach().cpu().numpy())
                val_pred = np.concatenate(val_preds).astype(np.float32)

                # Calibrate threshold on val predictions
                # Constrain search to meaningful range: ensure at least 5% positive rate
                min_threshold = float(np.quantile(y_val, 0.50))  # at least p50 of y_val
                thresholds = make_threshold_grid(y_val, val_pred, lower=min_threshold, upper=0.50)
                sweep = sweep_thresholds(y_val, val_pred, thresholds)
                # Filter to thresholds with at least 5% positive predictions
                valid_sweep = [r for r in sweep if r["alert_positive_count_pred"] >= 0.05 * len(y_val)]
                if valid_sweep:
                    best = choose_best_threshold(valid_sweep)
                else:
                    best = choose_best_threshold(sweep)
                calibrated_threshold = float(best["threshold"])

                # Apply calibrated threshold to test predictions
                calibrated_test_metrics = alert_metrics(y_test, test_pred, calibrated_threshold)

                # Also compute with shared threshold for comparison
                shared_threshold = result["alert_threshold"]
                shared_test_metrics = alert_metrics(y_test, test_pred, shared_threshold)

                print(f"  Calibrated threshold: {calibrated_threshold:.6f} (F1={calibrated_test_metrics['f1']:.4f})")
                print(f"  Shared threshold:     {shared_threshold:.6f} (F1={shared_test_metrics['f1']:.4f})")
            except Exception as cal_exc:
                print(f"  Threshold calibration failed: {cal_exc}")
                calibrated_test_metrics = result["alert_metrics"]
                calibrated_threshold = result["alert_threshold"]
        else:
            calibrated_test_metrics = result["alert_metrics"]
            calibrated_threshold = result["alert_threshold"]

        return {
            "variant": variant_name,
            "model_name": variant_info["model_name"],
            "seed": seed,
            "status": result["status"],
            "mae": result["metrics"]["mae"],
            "rmse": result["metrics"]["rmse"],
            "r2": result["metrics"]["r2"],
            "f1": result["alert_metrics"]["f1"],
            "f1_calibrated": calibrated_test_metrics["f1"],
            "precision_calibrated": calibrated_test_metrics["precision"],
            "recall_calibrated": calibrated_test_metrics["recall"],
            "calibrated_threshold": calibrated_threshold,
            "precision": result["alert_metrics"]["precision"],
            "recall": result["alert_metrics"]["recall"],
            "train_time_seconds": result["train_time_seconds"],
            "inference_time_seconds": result["inference_time_seconds"],
            "metrics_path": result.get("metrics_path"),
            "prediction_path": result.get("prediction_path"),
        }
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return {
            "variant": variant_name,
            "model_name": variant_info["model_name"],
            "seed": seed,
            "status": "failed",
            "error": str(exc),
        }


def compute_ablation_summary(results: list[dict]) -> dict:
    """Compute mean +/- std across seeds for each variant."""
    variants = {}
    for r in results:
        v = r["variant"]
        if v not in variants:
            variants[v] = []
        if r["status"] == "success":
            variants[v].append(r)

    summary = {}
    for v, runs in variants.items():
        if not runs:
            summary[v] = {"status": "no_successful_runs"}
            continue
        metrics = {}
        for metric in ["mae", "rmse", "r2", "f1", "precision", "recall", "f1_calibrated", "precision_calibrated", "recall_calibrated", "calibrated_threshold"]:
            values = [r[metric] for r in runs if metric in r]
            if not values:
                continue
            metrics[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": [float(x) for x in values],
            }
        summary[v] = {
            "n_seeds": len(runs),
            "status": "success",
            "metrics": metrics,
            "mean_train_time": float(np.mean([r["train_time_seconds"] for r in runs])),
        }
    return summary


def statistical_tests(summary: dict) -> list[dict]:
    """Paired tests between full model and each variant."""
    full_key = "full"
    if full_key not in summary or summary[full_key].get("status") != "success":
        return []

    full_r2_values = summary[full_key]["metrics"]["r2"]["values"]
    full_mae_values = summary[full_key]["metrics"]["mae"]["values"]
    tests = []

    for variant, info in summary.items():
        if variant == full_key or info.get("status") != "success":
            continue
        var_r2_values = info["metrics"]["r2"]["values"]
        var_mae_values = info["metrics"]["mae"]["values"]
        n = min(len(full_r2_values), len(var_r2_values))

        if n >= 3:
            # Wilcoxon signed-rank test (non-parametric, suitable for small n)
            try:
                r2_stat, r2_p = sp_stats.wilcoxon(full_r2_values[:n], var_r2_values[:n])
            except ValueError:
                r2_stat, r2_p = 0.0, 1.0
            try:
                mae_stat, mae_p = sp_stats.wilcoxon(full_mae_values[:n], var_mae_values[:n])
            except ValueError:
                mae_stat, mae_p = 0.0, 1.0

            # Effect size (Cohen's d)
            r2_diff = np.array(full_r2_values[:n]) - np.array(var_r2_values[:n])
            r2_cohens_d = float(np.mean(r2_diff) / max(np.std(r2_diff, ddof=1), 1e-10))

            tests.append({
                "variant": variant,
                "vs_full_r2_wilcoxon_p": round(r2_p, 6),
                "vs_full_r2_cohens_d": round(r2_cohens_d, 4),
                "vs_full_mae_wilcoxon_p": round(mae_p, 6),
                "full_r2_mean": summary[full_key]["metrics"]["r2"]["mean"],
                "variant_r2_mean": info["metrics"]["r2"]["mean"],
                "r2_difference": round(summary[full_key]["metrics"]["r2"]["mean"] - info["metrics"]["r2"]["mean"], 6),
            })
    return tests


def generate_comparison_table(summary: dict, tests: list[dict]) -> str:
    """Generate markdown comparison table."""
    lines = [
        "# Ablation Study Results",
        "",
        "Target: proxy_congestion_score (synthetic composite, NOT measured congestion).",
        "All results on chronological test split of NASA HTTP 1995 data.",
        "",
        "## Metrics (mean +/- std across seeds)",
        "",
        "| Variant | MAE | RMSE | R² | F1 (shared) | F1 (calibrated) | Threshold | Precision | Recall | Train Time (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    # Sort by R² descending
    sorted_variants = sorted(
        [(v, s) for v, s in summary.items() if s.get("status") == "success"],
        key=lambda x: x[1]["metrics"]["r2"]["mean"],
        reverse=True,
    )

    for v, s in sorted_variants:
        m = s["metrics"]
        best_marker = " **" if v == "full" else ""
        f1_cal = m.get("f1_calibrated", {})
        cal_thresh = m.get("calibrated_threshold", {})
        prec_cal = m.get("precision_calibrated", {})
        rec_cal = m.get("recall_calibrated", {})
        lines.append(
            f"| {v}{best_marker} | "
            f"{m['mae']['mean']:.6f} +/- {m['mae']['std']:.6f} | "
            f"{m['rmse']['mean']:.6f} +/- {m['rmse']['std']:.6f} | "
            f"{m['r2']['mean']:.6f} +/- {m['r2']['std']:.6f} | "
            f"{m['f1']['mean']:.6f} +/- {m['f1']['std']:.6f} | "
            f"{f1_cal.get('mean', 0):.6f} +/- {f1_cal.get('std', 0):.6f} | "
            f"{cal_thresh.get('mean', 0):.4f} | "
            f"{prec_cal.get('mean', 0):.6f} | "
            f"{rec_cal.get('mean', 0):.6f} | "
            f"{s['mean_train_time']:.1f} |"
        )

    if tests:
        lines.extend(["", "## Statistical Tests (Full vs. Each Variant)", ""])
        lines.append("| Variant | R² Difference | Wilcoxon p-value | Cohen's d | Significant (p<0.05)? |")
        lines.append("|---|---|---|---|---|")
        for t in tests:
            sig = "Yes" if t["vs_full_r2_wilcoxon_p"] < 0.05 else "No"
            lines.append(
                f"| {t['variant']} | {t['r2_difference']:+.6f} | "
                f"{t['vs_full_r2_wilcoxon_p']:.6f} | "
                f"{t['vs_full_r2_cohens_d']:.4f} | {sig} |"
            )

    lines.extend([
        "",
        "## Notes",
        "- **Bold** = proposed full model.",
        "- R² values can be negative (worse than predicting the mean).",
        "- Statistical tests use Wilcoxon signed-rank (non-parametric, suitable for small n).",
        "- Cohen's d > 0.8 = large effect, > 0.5 = medium, > 0.2 = small.",
        "- If the full model does not significantly outperform simpler variants,",
        "  the simpler variant is the recommended architecture (parsimony principle).",
    ])

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ablation study for TCN-Attention-BiLSTM")
    parser.add_argument("--data", required=True, help="Path to windows FP16 NPZ")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456], help="Random seeds")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (default: from config)")
    parser.add_argument("--output-dir", default="outputs/ablation_study", help="Output directory")
    parser.add_argument("--variants", nargs="+", default=None, help="Specific variants to run (default: all)")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify data exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        print("Run the data pipeline first:")
        print(f"  python scripts/run_data_pipeline.py --config configs/data/nasa_http_3m.yaml")
        return 1

    # Load data summary
    data = load_window_data(data_path)
    validate_window_data(data)
    print("Data summary:")
    print(json.dumps(get_data_summary(data), indent=2))

    # Select variants
    active_variants = args.variants or list(ABLATION_VARIANTS.keys())
    print(f"\nRunning ablation study with {len(active_variants)} variants x {len(args.seeds)} seeds = {len(active_variants) * len(args.seeds)} runs")

    # Run all variants
    all_results = []
    for variant_name in active_variants:
        if variant_name not in ABLATION_VARIANTS:
            print(f"Unknown variant: {variant_name}")
            continue
        for seed in args.seeds:
            result = run_single_ablation(
                variant_name, ABLATION_VARIANTS[variant_name],
                str(data_path), seed, args.epochs
            )
            all_results.append(result)

    # Save raw results
    (output_dir / "raw_results.json").write_text(
        json.dumps(all_results, indent=2, allow_nan=False), encoding="utf-8"
    )

    # Compute summary
    summary = compute_ablation_summary(all_results)
    (output_dir / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )

    # Statistical tests
    tests = statistical_tests(summary)
    (output_dir / "statistical_tests.json").write_text(
        json.dumps(tests, indent=2, allow_nan=False), encoding="utf-8"
    )

    # Generate comparison table
    table_md = generate_comparison_table(summary, tests)
    (output_dir / "ablation_comparison.md").write_text(table_md, encoding="utf-8")
    print(f"\n{table_md}")
    print(f"\nResults saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
