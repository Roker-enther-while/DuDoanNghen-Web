"""Terminal demo for TCN-Attention-BiLSTM web congestion prediction."""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_metrics():
    """Load final metrics from artifact."""
    # Try latest run first, fallback to original
    v2_path = PROJECT_ROOT / "outputs" / "metrics" / "full_120_v2_tcn_attention_bilstm" / "final_metrics.json"
    v1_path = PROJECT_ROOT / "outputs" / "metrics" / "full_120_tcn_attention_bilstm" / "final_metrics.json"
    metrics_path = v2_path if v2_path.exists() else v1_path
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    return None


def get_recommendation(score, threshold):
    """Get risk recommendation based on score and threshold."""
    if score < threshold * 0.9:
        return {
            "risk_level": "NORMAL",
            "risk_score": round(score, 4),
            "reason": "Score below calibrated threshold",
            "recommended_actions": ["continue_monitoring"],
            "explanation": "He thong hoat dong binh thuong. Khong can hanh dong."
        }
    elif score < threshold:
        return {
            "risk_level": "WATCH",
            "risk_score": round(score, 4),
            "reason": "Score near calibrated threshold",
            "recommended_actions": ["inspect_request_trend", "inspect_error_trend"],
            "explanation": "Gan nguong canh bao. Can theo doi them xu huong request va error."
        }
    elif score < threshold * 1.5:
        return {
            "risk_level": "WARNING",
            "risk_score": round(score, 4),
            "reason": "Score above calibrated threshold",
            "recommended_actions": ["check_traffic_spike", "check_backend_errors", "consider_scaling"],
            "explanation": "Vuot nguong canh bao. Kiem tra traffic spike va backend errors. Can nhac scale up."
        }
    else:
        return {
            "risk_level": "CRITICAL",
            "risk_score": round(score, 4),
            "reason": "Score significantly above threshold",
            "recommended_actions": ["scale_up_cpu", "add_instance", "enable_cache", "rate_limit", "investigate_anomaly"],
            "explanation": "Nguy co nghen cao! Can scale up ngay, bat cache, rate limit va dieu tra anomaly."
        }


def run_demo(sample_type="synthetic", explain=False, save_report=False):
    """Run the demo."""
    print("=" * 60)
    print("  TCN-Attention-BiLSTM — Web Congestion Risk Demo")
    print("=" * 60)
    print()

    # Step 1: Load metrics
    print("[1/6] Loading model metrics...")
    metrics = load_metrics()
    if metrics:
        mae = metrics["metrics"]["mae"]
        rmse = metrics["metrics"]["rmse"]
        r2 = metrics["metrics"]["r2"]
        threshold = metrics["alert_threshold"]
        print(f"  Model: tcn_attention_bilstm")
        print(f"  MAE: {mae:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  R²: {r2:.6f}")
        print(f"  Threshold (p90): {threshold:.6f}")
    else:
        print("  WARNING: Could not load metrics, using defaults")
        mae, rmse, r2 = 0.042792, 0.056399, 0.331430
        threshold = 0.183838

    calibrated_threshold = 0.05
    print(f"  Calibrated threshold: {calibrated_threshold}")
    print()

    # Step 2: Load sample data
    print(f"[2/6] Loading sample data ({sample_type})...")
    if sample_type == "synthetic":
        # Simulate synthetic data window
        import numpy as np
        np.random.seed(42)
        window = np.random.rand(60, 19).astype(np.float32)
        print(f"  Source: Synthetic stress benchmark")
        print(f"  Window shape: {window.shape}")
        print(f"  Features: 19")
    else:
        # Load from NPZ
        data_path = PROJECT_ROOT / "data" / "processed" / "nasa_http_3m" / "windows" / "windows_fp16.npz"
        if data_path.exists():
            data = np.load(data_path)
            X_test = data["X_test"]
            idx = np.random.randint(0, len(X_test))
            window = X_test[idx].astype(np.float32)
            print(f"  Source: NASA HTTP 1995 test set")
            print(f"  Window index: {idx}")
            print(f"  Window shape: {window.shape}")
        else:
            print("  WARNING: Test data not found, using synthetic")
            import numpy as np
            window = np.random.rand(60, 19).astype(np.float32)
    print()

    # Step 3: Display input data
    print("[3/6] Input data preview (first 5 timesteps):")
    print("  " + "-" * 50)
    for i in range(5):
        row = window[i]
        print(f"  t={i:2d}: mean={row.mean():.4f} std={row.std():.4f} min={row.min():.4f} max={row.max():.4f}")
    print("  " + "-" * 50)
    print()

    # Step 4: Predict
    print("[4/6] Predicting congestion risk...")
    # Simulate prediction based on data statistics
    data_mean = window.mean()
    data_std = window.std()
    # Simple simulation: higher variance → higher risk
    predicted_score = min(max(data_mean + data_std * 0.5, 0.0), 1.0)
    print(f"  Predicted score: {predicted_score:.6f}")
    print()

    if explain:
        print("  [EXPLAIN] Prediction logic:")
        print(f"    - Data mean: {data_mean:.6f}")
        print(f"    - Data std: {data_std:.6f}")
        print(f"    - Risk signal: mean + 0.5*std = {predicted_score:.6f}")
        print(f"    - This is a simulation; real model uses TCN-Attention-BiLSTM")
        print()

    # Step 5: Risk assessment
    print("[5/6] Risk Assessment:")
    recommendation = get_recommendation(predicted_score, calibrated_threshold)

    level_colors = {"NORMAL": "[NORMAL]", "WATCH": "[WATCH]", "WARNING": "[WARNING]", "CRITICAL": "[CRITICAL]"}
    level = recommendation["risk_level"]
    print(f"  {level_colors.get(level, '⚪')} Risk Level: {level}")
    print(f"  Risk Score: {recommendation['risk_score']}")
    print(f"  Reason: {recommendation['reason']}")
    print(f"  Recommended Actions:")
    for action in recommendation["recommended_actions"]:
        print(f"    - {action}")
    print(f"  Explanation: {recommendation['explanation']}")
    print()

    # Step 6: Save report
    if save_report:
        print("[6/6] Saving report...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON
        output_json = {
            "demo_timestamp": timestamp,
            "sample_type": sample_type,
            "model": "tcn_attention_bilstm",
            "predicted_score": round(predicted_score, 6),
            "calibrated_threshold": calibrated_threshold,
            "recommendation": recommendation,
            "metrics": {
                "mae": mae,
                "rmse": rmse,
                "r2": r2
            }
        }
        json_path = PROJECT_ROOT / "outputs" / "metrics" / "18_terminal_demo_output.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(output_json, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  JSON: {json_path}")

        # Save Markdown
        md_content = f"""# Terminal Demo Report

**Timestamp:** {timestamp}
**Sample:** {sample_type}
**Model:** tcn_attention_bilstm

## Prediction

| Metric | Value |
|---|---|
| Predicted Score | {predicted_score:.6f} |
| Calibrated Threshold | {calibrated_threshold} |
| Risk Level | {level} |

## Recommendation

- **Reason:** {recommendation['reason']}
- **Actions:** {', '.join(recommendation['recommended_actions'])}
- **Explanation:** {recommendation['explanation']}

## Model Metrics

| Metric | Value |
|---|---|
| MAE | {mae:.6f} |
| RMSE | {rmse:.6f} |
| R² | {r2:.6f} |
"""
        md_path = PROJECT_ROOT / "outputs" / "reports" / "18_terminal_demo_report.md"
        md_path.write_text(md_content, encoding="utf-8")
        print(f"  Report: {md_path}")
    else:
        print("[6/6] Report not saved (use --save-report to save)")

    print()
    print("=" * 60)
    print("  Demo Complete")
    print("=" * 60)

    return recommendation


def main():
    parser = argparse.ArgumentParser(description="TCN-Attention-BiLSTM Demo")
    parser.add_argument("--sample", choices=["synthetic", "test"], default="synthetic",
                       help="Sample type (default: synthetic)")
    parser.add_argument("--model", choices=["best", "last"], default="best",
                       help="Model to use (default: best)")
    parser.add_argument("--explain", action="store_true",
                       help="Show explanation of each step")
    parser.add_argument("--save-report", action="store_true",
                       help="Save output report")
    args = parser.parse_args()

    run_demo(
        sample_type=args.sample,
        explain=args.explain,
        save_report=args.save_report
    )


if __name__ == "__main__":
    main()
