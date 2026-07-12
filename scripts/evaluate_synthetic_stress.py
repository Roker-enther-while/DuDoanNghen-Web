"""Evaluate a trained PyTorch model on synthetic stress data separately from real tests."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.torch_models import build_torch_model


def _finite_float(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else 0.0


def binary_metrics_from_scores(y_true_binary, y_score, threshold: float) -> dict:
    """Compute alert metrics against explicit binary synthetic labels."""
    true = np.asarray(y_true_binary).astype(bool).reshape(-1)
    score = np.asarray(y_score, dtype=np.float32).reshape(-1)
    pred = score >= float(threshold)
    tp = int(np.logical_and(true, pred).sum())
    fp = int(np.logical_and(~true, pred).sum())
    tn = int(np.logical_and(~true, ~pred).sum())
    fn = int(np.logical_and(true, ~pred).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(true) if len(true) else 0.0
    return {
        "threshold": float(threshold),
        "precision": _finite_float(precision),
        "recall": _finite_float(recall),
        "f1": _finite_float(f1),
        "accuracy": _finite_float(accuracy),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "positive_true_count": int(true.sum()),
        "positive_pred_count": int(pred.sum()),
    }


def sweep_thresholds_for_labels(y_true_binary, y_score, thresholds=None) -> tuple[list[dict], dict]:
    """Sweep thresholds and return all rows plus best-by-F1 row."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    rows = [binary_metrics_from_scores(y_true_binary, y_score, float(threshold)) for threshold in thresholds]
    best = max(rows, key=lambda row: (row["f1"], row["recall"], row["precision"]))
    return rows, best


def grouped_metrics(labels: pd.DataFrame, y_score, threshold: float, group_col: str) -> pd.DataFrame:
    """Compute synthetic alert metrics per scenario or phase."""
    rows = []
    scores = np.asarray(y_score, dtype=np.float32)
    for group_value, group in labels.groupby(group_col, sort=True):
        idx = group.index.to_numpy()
        metrics = binary_metrics_from_scores(group["true_alert_label"].astype(bool).to_numpy(), scores[idx], threshold)
        metrics[group_col] = group_value
        metrics["sample_count"] = int(len(group))
        rows.append(metrics)
    return pd.DataFrame(rows)


def load_torch_predictions(model_path: str | Path, X, batch_size: int = 512) -> tuple[np.ndarray, dict]:
    """Load a PyTorch checkpoint produced by this repo and run inference."""
    import torch
    from torch.utils.data import DataLoader

    checkpoint = torch.load(model_path, map_location="cpu")
    model_name = checkpoint["model_name"]
    config = checkpoint.get("config", {})
    input_shape = tuple(checkpoint.get("input_shape", X.shape[1:]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_torch_model(model_name, input_shape, config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    loader = DataLoader(torch.as_tensor(X, dtype=torch.float32), batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"))
    preds = []
    started = time.perf_counter()
    with torch.no_grad():
        for xb in loader:
            preds.append(model(xb.to(device, non_blocking=True)).detach().cpu().numpy())
    inference_time = time.perf_counter() - started
    return np.concatenate(preds).astype(np.float32), {
        "model_name": model_name,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "inference_time_seconds": _finite_float(inference_time),
        "checkpoint_config": config,
    }


def resolve_checkpoint_threshold(model_path: str | Path, default: float = 0.5) -> float:
    """Use dashboard/final metric threshold when available; synthetic sweep is also reported separately."""
    model_path = Path(model_path).as_posix()
    if "full_120_tcn_attention_bilstm" in model_path:
        metrics_path = ROOT / "outputs" / "metrics" / "full_120_tcn_attention_bilstm" / "final_metrics.json"
    elif "multisource_full_120_tcn_attention_bilstm" in model_path:
        metrics_path = ROOT / "outputs" / "metrics" / "multisource_full_120_tcn_attention_bilstm" / "final_metrics.json"
    else:
        metrics_path = None
    if metrics_path and metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        return float(payload.get("alert_threshold", payload.get("alert_metrics", {}).get("alert_threshold", default)))
    return float(default)


def evaluate_synthetic(data_path: str | Path, labels_path: str | Path, model_path: str | Path, output_dir: str | Path) -> dict:
    """Evaluate synthetic stress labels and write metrics/report/predictions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(data_path, allow_pickle=True) as data:
        X_test = data["X_test"].astype(np.float32)
        y_test_proxy = data["y_test"].astype(np.float32)
    labels = pd.read_csv(labels_path).reset_index(drop=True)
    if len(labels) != len(X_test):
        raise ValueError(f"labels rows {len(labels)} do not match X_test samples {len(X_test)}")
    y_pred, infer_info = load_torch_predictions(model_path, X_test)
    checkpoint_threshold = resolve_checkpoint_threshold(model_path)
    checkpoint_metrics = binary_metrics_from_scores(labels["true_alert_label"].astype(bool).to_numpy(), y_pred, checkpoint_threshold)
    sweep_rows, best = sweep_thresholds_for_labels(labels["true_alert_label"].astype(bool).to_numpy(), y_pred)

    predictions = labels.copy()
    predictions["y_proxy"] = y_test_proxy
    predictions["y_pred"] = y_pred
    predictions["pred_alert_checkpoint_threshold"] = y_pred >= checkpoint_threshold
    predictions["pred_alert_best_synthetic_f1_threshold"] = y_pred >= best["threshold"]
    predictions_path = output_dir / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    scenario_df = grouped_metrics(labels, y_pred, best["threshold"], "scenario_name")
    phase_df = grouped_metrics(labels, y_pred, best["threshold"], "phase")
    scenario_path = output_dir / "scenario_metrics.csv"
    phase_path = output_dir / "phase_metrics.csv"
    scenario_df.to_csv(scenario_path, index=False)
    phase_df.to_csv(phase_path, index=False)

    metrics = {
        "result_type": "synthetic_stress_test",
        "synthetic_not_real_world": True,
        "not_mixed_with_real_public_result": True,
        "data_path": str(data_path),
        "labels_path": str(labels_path),
        "model_path": str(model_path),
        "sample_count": int(len(labels)),
        "positive_count": int(labels["true_alert_label"].astype(bool).sum()),
        "negative_count": int((~labels["true_alert_label"].astype(bool)).sum()),
        "positive_ratio": _finite_float(labels["true_alert_label"].astype(bool).mean()),
        "scenario_count": int(labels["scenario_name"].nunique()),
        "phase_count": int(labels["phase"].nunique()),
        "checkpoint_threshold": checkpoint_threshold,
        "checkpoint_threshold_metrics": checkpoint_metrics,
        "best_synthetic_f1_threshold": best,
        "threshold_sweep": sweep_rows,
        "inference": infer_info,
        "prediction_path": str(predictions_path),
        "scenario_metrics_path": str(scenario_path),
        "phase_metrics_path": str(phase_path),
        "warnings": [
            "synthetic_not_real_world",
            "generated_from_public_baseline",
            "must_not_mix_with_real_public_test",
            "threshold_chosen_on_synthetic_is_not_real_world_threshold",
        ],
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    best_scenario = scenario_df.sort_values(["f1", "recall"], ascending=False).iloc[0].to_dict()
    worst_scenario = scenario_df.sort_values(["f1", "recall"], ascending=True).iloc[0].to_dict()
    report_path = output_dir / "report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Synthetic Stress Evaluation",
                "",
                "- result_type: synthetic_stress_test",
                "- synthetic_not_real_world: true",
                "- not_mixed_with_real_public_result: true",
                f"- Model: {model_path}",
                f"- Samples: {metrics['sample_count']}",
                f"- Positive ratio: {metrics['positive_ratio']:.6f}",
                f"- Checkpoint threshold: {checkpoint_threshold:.6f}",
                f"- Checkpoint precision/recall/F1: {checkpoint_metrics['precision']:.6f} / {checkpoint_metrics['recall']:.6f} / {checkpoint_metrics['f1']:.6f}",
                f"- Best synthetic F1 threshold: {best['threshold']:.6f}",
                f"- Best synthetic precision/recall/F1: {best['precision']:.6f} / {best['recall']:.6f} / {best['f1']:.6f}",
                f"- Best scenario by F1: {best_scenario.get('scenario_name')} ({best_scenario.get('f1'):.6f})",
                f"- Worst scenario by F1: {worst_scenario.get('scenario_name')} ({worst_scenario.get('f1'):.6f})",
                "",
                "Synthetic stress results are controlled benchmark results only and must not be reported as real-world performance.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    metrics = evaluate_synthetic(args.data, args.labels, args.model_path, args.output_dir)
    print(
        json.dumps(
            {
                "status": "success",
                "metrics_path": str(Path(args.output_dir) / "metrics.json"),
                "precision": metrics["best_synthetic_f1_threshold"]["precision"],
                "recall": metrics["best_synthetic_f1_threshold"]["recall"],
                "f1": metrics["best_synthetic_f1_threshold"]["f1"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
