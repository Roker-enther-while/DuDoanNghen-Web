"""Calibrate alert threshold on validation predictions and evaluate full test."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.data_loader import load_window_data, validate_window_data
from src.training.metrics import alert_metrics
from src.training.threshold_calibration import (
    choose_balanced_threshold,
    choose_best_threshold,
    choose_recall_threshold,
    ensure_finite_json,
    make_threshold_grid,
    sweep_thresholds,
)
from src.training.torch_models import build_torch_model


def predict_with_torch_model(model_path: str | Path, X, batch_size: int = 512) -> np.ndarray:
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
    return np.concatenate(preds).astype(np.float32), time.perf_counter() - started, str(device)


def write_predictions(path: Path, y_true, y_pred, threshold: float, timestamps=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    if timestamps is not None and len(timestamps) == len(frame):
        frame.insert(0, "timestamp", timestamps)
    frame["abs_error"] = (frame["y_true"] - frame["y_pred"]).abs()
    frame["squared_error"] = (frame["y_true"] - frame["y_pred"]) ** 2
    frame["true_alert"] = frame["y_true"] >= threshold
    frame["pred_alert"] = frame["y_pred"] >= threshold
    frame.to_csv(path, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="outputs/full_120_tcn_attention_bilstm")
    parser.add_argument("--old-threshold", type=float, default=None)
    args = parser.parse_args(argv)

    data = load_window_data(args.data)
    validate_window_data(data)
    y_val = data["y_val"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)
    val_pred, val_infer, device = predict_with_torch_model(args.model_path, data["X_val"].astype(np.float32))
    test_pred, test_infer, _ = predict_with_torch_model(args.model_path, data["X_test"].astype(np.float32))

    metrics_path = Path("outputs/metrics/full_120_tcn_attention_bilstm/final_metrics.json")
    old_metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    old_threshold = args.old_threshold or float(old_metrics_payload.get("alert_threshold", 0.183837890625))
    old_test_metrics = alert_metrics(y_test, test_pred, old_threshold)

    thresholds = make_threshold_grid(y_val, val_pred, lower=0.05, upper=0.50)
    sweep = sweep_thresholds(y_val, val_pred, thresholds)
    best_f1 = choose_best_threshold(sweep)
    best_recall = choose_recall_threshold(sweep, min_recall=0.5)
    best_balanced = choose_balanced_threshold(sweep)
    chosen_threshold = float(best_f1["threshold"])
    calibrated_test_metrics = alert_metrics(y_test, test_pred, chosen_threshold)

    output_root = Path(args.output_dir)
    # User-requested canonical locations stay under outputs/metrics|reports|predictions/full_120_tcn_attention_bilstm.
    canonical_metrics_dir = Path("outputs/metrics/full_120_tcn_attention_bilstm")
    canonical_reports_dir = Path("outputs/reports/full_120_tcn_attention_bilstm")
    canonical_predictions_dir = Path("outputs/predictions/full_120_tcn_attention_bilstm")
    calibration_json = canonical_metrics_dir / "threshold_calibration.json"
    calibration_md = canonical_reports_dir / "threshold_calibration.md"
    calibrated_pred_path = canonical_predictions_dir / "test_predictions_calibrated.csv"
    write_predictions(calibrated_pred_path, y_test, test_pred, chosen_threshold, data.get("ts_test"))

    payload = {
        "model_path": str(args.model_path),
        "data_path": str(args.data),
        "device": device,
        "validation_inference_time_seconds": val_infer,
        "test_inference_time_seconds": test_infer,
        "old_threshold": old_threshold,
        "old_test_metrics": old_test_metrics,
        "best_threshold_by_val_f1": best_f1,
        "best_threshold_with_recall_gte_0_5": best_recall,
        "best_balanced_precision_recall_threshold": best_balanced,
        "calibrated_threshold": chosen_threshold,
        "calibrated_test_metrics": calibrated_test_metrics,
        "threshold_sweep": sweep,
        "prediction_path": str(calibrated_pred_path),
    }
    ensure_finite_json(calibration_json, payload)

    old_f1 = old_test_metrics["f1"]
    new_f1 = calibrated_test_metrics["f1"]
    old_recall = old_test_metrics["recall"]
    new_recall = calibrated_test_metrics["recall"]
    status_note = "under-alert" if calibrated_test_metrics["recall"] < 0.2 else ("over-alert" if calibrated_test_metrics["precision"] < 0.2 else "more balanced")
    lines = [
        "# Threshold Calibration",
        "",
        f"- Old threshold: {old_threshold:.6f}",
        f"- Old precision/recall/F1: {old_test_metrics['precision']:.6f} / {old_recall:.6f} / {old_f1:.6f}",
        f"- Best threshold by validation F1: {chosen_threshold:.6f}",
        f"- Calibrated test precision/recall/F1: {calibrated_test_metrics['precision']:.6f} / {new_recall:.6f} / {new_f1:.6f}",
        f"- Old confusion TP/FP/TN/FN: {old_test_metrics['tp']} / {old_test_metrics['fp']} / {old_test_metrics['tn']} / {old_test_metrics['fn']}",
        f"- New confusion TP/FP/TN/FN: {calibrated_test_metrics['tp']} / {calibrated_test_metrics['fp']} / {calibrated_test_metrics['tn']} / {calibrated_test_metrics['fn']}",
        f"- Model alert behavior after calibration: {status_note}",
        "",
        "## Alternative Thresholds",
        f"- Recall >= 0.5 candidate: {best_recall}",
        f"- Balanced precision/recall candidate: {best_balanced}",
    ]
    calibration_md.parent.mkdir(parents=True, exist_ok=True)
    calibration_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    dashboard_path = Path("outputs/web/full_120_tcn_attention_bilstm/model_dashboard_payload.json")
    if dashboard_path.exists():
        dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
        dashboard["threshold_calibration"] = {
            "path": str(calibration_json),
            "best_threshold_by_val_f1": best_f1,
            "best_threshold_with_recall_gte_0_5": best_recall,
            "best_balanced_precision_recall_threshold": best_balanced,
        }
        dashboard["old_metrics"] = old_test_metrics
        dashboard["calibrated_metrics"] = calibrated_test_metrics
        dashboard["recommendation_notes"] = [
            f"Use calibrated threshold {chosen_threshold:.6f} for alert evaluation if F1 is the priority.",
            "Old p90 validation threshold under-alerted on the test set.",
            "Threshold calibration changes alert classification only; regression predictions are unchanged.",
        ]
        ensure_finite_json(dashboard_path, dashboard)

    print(json.dumps({"calibration_json": str(calibration_json), "calibration_md": str(calibration_md), "prediction_path": str(calibrated_pred_path), "old_f1": old_f1, "new_f1": new_f1, "old_recall": old_recall, "new_recall": new_recall}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
