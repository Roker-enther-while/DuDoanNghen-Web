"""Regression and alert metrics without requiring sklearn."""

from __future__ import annotations

import numpy as np


def _arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true {true.shape}, y_pred {pred.shape}")
    return true, pred


def mae(y_true, y_pred) -> float:
    true, pred = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def rmse(y_true, y_pred) -> float:
    true, pred = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def r2_score(y_true, y_pred) -> float:
    true, pred = _arrays(y_true, y_pred)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    if ss_tot == 0:
        return 0.0 if ss_res > 0 else 1.0
    return float(1.0 - ss_res / ss_tot)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "r2": r2_score(y_true, y_pred)}


def resolve_alert_threshold(y_reference, mode: str = "fixed", value: float = 0.70) -> float:
    """Resolve an alert threshold from a fixed value or a reference quantile."""
    mode = (mode or "fixed").lower()
    if mode == "fixed":
        return float(value)
    if mode == "quantile":
        reference = np.asarray(y_reference, dtype=np.float64).reshape(-1)
        if reference.size == 0:
            raise ValueError("Cannot resolve quantile threshold from an empty reference array")
        return float(np.quantile(reference, float(value)))
    raise ValueError(f"Unsupported alert threshold mode: {mode}")


def alert_metrics(y_true, y_pred, threshold: float = 0.70) -> dict[str, float | int]:
    true, pred = _arrays(y_true, y_pred)
    true_alert = true >= threshold
    pred_alert = pred >= threshold
    tp = int(np.logical_and(true_alert, pred_alert).sum())
    fp = int(np.logical_and(~true_alert, pred_alert).sum())
    tn = int(np.logical_and(~true_alert, ~pred_alert).sum())
    fn = int(np.logical_and(true_alert, ~pred_alert).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(true) if len(true) else 0.0
    warning = None
    if int(true_alert.sum()) == 0:
        warning = "no_positive_cases_in_y_true_for_threshold"
    elif int(pred_alert.sum()) == 0:
        warning = "no_positive_predictions_for_threshold"
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "alert_threshold": float(threshold),
        "alert_positive_count_true": int(true_alert.sum()),
        "alert_positive_count_pred": int(pred_alert.sum()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "warning": warning,
    }
