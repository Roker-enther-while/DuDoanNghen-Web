"""Alert-threshold calibration utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.training.metrics import alert_metrics


def make_threshold_grid(y_reference, y_pred=None, lower: float = 0.05, upper: float = 0.50, steps: int = 91) -> np.ndarray:
    """Create a threshold grid from fixed range plus target/prediction quantiles."""
    values = [*np.linspace(lower, upper, steps)]
    ref = np.asarray(y_reference, dtype=np.float64).reshape(-1)
    if ref.size:
        values.extend(np.quantile(ref, np.linspace(0.50, 0.99, 50)).tolist())
    if y_pred is not None:
        pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        if pred.size:
            values.extend(np.quantile(pred, np.linspace(0.50, 0.99, 50)).tolist())
    clean = sorted({round(float(v), 10) for v in values if np.isfinite(v)})
    return np.asarray(clean, dtype=np.float64)


def sweep_thresholds(y_true, y_pred, thresholds) -> list[dict]:
    """Evaluate alert metrics for each candidate threshold."""
    rows = []
    for threshold in thresholds:
        metrics = alert_metrics(y_true, y_pred, float(threshold))
        rows.append({"threshold": float(threshold), **metrics})
    return rows


def choose_best_threshold(rows: list[dict]) -> dict:
    """Pick threshold by best F1, then recall, precision, and higher threshold for stability."""
    if not rows:
        raise ValueError("No threshold rows to choose from")
    return max(rows, key=lambda r: (r["f1"], r["recall"], r["precision"], r["threshold"]))


def choose_recall_threshold(rows: list[dict], min_recall: float = 0.5) -> dict | None:
    """Pick the best F1 threshold among rows satisfying a minimum recall."""
    candidates = [row for row in rows if row["recall"] >= min_recall]
    return choose_best_threshold(candidates) if candidates else None


def choose_balanced_threshold(rows: list[dict]) -> dict:
    """Pick threshold minimizing absolute precision/recall gap, then best F1."""
    if not rows:
        raise ValueError("No threshold rows to choose from")
    return min(rows, key=lambda r: (abs(r["precision"] - r["recall"]), -r["f1"], -r["threshold"]))


def ensure_finite_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
