import json
import math

import numpy as np

from src.training.threshold_calibration import (
    choose_best_threshold,
    choose_recall_threshold,
    ensure_finite_json,
    make_threshold_grid,
    sweep_thresholds,
)


def test_threshold_sweep_selects_expected_synthetic_case():
    y_true = np.array([0, 0, 1, 1], dtype=float)
    y_pred = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
    rows = sweep_thresholds(y_true, y_pred, [0.5, 0.85])
    best = choose_best_threshold(rows)
    assert best["threshold"] == 0.5
    assert best["f1"] == 1.0


def test_recall_constraint_threshold():
    y_true = np.array([0, 1, 1], dtype=float)
    y_pred = np.array([0.9, 0.8, 0.1], dtype=float)
    rows = sweep_thresholds(y_true, y_pred, [0.5, 0.05])
    best = choose_recall_threshold(rows, min_recall=0.5)
    assert best is not None
    assert best["recall"] >= 0.5


def test_grid_and_json_are_finite(tmp_path):
    grid = make_threshold_grid([0.1, 0.2, 0.3], [0.2, 0.4], steps=3)
    assert np.isfinite(grid).all()
    path = tmp_path / "calibration.json"
    ensure_finite_json(path, {"thresholds": grid[:3].tolist(), "metric": 1.0})
    payload = json.loads(path.read_text(encoding="utf-8"))

    def walk(value):
        if isinstance(value, float):
            assert math.isfinite(value)
        elif isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
