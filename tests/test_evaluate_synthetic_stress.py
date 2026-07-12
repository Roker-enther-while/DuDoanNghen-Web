import numpy as np
import pandas as pd

from scripts.evaluate_synthetic_stress import binary_metrics_from_scores, grouped_metrics, sweep_thresholds_for_labels


def test_binary_metrics_from_scores():
    metrics = binary_metrics_from_scores([0, 1, 1, 0], [0.1, 0.8, 0.2, 0.7], threshold=0.5)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1


def test_sweep_thresholds_picks_best_f1():
    rows, best = sweep_thresholds_for_labels([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2], thresholds=np.array([0.3, 0.7]))
    assert len(rows) == 2
    assert best["f1"] == 1.0


def test_grouped_metrics_by_phase():
    labels = pd.DataFrame(
        {
            "phase": ["background", "incident", "incident", "recovery"],
            "true_alert_label": [0, 1, 1, 0],
        }
    )
    result = grouped_metrics(labels, [0.1, 0.8, 0.9, 0.3], 0.5, "phase")
    assert set(result["phase"]) == {"background", "incident", "recovery"}
    assert float(result[result["phase"] == "incident"]["f1"].iloc[0]) == 1.0
