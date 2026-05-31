import numpy as np

from src.training.metrics import alert_metrics, resolve_alert_threshold


def test_fixed_threshold_resolves():
    assert resolve_alert_threshold([0.1, 0.9], "fixed", 0.7) == 0.7


def test_quantile_threshold_resolves():
    y = np.array([0.0, 0.5, 1.0])
    assert np.isclose(resolve_alert_threshold(y, "quantile", 0.5), 0.5)


def test_no_positive_warning():
    metrics = alert_metrics([0.1, 0.2], [0.1, 0.9], threshold=0.7)
    assert metrics["warning"] == "no_positive_cases_in_y_true_for_threshold"
    assert metrics["alert_positive_count_true"] == 0
    assert metrics["alert_positive_count_pred"] == 1


def test_positive_f1_correct():
    metrics = alert_metrics([0.8, 0.9, 0.1, 0.2], [0.8, 0.1, 0.9, 0.2], threshold=0.7)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["f1"] == 0.5
