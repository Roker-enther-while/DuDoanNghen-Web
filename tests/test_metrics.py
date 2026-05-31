import numpy as np

from src.training.metrics import alert_metrics, mae, r2_score, regression_metrics, rmse


def test_regression_metrics_small_example():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])
    assert mae(y_true, y_pred) == 1 / 3
    assert np.isclose(rmse(y_true, y_pred), np.sqrt(1 / 3))
    assert np.isclose(r2_score(y_true, y_pred), 0.5)
    metrics = regression_metrics(y_true, y_pred)
    assert set(metrics) == {"mae", "rmse", "r2"}


def test_alert_metrics_f1():
    y_true = np.array([0.8, 0.9, 0.1, 0.2])
    y_pred = np.array([0.8, 0.1, 0.9, 0.2])
    metrics = alert_metrics(y_true, y_pred, threshold=0.7)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fn"] == 1
    assert metrics["f1"] == 0.5
