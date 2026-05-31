import numpy as np

from src.training.target_diagnostics import describe_target, target_distribution_report, threshold_counts


def test_quantile_summary_and_threshold_count():
    y = np.array([0.0, 0.5, 1.0])
    summary = describe_target(y)
    assert np.isclose(summary["quantiles"]["p50"], 0.5)
    counts = threshold_counts(y, {"t": 0.5})
    assert counts["t"]["count"] == 2


def test_target_distribution_report_suggests_quantile():
    data = {
        "y_train": np.linspace(0, 1, 10, dtype=np.float16),
        "y_val": np.linspace(0, 0.5, 10, dtype=np.float16),
        "y_test": np.linspace(0, 0.2, 10, dtype=np.float16),
    }
    report = target_distribution_report(data)
    assert report["suggested_alert_threshold"]["mode"] == "quantile"
    assert "threshold_0.70_has_no_positive_cases_in_test" in report["warnings"]
