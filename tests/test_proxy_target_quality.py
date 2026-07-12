import json
import math
import numpy as np

from src.training.target_diagnostics import proxy_target_quality, write_target_reports, target_distribution_report


def test_proxy_quality_no_nan_and_writes_json(tmp_path):
    y = np.linspace(0, 1, 100)
    quality = proxy_target_quality(y, y, y)
    assert "autocorrelation" in quality
    distribution = target_distribution_report({"y_train": y, "y_val": y, "y_test": y})
    paths = write_target_reports(distribution, quality, tmp_path)
    payload = json.loads(open(paths["proxy_target_quality_json"], encoding="utf-8").read())

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
