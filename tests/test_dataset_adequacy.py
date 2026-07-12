import json

import numpy as np
import pandas as pd

from src.data.dataset_adequacy import analyze_windows_artifact, write_adequacy_report


def _npz(path, y_test):
    X = np.zeros((len(y_test), 4, 2), dtype=np.float16)
    y = np.asarray(y_test, dtype=np.float16)
    np.savez_compressed(
        path,
        X_train=X,
        y_train=y,
        ts_train=np.arange(len(y)),
        X_val=X,
        y_val=y,
        ts_val=np.arange(len(y)),
        X_test=X,
        y_test=y,
        ts_test=np.arange(len(y)),
        feature_columns=np.array(["request_count", "error_rate"], dtype=object),
        target_column=np.array("target", dtype=object),
    )


def test_detect_quiet_test(tmp_path):
    path = tmp_path / "quiet.npz"
    _npz(path, np.zeros(100, dtype=np.float16))
    result = analyze_windows_artifact(path, threshold=0.7, source_manifest={"valid_source_count": 1})
    assert result["test_quiet"] is True
    assert "test_split_has_too_few_positive_events" in result["warnings"]


def test_detect_enough_positive_cases_and_write_json(tmp_path):
    path = tmp_path / "events.npz"
    _npz(path, np.r_[np.zeros(50), np.ones(50)])
    result = analyze_windows_artifact(path, threshold=0.7, source_manifest={"valid_source_count": 1})
    assert result["splits"]["test"]["positive_count"] == 50
    output = write_adequacy_report(result, tmp_path / "out")
    payload = json.loads(open(output["json"], encoding="utf-8").read())
    assert payload["ready_for_training"] is False
    assert "NaN" not in json.dumps(payload)


def test_detect_all_positive_synthetic_labels(tmp_path):
    path = tmp_path / "all_positive.npz"
    labels = tmp_path / "labels.csv"
    _npz(path, np.ones(10))
    pd.DataFrame(
        {
            "phase": ["incident"] * 10,
            "scenario_name": ["flash_crowd"] * 10,
            "true_alert_label": [1] * 10,
            "severity": [0.9] * 10,
        }
    ).to_csv(labels, index=False)
    result = analyze_windows_artifact(path, threshold=0.7, source_manifest={"valid_source_count": 1}, labels_path=labels)
    assert result["label_summary"]["positive_ratio"] == 1.0
    assert "synthetic_positive_ratio_is_100_percent" in result["warnings"]
    assert "synthetic_negative_count_is_zero" in result["warnings"]


def test_balanced_synthetic_labels_have_phase_distribution(tmp_path):
    path = tmp_path / "balanced.npz"
    labels = tmp_path / "labels.csv"
    _npz(path, np.r_[np.zeros(7), np.ones(3)])
    pd.DataFrame(
        {
            "phase": ["background", "background", "pre_incident", "incident", "recovery"] * 2,
            "scenario_name": ["flash_crowd", "burst_traffic"] * 5,
            "true_alert_label": [0, 0, 0, 1, 1] * 2,
            "severity": [0, 0, 0.2, 0.8, 0.3] * 2,
        }
    ).to_csv(labels, index=False)
    result = analyze_windows_artifact(path, threshold=0.7, source_manifest={"valid_source_count": 1}, labels_path=labels)
    assert result["label_summary"]["negative_count"] > 0
    assert result["label_summary"]["phase_distribution"]["background"] == 4
    assert result["label_summary"]["scenario_distribution"]["flash_crowd"] == 5
