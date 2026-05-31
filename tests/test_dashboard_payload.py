import json

from src.training.trainer import write_dashboard_payload


def test_dashboard_payload_has_required_keys(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "target_distribution.json").write_text('{"splits": {"test": {"count": 1}}}', encoding="utf-8")
    results = [
        {
            "model": "m",
            "category": "baseline",
            "status": "success",
            "train_time_seconds": 0.1,
            "inference_time_seconds": 0.01,
            "metrics": {"mae": 0.1, "rmse": 0.2, "r2": 0.3},
            "alert_metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 1.0, "tp": 0, "fp": 0, "tn": 1, "fn": 0},
            "alert_threshold": 0.5,
            "alert_threshold_mode": "quantile",
            "alert_threshold_value": 0.9,
            "alert_threshold_reference_split": "val",
            "prediction_path": "pred.csv",
            "history_path": "history.json",
            "model_path": "model.keras",
            "data_summary": {"train": {"X_shape": [10, 6, 3]}},
        }
    ]
    json_path, csv_path = write_dashboard_payload(results, "data.npz", tmp_path)
    payload = json.loads(open(json_path, encoding="utf-8").read())
    assert {"run_id", "model_comparison", "best_model", "metrics_by_model"} <= set(payload)
    assert payload["threshold_info"]["mode"] == "quantile"
    assert payload["target_distribution_summary"]["splits"]["test"]["count"] == 1
    assert payload["best_model"] == "m"
    assert open(csv_path, encoding="utf-8").read().startswith("model,")
