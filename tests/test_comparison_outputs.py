import json
import math

from src.training.trainer import write_model_comparison


def _results():
    return [
        {
            "model": "naive_last_value",
            "category": "baseline",
            "status": "success",
            "train_time_seconds": 0.1,
            "inference_time_seconds": 0.01,
            "metrics": {"mae": 0.1, "rmse": 0.2, "r2": 0.3},
            "alert_metrics": {"precision": 1.0, "recall": 0.5, "f1": 0.667, "accuracy": 0.75, "tp": 1, "fp": 0, "tn": 2, "fn": 1},
            "prediction_path": "pred.csv",
            "model_path": "model.json",
            "history_path": "history.json",
            "data_summary": {"train": {"X_shape": [10, 6, 3]}},
        }
    ]


def test_comparison_outputs_json_and_markdown_without_nan(tmp_path):
    json_path, md_path = write_model_comparison(_results(), tmp_path)
    payload = json.loads(open(json_path, encoding="utf-8").read())
    assert payload["models"][0]["model"] == "naive_last_value"
    text = open(md_path, encoding="utf-8").read()
    assert "smoke/small artifact" in text
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
