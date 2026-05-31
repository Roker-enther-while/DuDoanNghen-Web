"""Naive last-value baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class NaiveLastValueBaseline:
    """Predict with the last observed proxy pressure feature in each window."""

    def __init__(self, feature_name: str = "congestion_score_proxy"):
        self.feature_name = feature_name
        self.feature_index: int | None = None
        self.fallback_value = 0.0

    def fit(self, X, y, feature_columns=None):
        columns = [str(c) for c in ([] if feature_columns is None else list(feature_columns))]
        if self.feature_name in columns:
            self.feature_index = columns.index(self.feature_name)
        else:
            pressure_names = ["request_count", "bytes_sum", "unique_hosts", "error_rate", "request_spike_score"]
            indices = [columns.index(name) for name in pressure_names if name in columns]
            self.feature_index = indices[0] if indices else None
        self.fallback_value = float(np.mean(y)) if len(y) else 0.0
        return self

    def predict(self, X):
        if self.feature_index is None:
            pred = np.full((len(X),), self.fallback_value, dtype=np.float32)
        else:
            pred = X[:, -1, self.feature_index].astype(np.float32)
        return np.clip(pred.reshape(-1), 0.0, 1.0)

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / "baseline.json"
        model_path.write_text(
            json.dumps(
                {
                    "type": "naive_last_value",
                    "feature_name": self.feature_name,
                    "feature_index": self.feature_index,
                    "fallback_value": self.fallback_value,
                    "note": "Simple comparison baseline, not an optimized forecasting model.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return str(model_path)


def build_model(config: dict | None = None):
    config = config or {}
    return NaiveLastValueBaseline(feature_name=config.get("feature_name", "congestion_score_proxy"))
