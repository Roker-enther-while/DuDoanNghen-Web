"""Moving-average baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class MovingAverageBaseline:
    """Predict with the mean of the last N observed proxy pressure values."""

    def __init__(self, average_steps: int = 5, feature_name: str = "congestion_score_proxy"):
        self.average_steps = average_steps
        self.feature_name = feature_name
        self.feature_index: int | None = None
        self.fallback_value = 0.0

    def fit(self, X, y, feature_columns=None):
        columns = [str(c) for c in ([] if feature_columns is None else list(feature_columns))]
        self.feature_index = columns.index(self.feature_name) if self.feature_name in columns else None
        self.fallback_value = float(np.mean(y)) if len(y) else 0.0
        return self

    def predict(self, X):
        if self.feature_index is None:
            pred = np.full((len(X),), self.fallback_value, dtype=np.float32)
        else:
            steps = min(self.average_steps, X.shape[1])
            pred = np.mean(X[:, -steps:, self.feature_index], axis=1).astype(np.float32)
        return np.clip(pred.reshape(-1), 0.0, 1.0)

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / "baseline.json"
        model_path.write_text(
            json.dumps(
                {
                    "type": "moving_average",
                    "average_steps": self.average_steps,
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
    return MovingAverageBaseline(
        average_steps=int(config.get("average_steps", 5)),
        feature_name=config.get("feature_name", "congestion_score_proxy"),
    )
