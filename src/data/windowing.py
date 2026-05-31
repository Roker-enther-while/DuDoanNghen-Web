"""Sliding-window creation for sequence models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np


def create_sliding_windows(
    features: np.ndarray,
    target: np.ndarray,
    timestamps: np.ndarray,
    lookback_steps: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create strict within-split windows where y is aligned after the lookback."""
    if lookback_steps <= 0:
        raise ValueError("lookback_steps must be positive")
    if len(features) != len(target) or len(features) != len(timestamps):
        raise ValueError("features, target, and timestamps must have matching lengths")
    n_windows = len(features) - lookback_steps
    if n_windows <= 0:
        return (
            np.empty((0, lookback_steps, features.shape[1]), dtype=features.dtype),
            np.empty((0,), dtype=target.dtype),
            np.empty((0,), dtype=timestamps.dtype),
        )
    X = np.stack([features[i : i + lookback_steps] for i in range(n_windows)], axis=0)
    y = np.array([target[i + lookback_steps] for i in range(n_windows)], dtype=target.dtype)
    ts = np.array([timestamps[i + lookback_steps] for i in range(n_windows)], dtype=timestamps.dtype)
    return X.astype(features.dtype), y, ts


def load_npz(path: str | Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def build_windows_from_normalized(
    normalized_paths: Mapping[str, str | Path],
    output_path: str | Path,
    lookback_steps: int,
    config: dict,
) -> dict:
    """Build windows independently per split and save one model-ready NPZ."""
    payload = {}
    shapes = {}
    feature_columns = None
    target_column = None
    for split in ["train", "val", "test"]:
        data = load_npz(normalized_paths[split])
        X, y, ts = create_sliding_windows(data["features"], data["target"], data["timestamps"], lookback_steps)
        payload[f"X_{split}"] = X.astype(np.float16)
        payload[f"y_{split}"] = y.astype(np.float16)
        payload[f"ts_{split}"] = ts
        shapes[split] = {"X": list(X.shape), "y": list(y.shape)}
        feature_columns = data["feature_columns"]
        target_column = data["target_column"]

    payload["feature_columns"] = feature_columns
    payload["target_column"] = target_column
    payload["config"] = np.array(json.dumps(config), dtype=object)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return {"path": str(output_path), "shapes": shapes}
