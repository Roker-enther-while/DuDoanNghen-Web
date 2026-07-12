"""Load and validate model-ready sliding-window artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_KEYS = ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")
OPTIONAL_KEYS = ("ts_train", "ts_val", "ts_test", "feature_columns", "target_column", "config")


def load_window_data(path: str | Path) -> dict[str, Any]:
    """Load a windows NPZ artifact into memory."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Window data artifact not found: {path}")
    with np.load(path, allow_pickle=True) as npz:
        data = {key: npz[key] for key in npz.files}
    data["_path"] = str(path)
    return data


def validate_window_data(data: dict[str, Any], require_float16: bool = True) -> bool:
    """Validate required split keys, shapes, dtypes, and numeric sanity."""
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError(f"Window data missing required keys: {missing}")

    for split in ("train", "val", "test"):
        X = data[f"X_{split}"]
        y = data[f"y_{split}"]
        if X.ndim != 3:
            raise ValueError(f"X_{split} must have shape [samples, lookback_steps, num_features], got {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y_{split} must have shape [samples], got {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X_{split} and y_{split} sample counts differ: {X.shape[0]} vs {y.shape[0]}")
        if require_float16 and X.dtype != np.float16:
            raise ValueError(f"X_{split} must be float16 in the saved artifact, got {X.dtype}")
        if require_float16 and y.dtype != np.float16:
            raise ValueError(f"y_{split} must be float16 in the saved artifact, got {y.dtype}")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError(f"{split} split contains NaN or Inf")
    return True


def get_data_summary(data: dict[str, Any]) -> dict[str, Any]:
    """Return a compact summary for logging and manifests."""
    summary: dict[str, Any] = {"path": data.get("_path")}
    warnings: list[str] = []
    for split in ("train", "val", "test"):
        X = data[f"X_{split}"]
        y = data[f"y_{split}"]
        split_min = float(min(np.min(X), np.min(y))) if X.size and y.size else None
        split_max = float(max(np.max(X), np.max(y))) if X.size and y.size else None
        if split_min is not None and (split_min < -1e-6 or split_max > 1.0 + 1e-6):
            warnings.append(f"{split} values outside [0, 1]: min={split_min}, max={split_max}")
        summary[split] = {
            "X_shape": list(X.shape),
            "y_shape": list(y.shape),
            "X_dtype": str(X.dtype),
            "y_dtype": str(y.dtype),
            "min": split_min,
            "max": split_max,
            "nan_count": int(np.isnan(X).sum() + np.isnan(y).sum()),
            "inf_count": int(np.isinf(X).sum() + np.isinf(y).sum()),
        }
    summary["feature_columns"] = [str(x) for x in data.get("feature_columns", [])]
    summary["target_column"] = str(data.get("target_column", ""))
    summary["warnings"] = warnings
    return summary


def convert_to_train_dtype(data: dict[str, Any], dtype: str = "float32") -> dict[str, Any]:
    """Copy data and convert X/y arrays to the dtype used by a training framework."""
    converted = dict(data)
    np_dtype = np.dtype(dtype)
    for split in ("train", "val", "test"):
        converted[f"X_{split}"] = data[f"X_{split}"].astype(np_dtype)
        converted[f"y_{split}"] = data[f"y_{split}"].astype(np_dtype)
    return converted


def subset_window_data(data: dict[str, Any], max_samples: dict[str, int | None]) -> dict[str, Any]:
    """Take the first N samples of each chronological split without shuffling."""
    subset = dict(data)
    for split in ("train", "val", "test"):
        limit = max_samples.get(split)
        if limit is None or limit <= 0:
            continue
        n = min(int(limit), len(data[f"X_{split}"]))
        subset[f"X_{split}"] = data[f"X_{split}"][:n]
        subset[f"y_{split}"] = data[f"y_{split}"][:n]
        ts_key = f"ts_{split}"
        if ts_key in data:
            subset[ts_key] = data[ts_key][:n]
    return subset


def select_indices(n_samples: int, limit: int | None, strategy: str = "head") -> np.ndarray:
    """Select chronological/evenly spaced indices for evaluation sampling."""
    if limit is None or limit <= 0 or strategy == "full" or limit >= n_samples:
        return np.arange(n_samples)
    strategy = strategy or "head"
    if strategy == "head":
        return np.arange(min(limit, n_samples))
    if strategy == "tail":
        return np.arange(max(0, n_samples - limit), n_samples)
    if strategy == "evenly_spaced":
        return np.unique(np.linspace(0, n_samples - 1, num=min(limit, n_samples), dtype=int))
    raise ValueError(f"Unsupported evaluation sample strategy: {strategy}")


def subset_split(data: dict[str, Any], split: str, limit: int | None, strategy: str = "head") -> dict[str, np.ndarray]:
    """Return one split subset without shuffling."""
    indices = select_indices(len(data[f"X_{split}"]), limit, strategy)
    result = {
        "X": data[f"X_{split}"][indices],
        "y": data[f"y_{split}"][indices],
        "indices": indices,
    }
    ts_key = f"ts_{split}"
    if ts_key in data:
        result["timestamps"] = data[ts_key][indices]
    return result
