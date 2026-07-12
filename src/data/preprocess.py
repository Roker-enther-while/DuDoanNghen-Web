"""Chronological splitting and train-only normalization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Split a time series in chronological order without shuffle."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")
    if len(df) < 3:
        raise ValueError("Need at least 3 rows for train/validation/test split")

    ordered = df.sort_values("timestamp").reset_index(drop=True)
    n = len(ordered)
    train_end = max(1, int(n * train_ratio))
    val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))
    if val_end >= n:
        val_end = n - 1
    train = ordered.iloc[:train_end].reset_index(drop=True)
    val = ordered.iloc[train_end:val_end].reset_index(drop=True)
    test = ordered.iloc[val_end:].reset_index(drop=True)
    metadata = {
        "split": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "chronological": True,
        "shuffle": False,
        "leak_check": {
            "timestamp_overlap": False,
            "scaler_fit_scope": "train_only",
        },
    }
    return train, val, test, metadata


def fit_minmax_scaler(train_df: pd.DataFrame, feature_columns: Sequence[str], target_column: str) -> dict:
    """Fit min-max statistics using train rows only."""
    scaler = {"type": "minmax", "fit_scope": "train_only", "features": {}, "target": {}}
    for col in feature_columns:
        values = pd.to_numeric(train_df[col], errors="coerce").astype("float32")
        scaler["features"][col] = {"min": float(values.min()), "max": float(values.max())}
    target_values = pd.to_numeric(train_df[target_column], errors="coerce").astype("float32")
    scaler["target"][target_column] = {"min": float(target_values.min()), "max": float(target_values.max())}
    return scaler


def _scale_array(values: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    values = values.astype(np.float32)
    denom = max_value - min_value
    if denom == 0:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - min_value) / denom, 0.0, 1.0).astype(np.float32)


def transform_split(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    scaler: dict,
    output_dtype: np.dtype = np.float32,
) -> dict:
    """Normalize features and target with train-fitted stats."""
    feature_arrays = []
    for col in feature_columns:
        stats = scaler["features"][col]
        values = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        feature_arrays.append(_scale_array(values, stats["min"], stats["max"]))
    features = np.stack(feature_arrays, axis=1).astype(output_dtype)

    target_stats = scaler["target"][target_column]
    target_values = pd.to_numeric(df[target_column], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    target = _scale_array(target_values, target_stats["min"], target_stats["max"]).astype(output_dtype)
    timestamps = pd.to_datetime(df["timestamp"], utc=True).astype("int64").to_numpy()
    return {
        "features": features,
        "target": target,
        "timestamps": timestamps,
        "feature_columns": np.array(list(feature_columns), dtype=object),
        "target_column": np.array(target_column, dtype=object),
    }


def save_npz(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def save_scaler(path: str | Path, scaler: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scaler, indent=2), encoding="utf-8")


def normalize_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    output_dir: str | Path,
) -> tuple[dict, dict]:
    """Write fp32 and fp16 normalized NPZ files for each split."""
    output_dir = Path(output_dir)
    scaler = fit_minmax_scaler(train, feature_columns, target_column)
    outputs: dict[str, dict[str, str]] = {}
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        fp32 = transform_split(split_df, feature_columns, target_column, scaler, np.float32)
        fp16 = {
            key: (value.astype(np.float16) if key in {"features", "target"} else value)
            for key, value in fp32.items()
        }
        fp32_path = output_dir / f"{split_name}_fp32.npz"
        fp16_path = output_dir / f"{split_name}_fp16.npz"
        save_npz(fp32_path, fp32)
        save_npz(fp16_path, fp16)
        outputs[split_name] = {"fp32": str(fp32_path), "fp16": str(fp16_path)}
    return scaler, outputs
