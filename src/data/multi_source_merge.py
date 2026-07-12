"""Merge approved public web-log datasets while preserving source identity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def chronological_within_source_split(
    df: pd.DataFrame,
    source_col: str = "source_id",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split each source chronologically, then concatenate split partitions."""
    train_parts = []
    val_parts = []
    test_parts = []
    for _, group in df.sort_values([source_col, "timestamp"]).groupby(source_col, sort=True):
        ordered = group.sort_values("timestamp").reset_index(drop=True)
        n = len(ordered)
        train_end = max(1, int(n * train_ratio))
        val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))
        if val_end >= n:
            val_end = n - 1
        train_parts.append(ordered.iloc[:train_end])
        val_parts.append(ordered.iloc[train_end:val_end])
        test_parts.append(ordered.iloc[val_end:])
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(val_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def leave_one_source_out_split(
    df: pd.DataFrame,
    test_source_id: str,
    source_col: str = "source_id",
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train on all but one source; reserve one source for test."""
    test = df[df[source_col] == test_source_id].sort_values("timestamp").reset_index(drop=True)
    train_val = df[df[source_col] != test_source_id].sort_values([source_col, "timestamp"]).reset_index(drop=True)
    if train_val.empty or test.empty:
        raise ValueError("leave_one_source_out requires at least one train source and one test source")
    train_parts = []
    val_parts = []
    for _, group in train_val.groupby(source_col, sort=True):
        ordered = group.sort_values("timestamp").reset_index(drop=True)
        split_at = max(1, int(len(ordered) * (1.0 - val_ratio)))
        if split_at >= len(ordered):
            split_at = len(ordered) - 1
        train_parts.append(ordered.iloc[:split_at])
        val_parts.append(ordered.iloc[split_at:])
    return pd.concat(train_parts, ignore_index=True), pd.concat(val_parts, ignore_index=True), test


def _load_windows(path: str | Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def merge_window_artifacts(inputs: Sequence[dict], output_path: str | Path) -> dict:
    """Concatenate window artifacts and store source labels per split."""
    if not inputs:
        raise ValueError("At least one window artifact is required")
    payload: dict[str, np.ndarray] = {}
    source_distribution: dict[str, dict[str, int]] = {}
    feature_columns = None
    target_column = None
    for split in ["train", "val", "test"]:
        X_parts = []
        y_parts = []
        ts_parts = []
        source_parts = []
        source_distribution[split] = {}
        for item in inputs:
            source_id = item["source_id"]
            data = _load_windows(item["path"])
            X = data[f"X_{split}"]
            y = data[f"y_{split}"]
            ts = data.get(f"ts_{split}", np.arange(len(y)))
            X_parts.append(X)
            y_parts.append(y)
            ts_parts.append(ts)
            source_parts.append(np.array([source_id] * len(y), dtype=object))
            source_distribution[split][source_id] = int(len(y))
            feature_columns = data.get("feature_columns", feature_columns)
            target_column = data.get("target_column", target_column)
        payload[f"X_{split}"] = np.concatenate(X_parts, axis=0).astype(np.float16)
        payload[f"y_{split}"] = np.concatenate(y_parts, axis=0).astype(np.float16)
        payload[f"ts_{split}"] = np.concatenate(ts_parts, axis=0)
        payload[f"source_id_{split}"] = np.concatenate(source_parts, axis=0)

    payload["feature_columns"] = feature_columns
    payload["target_column"] = target_column
    payload["config"] = np.array(
        json.dumps({"merge_mode": "concat_preserve_source_id", "sources": [item["source_id"] for item in inputs]}),
        dtype=object,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return {
        "path": str(output_path),
        "sources": [item["source_id"] for item in inputs],
        "source_distribution": source_distribution,
        "shapes": {split: list(payload[f"X_{split}"].shape) for split in ["train", "val", "test"]},
    }


def merge_readiness(source_ids: Sequence[str]) -> dict:
    """Summarize whether a merged artifact supports cross-source claims."""
    unique_sources = sorted(set(source_ids))
    single_source = len(unique_sources) == 1
    return {
        "source_count": len(unique_sources),
        "sources": unique_sources,
        "warnings": ["multi_source_contains_single_source_only"] if single_source else [],
        "ready_for_training": len(unique_sources) >= 1,
        "ready_for_cross_source_claim": len(unique_sources) >= 2,
        "ready_for_real_world_claim": False,
    }
