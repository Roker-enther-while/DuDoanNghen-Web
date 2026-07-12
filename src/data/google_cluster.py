"""Skeleton adapter for Google Cluster Trace samples or BigQuery exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_google_cluster_sample(path: str | Path) -> pd.DataFrame:
    """Read a small local CSV export; no public Google trace is downloaded here."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Google Cluster sample not found: {path}")
    return pd.read_csv(path)


def aggregate_google_cluster_sample(df: pd.DataFrame, window_minutes: int = 5) -> pd.DataFrame:
    """Aggregate a local Google Cluster sample to the common resource-pressure schema."""
    if "timestamp" not in df:
        raise ValueError("Input must include a timestamp column")
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    work["window_start"] = work["timestamp"].dt.floor(f"{window_minutes}min")
    for col in [
        "cpu_usage",
        "memory_usage",
        "failed_task_count",
        "evicted_task_count",
        "cpu_request",
        "memory_request",
    ]:
        if col not in work:
            work[col] = 0.0

    grouped = work.groupby("window_start", sort=True)
    result = pd.DataFrame(index=grouped.size().index)
    result["cpu_usage"] = grouped["cpu_usage"].mean()
    result["memory_usage"] = grouped["memory_usage"].mean()
    result["task_count"] = grouped.size()
    result["failed_task_count"] = grouped["failed_task_count"].sum()
    result["evicted_task_count"] = grouped["evicted_task_count"].sum()
    result["mean_cpu_request"] = grouped["cpu_request"].mean()
    result["mean_memory_request"] = grouped["memory_request"].mean()
    pressure = (
        result["cpu_usage"].rank(pct=True) * 0.4
        + result["memory_usage"].rank(pct=True) * 0.4
        + (result["failed_task_count"] + result["evicted_task_count"]).rank(pct=True) * 0.2
    )
    result["resource_pressure_score"] = pressure.clip(0, 1)
    result["target_next_resource_pressure_score"] = result["resource_pressure_score"].shift(-1)
    result = result.dropna(subset=["target_next_resource_pressure_score"]).reset_index()
    result = result.rename(columns={"window_start": "timestamp"})
    return result


def google_cluster_guidance() -> str:
    """Return safe usage guidance for Google Cluster 2019."""
    return (
        "Google Cluster 2019 is not downloaded by this pipeline. Use BigQuery or a small CSV export, "
        "then pass it with --input. Example BigQuery workflow: select a bounded time range and only "
        "needed telemetry columns, export to CSV/Parquet, then run scripts/prepare_google_cluster.py."
    )
