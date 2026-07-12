"""Data quality reporting for prepared time series."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def _outlier_rate(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 0.0
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return float(((values < lower) | (values > upper)).mean())


def generate_quality_report(
    df: pd.DataFrame,
    parse_stats: dict | None = None,
    feature_columns: Sequence[str] | None = None,
) -> dict:
    """Return machine-readable quality metrics for a time-series dataframe."""
    parse_stats = parse_stats or {}
    feature_columns = list(feature_columns or [col for col in df.columns if col != "timestamp"])
    timestamps = pd.to_datetime(df["timestamp"], utc=True) if "timestamp" in df else pd.Series(dtype="datetime64[ns, UTC]")
    feature_stats = {}
    for col in feature_columns:
        if col not in df:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        feature_stats[col] = {
            "min": float(values.min()) if len(values) else 0.0,
            "max": float(values.max()) if len(values) else 0.0,
            "mean": float(values.mean()) if len(values) else 0.0,
            "std": float(values.std(ddof=0)) if len(values) else 0.0,
        }

    status_distribution = {
        key: int(pd.to_numeric(df[key], errors="coerce").fillna(0).sum())
        for key in ["status_2xx", "status_3xx", "status_4xx", "status_5xx"]
        if key in df
    }
    report = {
        "raw_rows_parsed": int(parse_stats.get("parsed_lines", len(df))),
        "raw_rows_skipped": int(parse_stats.get("skipped_lines", 0)),
        "window_count": int(len(df)),
        "time_range": {
            "start": timestamps.min().isoformat() if len(timestamps) else None,
            "end": timestamps.max().isoformat() if len(timestamps) else None,
        },
        "missing_timestamp_count": int(timestamps.isna().sum()) if len(timestamps) else 0,
        "duplicate_timestamp_count": int(timestamps.duplicated().sum()) if len(timestamps) else 0,
        "feature_stats": feature_stats,
        "outlier_rate": {
            col: _outlier_rate(df[col]) for col in ["request_count", "bytes_sum", "error_rate"] if col in df
        },
        "status_class_distribution": status_distribution,
        "notes": [
            "Dataset is suitable for minute-level HTTP workload time-series modeling.",
            "NASA HTTP logs do not include true CPU, memory, or response-time telemetry.",
            "Current congestion target is a proxy derived from request volume, bytes, hosts, errors, and spikes.",
        ],
    }
    return report


def write_quality_report(report: dict, json_path: str | Path, markdown_path: str | Path) -> None:
    """Write JSON and Markdown versions of a quality report."""
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Data Quality Report",
        "",
        f"- Raw rows parsed: {report['raw_rows_parsed']}",
        f"- Raw rows skipped: {report['raw_rows_skipped']}",
        f"- Window count: {report['window_count']}",
        f"- Time range: {report['time_range']['start']} to {report['time_range']['end']}",
        f"- Missing timestamp count: {report['missing_timestamp_count']}",
        f"- Duplicate timestamp count: {report['duplicate_timestamp_count']}",
        "",
        "## Outlier Rates",
    ]
    for key, value in report["outlier_rate"].items():
        lines.append(f"- {key}: {value:.6f}")
    lines.extend(["", "## Status Class Distribution"])
    for key, value in report["status_class_distribution"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Feature Statistics"])
    for key, stats in report["feature_stats"].items():
        lines.append(
            f"- {key}: min={stats['min']:.6f}, max={stats['max']:.6f}, "
            f"mean={stats['mean']:.6f}, std={stats['std']:.6f}"
        )
    lines.extend(["", "## Notes"])
    for note in report["notes"]:
        lines.append(f"- {note}")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
