"""NASA HTTP access log parsing, aggregation, and proxy target creation."""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd


LOG_PATTERN = re.compile(
    r'^(?P<host>\S+)\s+\S+\s+\S+\s+\[(?P<timestamp>[^\]]+)\]\s+'
    r'"(?P<request>[^"]*)"\s+(?P<status>\S+)\s+(?P<bytes>\S+)'
)


@dataclass
class ParseStats:
    total_lines: int = 0
    parsed_lines: int = 0
    skipped_lines: int = 0

    def to_dict(self) -> dict:
        return {
            "total_lines": self.total_lines,
            "parsed_lines": self.parsed_lines,
            "skipped_lines": self.skipped_lines,
        }


def parse_nasa_log_line(line: str) -> dict | None:
    """Parse one NASA access log line into a normalized request record."""
    match = LOG_PATTERN.match(line.strip())
    if not match:
        return None

    timestamp_text = match.group("timestamp")
    try:
        timestamp = pd.Timestamp(datetime.strptime(timestamp_text, "%d/%b/%Y:%H:%M:%S %z")).tz_convert("UTC")
    except Exception:
        return None

    request = match.group("request").strip()
    request_parts = request.split()
    method = request_parts[0].upper() if len(request_parts) >= 1 else "OTHER"
    path = request_parts[1] if len(request_parts) >= 2 else ""
    protocol = request_parts[2] if len(request_parts) >= 3 else ""

    status_text = match.group("status")
    try:
        status = int(status_text)
    except ValueError:
        status = np.nan

    bytes_text = match.group("bytes")
    try:
        byte_count = int(bytes_text) if bytes_text != "-" else 0
    except ValueError:
        byte_count = 0

    return {
        "host": match.group("host"),
        "timestamp": timestamp,
        "method": method,
        "path": path,
        "protocol": protocol,
        "status": status,
        "bytes": byte_count,
    }


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="latin-1", errors="replace")
    return path.open("rt", encoding="latin-1", errors="replace")


def iter_nasa_log_records(path: str | Path, limit_lines: int | None = None) -> tuple[pd.DataFrame, ParseStats]:
    """Read a text/gzip NASA log file and return parsed records plus line stats."""
    stats = ParseStats()
    records: list[dict] = []
    path = Path(path)
    with _open_text(path) as handle:
        for line in handle:
            if limit_lines is not None and stats.total_lines >= limit_lines:
                break
            stats.total_lines += 1
            record = parse_nasa_log_line(line)
            if record is None:
                stats.skipped_lines += 1
                continue
            stats.parsed_lines += 1
            records.append(record)

    return pd.DataFrame.from_records(records), stats


def records_from_lines(lines: Sequence[str]) -> tuple[pd.DataFrame, ParseStats]:
    """Parse in-memory log lines; used by tests and offline smoke runs."""
    stats = ParseStats()
    records = []
    for line in lines:
        stats.total_lines += 1
        record = parse_nasa_log_line(line)
        if record is None:
            stats.skipped_lines += 1
            continue
        stats.parsed_lines += 1
        records.append(record)
    return pd.DataFrame.from_records(records), stats


def aggregate_requests(df: pd.DataFrame, window_minutes: int = 1) -> pd.DataFrame:
    """Aggregate request-level records into an evenly spaced time series."""
    if window_minutes <= 0:
        raise ValueError("window_minutes must be positive")

    columns = [
        "timestamp",
        "request_count",
        "bytes_sum",
        "bytes_mean",
        "bytes_max",
        "bytes_p95",
        "unique_hosts",
        "status_2xx",
        "status_3xx",
        "status_4xx",
        "status_5xx",
        "error_count",
        "error_rate",
        "method_get",
        "method_post",
        "method_head",
        "method_other",
        "throughput_bytes_per_min",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    freq = f"{window_minutes}min"
    work["window_start"] = work["timestamp"].dt.floor(freq)
    work["status"] = pd.to_numeric(work["status"], errors="coerce")
    work["bytes"] = pd.to_numeric(work["bytes"], errors="coerce").fillna(0)
    work["method"] = work["method"].fillna("OTHER").str.upper()

    grouped = work.groupby("window_start", sort=True)
    result = pd.DataFrame(index=grouped.size().index)
    result["request_count"] = grouped.size()
    result["bytes_sum"] = grouped["bytes"].sum()
    result["bytes_mean"] = grouped["bytes"].mean()
    result["bytes_max"] = grouped["bytes"].max()
    result["bytes_p95"] = grouped["bytes"].quantile(0.95)
    result["unique_hosts"] = grouped["host"].nunique()

    status_class = (work["status"] // 100).astype("Int64")
    work["status_class"] = status_class
    status_counts = pd.crosstab(work["window_start"], work["status_class"])
    for klass in [2, 3, 4, 5]:
        result[f"status_{klass}xx"] = status_counts.get(klass, pd.Series(0, index=result.index)).reindex(result.index).fillna(0)

    result["error_count"] = result["status_4xx"] + result["status_5xx"]
    result["error_rate"] = np.where(result["request_count"] > 0, result["error_count"] / result["request_count"], 0.0)
    method_counts = pd.crosstab(work["window_start"], work["method"])
    for method in ["GET", "POST", "HEAD"]:
        result[f"method_{method.lower()}"] = method_counts.get(method, pd.Series(0, index=result.index)).reindex(result.index).fillna(0)
    known_methods = {"GET", "POST", "HEAD"}
    result["method_other"] = grouped["method"].apply(lambda values: int((~values.isin(known_methods)).sum()))
    result["throughput_bytes_per_min"] = result["bytes_sum"] / float(window_minutes)

    full_index = pd.date_range(result.index.min(), result.index.max(), freq=freq, tz="UTC")
    result = result.reindex(full_index)
    fill_zero = [col for col in result.columns]
    result[fill_zero] = result[fill_zero].fillna(0)
    result.index.name = "timestamp"
    result = result.reset_index()

    numeric_cols = [col for col in result.columns if col != "timestamp"]
    result[numeric_cols] = result[numeric_cols].replace([np.inf, -np.inf], 0).fillna(0)
    return result[columns]


def _expanding_minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0).astype(float)
    running_min = values.cummin()
    running_max = values.cummax()
    denom = (running_max - running_min).replace(0, np.nan)
    return ((values - running_min) / denom).fillna(0).clip(0, 1)


def add_congestion_proxy_target(
    df: pd.DataFrame,
    horizon_steps: int = 15,
    rolling_window: int = 60,
) -> tuple[pd.DataFrame, dict]:
    """Create a leakage-conscious proxy congestion score and shifted target."""
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")
    work = df.copy()
    request_count = pd.to_numeric(work["request_count"], errors="coerce").fillna(0)
    rolling_mean = request_count.rolling(rolling_window, min_periods=2).mean().shift(1)
    rolling_std = request_count.rolling(rolling_window, min_periods=2).std().shift(1).replace(0, np.nan)
    z_score = ((request_count - rolling_mean) / rolling_std).replace([np.inf, -np.inf], 0).fillna(0)
    work["request_spike_score"] = (z_score.clip(lower=0, upper=3) / 3.0).clip(0, 1)

    weights = {
        "request_count": 0.35,
        "bytes_sum": 0.20,
        "unique_hosts": 0.15,
        "error_rate": 0.20,
        "request_spike_score": 0.10,
    }
    components = {
        "request_count": _expanding_minmax(work["request_count"]),
        "bytes_sum": _expanding_minmax(work["bytes_sum"]),
        "unique_hosts": _expanding_minmax(work["unique_hosts"]),
        "error_rate": pd.to_numeric(work["error_rate"], errors="coerce").fillna(0).clip(0, 1),
        "request_spike_score": work["request_spike_score"],
    }
    score = sum(components[name] * weight for name, weight in weights.items())
    work["congestion_score_proxy"] = score.clip(0, 1)
    work["target_next_congestion_score"] = work["congestion_score_proxy"].shift(-horizon_steps)
    work = work.dropna(subset=["target_next_congestion_score"]).reset_index(drop=True)
    metadata = {
        "target_mode": "congestion_proxy",
        "target_column": "target_next_congestion_score",
        "horizon_steps": horizon_steps,
        "is_proxy_target": True,
        "proxy_notice": "NASA HTTP logs do not contain true congestion labels, CPU, memory, or response time.",
        "formula": {
            "weights": weights,
            "normalization": "expanding min-max for request_count, bytes_sum, unique_hosts; error_rate already bounded; rolling z-score spike uses past window only",
        },
    }
    return work, metadata
