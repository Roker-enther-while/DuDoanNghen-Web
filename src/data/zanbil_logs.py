"""Zanbil / Nginx web access log parser with explicit PII handling."""

from __future__ import annotations

import hashlib
import gzip
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd

from src.data.nasa_http import add_congestion_proxy_target
from src.data.preprocess import chronological_split, normalize_splits
from src.data.windowing import build_windows_from_normalized


LOG_PATTERN = re.compile(
    r'^(?P<client>\S+)\s+\S+\s+\S+\s+\[(?P<timestamp>[^\]]+)\]\s+'
    r'"(?P<request>[^"]*)"\s+(?P<status>\S+)\s+(?P<bytes>\S+)'
    r'(?:\s+"(?P<referer>[^"]*)"\s+"(?P<user_agent>[^"]*)")?'
)


@dataclass
class ZanbilParseStats:
    total_lines: int = 0
    parsed_lines: int = 0
    skipped_lines: int = 0
    bad_line_samples: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        parse_rate = self.parsed_lines / self.total_lines if self.total_lines else 0.0
        return {
            "total_lines": self.total_lines,
            "parsed_lines": self.parsed_lines,
            "skipped_lines": self.skipped_lines,
            "parse_rate": parse_rate,
            "bad_line_samples": list(self.bad_line_samples),
        }


def hash_client_identifier(client: str, salt: str) -> str:
    """Hash IP/hostname before writing processed data."""
    return hashlib.sha256(f"{salt}:{client}".encode("utf-8")).hexdigest()


def strip_query_string(path: str) -> str:
    """Drop query and fragment from a URL path to reduce PII risk."""
    try:
        parts = urlsplit(path)
        if parts.scheme or parts.netloc:
            return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
        return parts.path or path.split("?", 1)[0]
    except Exception:
        return path.split("?", 1)[0]


def parse_zanbil_log_line(
    line: str,
    salt: str = "local-zanbil-salt",
    drop_user_agent: bool = True,
    drop_query_string: bool = True,
) -> dict | None:
    """Parse one Nginx common/combined access log line with privacy handling."""
    match = LOG_PATTERN.match(line.strip())
    if not match:
        return None
    try:
        timestamp = pd.Timestamp(datetime.strptime(match.group("timestamp"), "%d/%b/%Y:%H:%M:%S %z")).tz_convert("UTC")
    except Exception:
        try:
            timestamp = pd.to_datetime(match.group("timestamp"), utc=True)
        except Exception:
            return None

    request = match.group("request").strip()
    parts = request.split()
    method = parts[0].upper() if len(parts) >= 1 else "OTHER"
    path = parts[1] if len(parts) >= 2 else ""
    protocol = parts[2] if len(parts) >= 3 else ""
    if drop_query_string:
        path = strip_query_string(path)

    try:
        status = int(match.group("status"))
    except ValueError:
        status = np.nan
    bytes_text = match.group("bytes")
    try:
        byte_count = int(bytes_text) if bytes_text != "-" else 0
    except ValueError:
        byte_count = 0

    record = {
        "client_hash": hash_client_identifier(match.group("client"), salt),
        "timestamp": timestamp,
        "method": method,
        "path": path,
        "protocol": protocol,
        "status": status,
        "bytes": byte_count,
        "source_id": "zanbil_web_logs",
    }
    if not drop_user_agent:
        record["user_agent"] = match.group("user_agent") or ""
    return record


def open_zanbil_text(path: str | Path):
    """Open .log/.txt/.gz Zanbil-like raw logs as text."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".zip":
        raise ValueError("zip input must be imported first with scripts/import_zanbil_raw.py")
    if suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def sample_zanbil_lines(path: str | Path, max_lines: int = 5) -> list[str]:
    """Read a small raw sample without dumping the full log."""
    lines: list[str] = []
    with open_zanbil_text(path) as handle:
        for _, line in zip(range(max_lines), handle):
            lines.append(line.rstrip("\n"))
    return lines


def read_zanbil_records(
    path: str | Path,
    limit_lines: int | None = None,
    salt: str = "local-zanbil-salt",
    drop_user_agent: bool = True,
    drop_query_string: bool = True,
) -> tuple[pd.DataFrame, ZanbilParseStats]:
    """Read a local Zanbil access.log without releasing raw client identifiers."""
    stats = ZanbilParseStats()
    records: list[dict] = []
    bad_samples: list[str] = []
    with open_zanbil_text(path) as handle:
        for line in handle:
            if limit_lines is not None and stats.total_lines >= limit_lines:
                break
            stats.total_lines += 1
            record = parse_zanbil_log_line(line, salt, drop_user_agent, drop_query_string)
            if record is None:
                stats.skipped_lines += 1
                if len(bad_samples) < 5:
                    bad_samples.append(line.strip()[:240])
                continue
            stats.parsed_lines += 1
            records.append(record)
    stats.bad_line_samples = tuple(bad_samples)
    return pd.DataFrame.from_records(records), stats


def aggregate_zanbil_requests(df: pd.DataFrame, window_minutes: int = 1) -> pd.DataFrame:
    """Aggregate Zanbil requests to the common web-log time-series schema."""
    columns = [
        "timestamp",
        "request_count",
        "bytes_sum",
        "bytes_mean",
        "bytes_p95",
        "unique_clients",
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
        "source_id",
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
    result["bytes_p95"] = grouped["bytes"].quantile(0.95)
    result["unique_clients"] = grouped["client_hash"].nunique()
    work["status_class"] = (work["status"] // 100).astype("Int64")
    status_counts = pd.crosstab(work["window_start"], work["status_class"])
    for klass in [2, 3, 4, 5]:
        result[f"status_{klass}xx"] = status_counts.get(klass, pd.Series(0, index=result.index)).reindex(result.index).fillna(0)
    result["error_count"] = result["status_4xx"] + result["status_5xx"]
    result["error_rate"] = np.where(result["request_count"] > 0, result["error_count"] / result["request_count"], 0.0)
    method_counts = pd.crosstab(work["window_start"], work["method"])
    for method in ["GET", "POST", "HEAD"]:
        result[f"method_{method.lower()}"] = method_counts.get(method, pd.Series(0, index=result.index)).reindex(result.index).fillna(0)
    known = {"GET", "POST", "HEAD"}
    result["method_other"] = grouped["method"].apply(lambda values: int((~values.isin(known)).sum()))
    result["throughput_bytes_per_min"] = result["bytes_sum"] / float(window_minutes)
    result["source_id"] = "zanbil_web_logs"

    full_index = pd.date_range(result.index.min(), result.index.max(), freq=freq, tz="UTC")
    result = result.reindex(full_index)
    result["source_id"] = result["source_id"].fillna("zanbil_web_logs")
    numeric_cols = [col for col in result.columns if col != "source_id"]
    result[numeric_cols] = result[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)
    result.index.name = "timestamp"
    return result.reset_index()[columns]


def prepare_zanbil_dataset(
    input_path: str | Path,
    output_root: str | Path,
    window_minutes: int = 1,
    lookback_steps: int = 60,
    horizon_steps: int = 15,
    limit_lines: int | None = None,
    salt: str = "local-zanbil-salt",
    min_parse_rate: float = 0.20,
) -> dict:
    """Prepare local Zanbil access.log into normalized windows without raw PII."""
    records, stats = read_zanbil_records(input_path, limit_lines=limit_lines, salt=salt)
    stats_dict = stats.to_dict()
    if stats.total_lines and stats_dict["parse_rate"] < min_parse_rate:
        raise ValueError(
            f"parser_parse_rate_too_low: {stats_dict['parse_rate']:.4f} < {min_parse_rate}; "
            f"bad_line_samples={stats_dict['bad_line_samples']}"
        )
    if records.empty:
        raise ValueError("no_zanbil_records_parsed")
    timeseries = aggregate_zanbil_requests(records, window_minutes=window_minutes)
    target_input = timeseries.rename(columns={"unique_clients": "unique_hosts"}).copy()
    targeted, target_meta = add_congestion_proxy_target(target_input, horizon_steps=horizon_steps)
    targeted = targeted.rename(columns={"unique_hosts": "unique_clients"})

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    timeseries_path = output_root / f"timeseries_{window_minutes}min.csv"
    targeted.to_csv(timeseries_path, index=False)

    feature_columns = [
        col for col in targeted.columns
        if col not in {"timestamp", "source_id", "target_next_congestion_score"}
        and pd.api.types.is_numeric_dtype(targeted[col])
    ]
    train, val, test, split_meta = chronological_split(targeted, 0.70, 0.15, 0.15)
    normalized_dir = output_root / "normalized"
    scaler, normalized_paths = normalize_splits(
        train, val, test, feature_columns, "target_next_congestion_score", normalized_dir
    )
    windows_path = output_root / "windows" / "windows_fp16.npz"
    windows_meta = build_windows_from_normalized(
        {split: paths["fp16"] for split, paths in normalized_paths.items()},
        windows_path,
        lookback_steps,
        {"source_id": "zanbil_web_logs", "target": target_meta, "window_minutes": window_minutes},
    )
    manifest = {
        "source_id": "zanbil_web_logs",
        "parse_stats": stats_dict,
        "timeseries_csv": str(timeseries_path),
        "windows": windows_meta,
        "split": split_meta,
        "feature_columns": feature_columns,
        "scaler_fit_scope": scaler.get("fit_scope"),
    }
    (output_root / "zanbil_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
