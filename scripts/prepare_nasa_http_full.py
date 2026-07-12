"""Memory-efficient NASA HTTP pipeline for full dataset (3.46M records).

Handles the pandas 3.0 + numpy 2.4 memory issue by processing in chunks
and using more efficient aggregation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.nasa_http import LOG_PATTERN, add_congestion_proxy_target, _expanding_minmax
from src.data.preprocess import chronological_split, normalize_splits, save_scaler
from src.data.schemas import NASA_FEATURE_COLUMNS, NASA_TARGET_COLUMN
from src.data.windowing import build_windows_from_normalized


def parse_logs_chunked(input_paths: list[Path], chunk_size: int = 500000) -> pd.DataFrame:
    """Parse NASA logs in chunks to avoid memory spikes."""
    import gzip

    all_records = []
    for path in input_paths:
        print(f"  Parsing {path.name}...", flush=True)
        count = 0
        chunk_records = []
        with gzip.open(path, "rt", encoding="latin-1", errors="replace") as f:
            for line in f:
                match = LOG_PATTERN.match(line.strip())
                if not match:
                    continue
                try:
                    ts_text = match.group("timestamp")
                    from datetime import datetime
                    ts = pd.Timestamp(datetime.strptime(ts_text, "%d/%b/%Y:%H:%M:%S %z")).tz_convert("UTC")
                except Exception:
                    continue
                request = match.group("request").strip()
                parts = request.split()
                method = parts[0].upper() if parts else "OTHER"
                try:
                    status = int(match.group("status"))
                except ValueError:
                    continue
                bytes_val = int(match.group("bytes")) if match.group("bytes") != "-" else 0
                chunk_records.append({
                    "host": match.group("host"),
                    "timestamp": ts,
                    "method": method,
                    "status": status,
                    "bytes": bytes_val,
                })
                count += 1
                if count % chunk_size == 0:
                    all_records.extend(chunk_records)
                    chunk_records = []
                    print(f"    ...{count} records", flush=True)
        all_records.extend(chunk_records)
        print(f"  {path.name}: {count} records parsed", flush=True)

    return pd.DataFrame.from_records(all_records)


def aggregate_efficient(df: pd.DataFrame, window_minutes: int = 1) -> pd.DataFrame:
    """Efficient aggregation avoiding the memory-heavy crosstab."""
    freq = f"{window_minutes}min"
    work = df.copy()
    work["window_start"] = work["timestamp"].dt.floor(freq)
    work["status"] = pd.to_numeric(work["status"], errors="coerce").fillna(0).astype(int)
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

    # Status classes — compute directly without crosstab
    status_class = work["status"] // 100
    for klass in [2, 3, 4, 5]:
        mask = status_class == klass
        result[f"status_{klass}xx"] = work.loc[mask].groupby("window_start").size().reindex(result.index).fillna(0)

    result["error_count"] = result["status_4xx"] + result["status_5xx"]
    result["error_rate"] = np.where(result["request_count"] > 0, result["error_count"] / result["request_count"], 0.0)

    # Method counts
    for method in ["GET", "POST", "HEAD"]:
        mask = work["method"] == method
        result[f"method_{method.lower()}"] = work.loc[mask].groupby("window_start").size().reindex(result.index).fillna(0)
    known = {"GET", "POST", "HEAD"}
    mask = ~work["method"].isin(known)
    result["method_other"] = work.loc[mask].groupby("window_start").size().reindex(result.index).fillna(0)

    result["throughput_bytes_per_min"] = result["bytes_sum"] / float(window_minutes)

    # Fill gaps
    full_index = pd.date_range(result.index.min(), result.index.max(), freq=freq, tz="UTC")
    result = result.reindex(full_index).fillna(0)
    result.index.name = "timestamp"
    result = result.reset_index()

    # Clean infinities
    numeric_cols = [col for col in result.columns if col != "timestamp"]
    result[numeric_cols] = result[numeric_cols].replace([np.inf, -np.inf], 0).fillna(0)

    columns = [
        "timestamp", "request_count", "bytes_sum", "bytes_mean", "bytes_max", "bytes_p95",
        "unique_hosts", "status_2xx", "status_3xx", "status_4xx", "status_5xx",
        "error_count", "error_rate", "method_get", "method_post", "method_head",
        "method_other", "throughput_bytes_per_min",
    ]
    return result[columns]


def main():
    input_paths = [
        ROOT / "data" / "raw" / "nasa_http" / "NASA_access_log_Jul95.gz",
        ROOT / "data" / "raw" / "nasa_http" / "NASA_access_log_Aug95.gz",
    ]
    output_root = ROOT / "data" / "processed" / "nasa_http_3m"
    output_root.mkdir(parents=True, exist_ok=True)

    lookback_steps = 60
    horizon_steps = 15

    print("Step 1: Parsing logs...", flush=True)
    raw = parse_logs_chunked(input_paths)
    print(f"  Total records: {len(raw)}", flush=True)

    print("Step 2: Aggregating to 1-min windows...", flush=True)
    timeseries = aggregate_efficient(raw, window_minutes=1)
    print(f"  Windows: {len(timeseries)}", flush=True)

    print("Step 3: Adding proxy target...", flush=True)
    timeseries, target_meta = add_congestion_proxy_target(timeseries, horizon_steps=horizon_steps)
    print(f"  After target shift: {len(timeseries)}", flush=True)

    # Save timeseries
    ts_csv = output_root / "timeseries_1min.csv"
    timeseries.to_csv(ts_csv, index=False)
    print(f"  Saved: {ts_csv}", flush=True)

    print("Step 4: Chronological split...", flush=True)
    train, val, test, split_meta = chronological_split(timeseries)
    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}", flush=True)

    print("Step 5: Normalizing...", flush=True)
    scaler, norm_outputs = normalize_splits(
        train, val, test, NASA_FEATURE_COLUMNS, NASA_TARGET_COLUMN,
        output_root / "normalized",
    )
    save_scaler(output_root / "scaler.json", scaler)
    print("  Normalized", flush=True)

    print("Step 6: Building windows...", flush=True)
    window_info = build_windows_from_normalized(
        {split: paths["fp16"] for split, paths in norm_outputs.items()},
        output_root / "windows" / "windows_fp16.npz",
        lookback_steps,
        {"lookback_steps": lookback_steps, "horizon_steps": horizon_steps, "allow_context_overlap": False},
    )
    print(f"  Window info: {json.dumps(window_info, indent=2)}", flush=True)

    # Verify output
    with np.load(output_root / "windows" / "windows_fp16.npz", allow_pickle=True) as data:
        for key in data.files:
            print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}", flush=True)

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
