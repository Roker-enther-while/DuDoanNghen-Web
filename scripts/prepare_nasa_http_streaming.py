"""Streaming NASA HTTP pipeline — uses string-based aggregation to avoid pandas 3.0 + Python 3.14 crash."""

from __future__ import annotations

import gzip
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.nasa_http import add_congestion_proxy_target
from src.data.preprocess import chronological_split, normalize_splits, save_scaler
from src.data.schemas import NASA_FEATURE_COLUMNS, NASA_TARGET_COLUMN
from src.data.windowing import build_windows_from_normalized

LOG_PATTERN = re.compile(
    r'^(?P<host>\S+)\s+\S+\s+\S+\s+\[(?P<timestamp>[^\]]+)\]\s+'
    r'"(?P<request>[^"]*)"\s+(?P<status>\S+)\s+(?P<bytes>\S+)'
)


def stream_parse_and_aggregate(input_paths: list[Path], window_minutes: int = 1) -> pd.DataFrame:
    """Parse logs and aggregate to 1-min windows using string keys (avoids pandas Timestamp crash)."""
    windows: dict[str, dict] = {}
    total_lines = 0
    parsed_lines = 0

    for path in input_paths:
        print(f"  Processing {path.name}...", flush=True)
        with gzip.open(path, "rt", encoding="latin-1", errors="replace") as f:
            for line in f:
                total_lines += 1
                match = LOG_PATTERN.match(line.strip())
                if not match:
                    continue
                parsed_lines += 1

                try:
                    dt = datetime.strptime(match.group("timestamp"), "%d/%b/%Y:%H:%M:%S %z")
                    ws_str = dt.strftime("%Y-%m-%d %H:%M:00")
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
                host = match.group("host")

                if ws_str not in windows:
                    windows[ws_str] = {
                        "request_count": 0, "bytes_sum": 0, "bytes_vals": [],
                        "hosts": set(),
                        "s2": 0, "s3": 0, "s4": 0, "s5": 0,
                        "m_get": 0, "m_post": 0, "m_head": 0, "m_other": 0,
                    }
                w = windows[ws_str]
                w["request_count"] += 1
                w["bytes_sum"] += bytes_val
                if len(w["bytes_vals"]) < 5000:
                    w["bytes_vals"].append(bytes_val)
                w["hosts"].add(host)

                sc = status // 100
                if sc == 2: w["s2"] += 1
                elif sc == 3: w["s3"] += 1
                elif sc == 4: w["s4"] += 1
                elif sc == 5: w["s5"] += 1

                if method == "GET": w["m_get"] += 1
                elif method == "POST": w["m_post"] += 1
                elif method == "HEAD": w["m_head"] += 1
                else: w["m_other"] += 1

                if parsed_lines % 500000 == 0:
                    print(f"    ...{parsed_lines} parsed, {len(windows)} windows", flush=True)

    print(f"  Total: {parsed_lines} parsed, {len(windows)} windows", flush=True)

    # Build rows using plain values (no Timestamps in list)
    rows = []
    for ws_str in sorted(windows.keys()):
        w = windows[ws_str]
        rc = w["request_count"]
        bv = w["bytes_vals"]
        ec = w["s4"] + w["s5"]
        rows.append((
            ws_str, rc, w["bytes_sum"],
            w["bytes_sum"] / max(rc, 1),
            max(bv) if bv else 0,
            float(np.percentile(bv, 95)) if bv else 0,
            len(w["hosts"]),
            w["s2"], w["s3"], w["s4"], w["s5"],
            ec, ec / max(rc, 1),
            w["m_get"], w["m_post"], w["m_head"], w["m_other"],
            w["bytes_sum"] / float(window_minutes),
        ))
    del windows

    cols = [
        "timestamp_str", "request_count", "bytes_sum", "bytes_mean", "bytes_max", "bytes_p95",
        "unique_hosts", "status_2xx", "status_3xx", "status_4xx", "status_5xx",
        "error_count", "error_rate", "method_get", "method_post", "method_head",
        "method_other", "throughput_bytes_per_min",
    ]
    df = pd.DataFrame(rows, columns=cols)

    # Convert string timestamps to proper datetime AFTER DataFrame creation
    df["timestamp"] = pd.to_datetime(df["timestamp_str"], utc=True)
    df = df.drop(columns=["timestamp_str"])

    # Sort and clean
    df = df.sort_values("timestamp").reset_index(drop=True)
    numeric_cols = [c for c in df.columns if c != "timestamp"]
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0).fillna(0)

    columns = [
        "timestamp", "request_count", "bytes_sum", "bytes_mean", "bytes_max", "bytes_p95",
        "unique_hosts", "status_2xx", "status_3xx", "status_4xx", "status_5xx",
        "error_count", "error_rate", "method_get", "method_post", "method_head",
        "method_other", "throughput_bytes_per_min",
    ]
    return df[columns]


def main():
    input_paths = [
        ROOT / "data" / "raw" / "nasa_http" / "NASA_access_log_Jul95.gz",
        ROOT / "data" / "raw" / "nasa_http" / "NASA_access_log_Aug95.gz",
    ]
    output_root = ROOT / "data" / "processed" / "nasa_http_3m"
    output_root.mkdir(parents=True, exist_ok=True)

    lookback_steps = 60
    horizon_steps = 15

    print("Step 1: Streaming parse + aggregate...", flush=True)
    timeseries = stream_parse_and_aggregate(input_paths, window_minutes=1)
    print(f"  Windows: {len(timeseries)}", flush=True)

    print("Step 2: Adding proxy target...", flush=True)
    # Workaround: pandas 3.0.4 + Python 3.14 crashes on iloc/loc with Timestamp columns at 80K+ rows
    # Convert timestamp to string for all heavy operations, convert back at the end
    ts_strings = timeseries["timestamp"].astype(str).values
    work = timeseries.drop(columns=["timestamp"]).copy()

    request_count = pd.to_numeric(work["request_count"], errors="coerce").fillna(0)
    rolling_mean = request_count.rolling(60, min_periods=2).mean().shift(1)
    rolling_std = request_count.rolling(60, min_periods=2).std().shift(1).replace(0, np.nan)
    z_score = ((request_count - rolling_mean) / rolling_std).replace([np.inf, -np.inf], 0).fillna(0)
    work["request_spike_score"] = (z_score.clip(lower=0, upper=3) / 3.0).clip(0, 1)

    def expanding_minmax(series):
        values = pd.to_numeric(series, errors="coerce").fillna(0).astype(float)
        running_min = values.cummin()
        running_max = values.cummax()
        denom = (running_max - running_min).replace(0, np.nan)
        return ((values - running_min) / denom).fillna(0).clip(0, 1)

    score = (
        expanding_minmax(work["request_count"]) * 0.35 +
        expanding_minmax(work["bytes_sum"]) * 0.20 +
        expanding_minmax(work["unique_hosts"]) * 0.15 +
        work["error_rate"].clip(0, 1) * 0.20 +
        work["request_spike_score"] * 0.10
    )
    work["congestion_score_proxy"] = score.clip(0, 1)
    work["target_next_congestion_score"] = work["congestion_score_proxy"].shift(-horizon_steps)

    # Drop NaN targets using numpy (no Timestamp column present, so safe)
    target_vals = work["target_next_congestion_score"].values
    valid_mask = ~np.isnan(target_vals)
    valid_indices = np.where(valid_mask)[0]
    work = work.iloc[valid_indices].copy()
    work.index = range(len(work))
    ts_strings = ts_strings[valid_indices]
    print(f"  After target shift: {len(work)}", flush=True)

    # Convert timestamps back
    timeseries = work.copy()
    timeseries.insert(0, "timestamp", pd.to_datetime(ts_strings, utc=True))

    target_meta = {
        "target_mode": "congestion_proxy",
        "target_column": "target_next_congestion_score",
        "horizon_steps": horizon_steps,
        "is_proxy_target": True,
    }

    ts_csv = output_root / "timeseries_1min.csv"
    timeseries.to_csv(ts_csv, index=False)
    print(f"  Saved: {ts_csv}", flush=True)

    print("Step 3: Chronological split...", flush=True)
    train, val, test, split_meta = chronological_split(timeseries)
    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}", flush=True)

    print("Step 4: Normalizing...", flush=True)
    scaler, norm_outputs = normalize_splits(
        train, val, test, NASA_FEATURE_COLUMNS, NASA_TARGET_COLUMN,
        output_root / "normalized",
    )
    save_scaler(output_root / "scaler.json", scaler)
    print("  Normalized", flush=True)

    print("Step 5: Building windows...", flush=True)
    window_info = build_windows_from_normalized(
        {split: paths["fp16"] for split, paths in norm_outputs.items()},
        output_root / "windows" / "windows_fp16.npz",
        lookback_steps,
        {"lookback_steps": lookback_steps, "horizon_steps": horizon_steps, "allow_context_overlap": False},
    )
    print(f"  Window info: {json.dumps(window_info, indent=2)}", flush=True)

    with np.load(output_root / "windows" / "windows_fp16.npz", allow_pickle=True) as data:
        for key in data.files:
            print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}", flush=True)

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
