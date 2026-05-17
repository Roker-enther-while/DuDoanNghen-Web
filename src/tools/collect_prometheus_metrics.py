from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


PROMQL = {
    "cpu_usage": 'sum(rate(container_cpu_usage_seconds_total{name=~".*congestion-webapp.*"}[1m]) or rate(container_cpu_usage_seconds_total{id="/docker", cpu="total"}[1m])) * 100',
    "memory_usage": 'sum(container_memory_working_set_bytes{name=~".*congestion-webapp.*"} or container_memory_working_set_bytes{id="/docker"}) / 1024 / 1024',
    "network_in": 'sum(rate(container_network_receive_bytes_total{name=~".*congestion-webapp.*"}[1m]) or rate(container_network_receive_bytes_total{id="/"}[1m]))',
    "network_out": 'sum(rate(container_network_transmit_bytes_total{name=~".*congestion-webapp.*"}[1m]) or rate(container_network_transmit_bytes_total{id="/"}[1m]))',
    "disk_io": 'sum(rate(container_fs_reads_bytes_total{name=~".*congestion-webapp.*"}[1m]) + rate(container_fs_writes_bytes_total{name=~".*congestion-webapp.*"}[1m]) or rate(container_fs_reads_bytes_total{id="/docker"}[1m]) + rate(container_fs_writes_bytes_total{id="/docker"}[1m]))',
    "request_rate": 'sum(rate(webapp_requests_total[1m]))',
    "error_rate": 'sum(rate(webapp_requests_total{status=~"5.."}[1m])) / clamp_min(sum(rate(webapp_requests_total[1m])), 0.001) * 100',
    "response_time": 'sum(rate(webapp_request_latency_seconds_sum[1m])) / clamp_min(sum(rate(webapp_request_latency_seconds_count[1m])), 0.001) * 1000',
    "inflight_requests": "sum(webapp_inflight_requests)",
}


def parse_time(value: str | None) -> float:
    if not value:
        return datetime.now(tz=timezone.utc).timestamp()
    try:
        return float(value)
    except ValueError:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()


def query_range(base_url: str, query: str, start: float, end: float, step: str) -> list[tuple[float, float]]:
    response = requests.get(
        f"{base_url.rstrip('/')}/api/v1/query_range",
        params={"query": query, "start": start, "end": end, "step": step},
        timeout=30,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    if payload.get("status") != "success":
        raise RuntimeError(json.dumps(payload, indent=2))
    result = payload.get("data", {}).get("result", [])
    if not result:
        return []
    values = result[0].get("values", [])
    return [(float(ts), float(v) if v not in {"NaN", "+Inf", "-Inf"} else np.nan) for ts, v in values]


def collect(args: argparse.Namespace) -> pd.DataFrame:
    end = parse_time(args.end)
    start = parse_time(args.start) if args.start else end - args.minutes * 60
    series: dict[str, pd.Series] = {}
    raw_queries: dict[str, str] = {}
    for name, query in PROMQL.items():
        values = query_range(args.prometheus_url, query, start, end, args.step)
        raw_queries[name] = query
        if values:
            idx = pd.to_datetime([ts for ts, _ in values], unit="s", utc=True)
            series[name] = pd.Series([v for _, v in values], index=idx, dtype="float64")

    if not series:
        raise RuntimeError("No Prometheus samples returned. Check Docker Compose and scrape targets.")

    df = pd.DataFrame(series).sort_index()
    df.index.name = "timestamp"
    df = df.reset_index()
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    for col in ["cpu_usage", "memory_usage", "disk_io", "network_in", "network_out", "request_rate", "response_time", "error_rate"]:
        if col not in df.columns:
            df[col] = np.nan
    df["throughput"] = df[["network_in", "network_out"]].sum(axis=1, min_count=1)
    df["source_name"] = args.source_name
    df["machine_id"] = args.machine_id
    df["service_id"] = args.service_id
    df["load_profile"] = args.load_profile
    df["is_synthetic"] = 0
    df["is_noisy"] = 0
    df["time_index"] = np.arange(len(df), dtype=int)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    meta = {
        "output": str(output),
        "rows": len(df),
        "start_epoch": start,
        "end_epoch": end,
        "step": args.step,
        "queries": raw_queries,
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Prometheus/cAdvisor metrics into a training CSV.")
    parser.add_argument("--prometheus-url", default="http://localhost:9090")
    parser.add_argument("--output", default="Data/testbed/prometheus_metrics.csv")
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--step", default="5s")
    parser.add_argument("--source-name", default="docker_prometheus_testbed")
    parser.add_argument("--machine-id", default="docker_host")
    parser.add_argument("--service-id", default="congestion-webapp")
    parser.add_argument("--load-profile", default="mixed")
    args = parser.parse_args()
    df = collect(args)
    print(json.dumps({"output": args.output, "rows": len(df)}, indent=2))


if __name__ == "__main__":
    main()
