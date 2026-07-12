from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.sql_data_pool import COMMON_SCHEMA


ALIASES = {
    "timestamp": ["timestamp", "time", "datetime", "start_time", "end_time"],
    "cpu_usage": ["cpu_usage", "cpu", "cpu_util", "cpu_utilization", "avg_cpu"],
    "memory_usage": ["memory_usage", "memory", "mem", "mem_usage", "memory_utilization"],
    "disk_io": ["disk_io", "disk", "io", "disk_usage"],
    "network_in": ["network_in", "net_in", "rx", "bytes_in"],
    "network_out": ["network_out", "net_out", "tx", "bytes_out"],
    "request_rate": ["request_rate", "requests", "qps", "rps"],
    "throughput": ["throughput", "network_io", "network", "net_io"],
    "response_time": ["response_time", "latency", "rt", "duration"],
    "error_rate": ["error_rate", "error_rate_5xx", "errors", "failure_rate"],
    "machine_id": ["machine_id", "machine", "vm_id", "host_id", "container_id"],
    "service_id": ["service_id", "service", "job_id", "app_id", "task_id"],
}


def _find_column(df: pd.DataFrame, names: list[str]) -> str | None:
    by_lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in by_lower:
            return by_lower[name.lower()]
    return None


def infer_congestion_label(df: pd.DataFrame) -> pd.Series:
    cpu = pd.to_numeric(df.get("cpu_usage", 0), errors="coerce").fillna(0)
    mem = pd.to_numeric(df.get("memory_usage", 0), errors="coerce").fillna(0)
    rt = pd.to_numeric(df.get("response_time", 0), errors="coerce").fillna(0)
    err = pd.to_numeric(df.get("error_rate", 0), errors="coerce").fillna(0)
    rt_thr = rt.quantile(0.90) if rt.notna().any() else np.inf
    return ((cpu >= 85) | (mem >= 90) | (rt >= rt_thr) | (err >= 1.0)).astype(int)


def harmonize_dataframe(
    df: pd.DataFrame,
    source_name: str,
    default_start: str = "2026-01-01 00:00:00",
    mapping_out: dict | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    mapping: dict[str, str] = {}
    for target, aliases in ALIASES.items():
        col = _find_column(df, aliases)
        if col is not None:
            out[target] = df[col]
            mapping[target] = col
        else:
            out[target] = np.nan
            mapping[target] = "<missing>"

    if out["timestamp"].isna().all():
        out["timestamp"] = pd.date_range(default_start, periods=len(out), freq="10min")
        mapping["timestamp"] = "<generated_time_index_10min>"
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["timestamp"] = out["timestamp"].ffill().bfill()
    if out["timestamp"].isna().all():
        out["timestamp"] = pd.date_range(default_start, periods=len(out), freq="10min")

    for col in [
        "cpu_usage",
        "memory_usage",
        "disk_io",
        "network_in",
        "network_out",
        "request_rate",
        "throughput",
        "response_time",
        "error_rate",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if out["throughput"].isna().all() and "network_io" in {c.lower() for c in df.columns}:
        pass
    if out["network_in"].isna().all() and out["throughput"].notna().any():
        out["network_in"] = out["throughput"] * 0.5
        mapping["network_in"] = "proxy: throughput * 0.5"
    if out["network_out"].isna().all() and out["throughput"].notna().any():
        out["network_out"] = out["throughput"] * 0.5
        mapping["network_out"] = "proxy: throughput * 0.5"
    if out["throughput"].isna().all() and (out["network_in"].notna().any() or out["network_out"].notna().any()):
        out["throughput"] = out[["network_in", "network_out"]].sum(axis=1, min_count=1)
        mapping["throughput"] = "proxy: network_in + network_out"

    out["source_name"] = source_name
    out["machine_id"] = out["machine_id"].fillna(source_name + "_machine_0").astype(str)
    out["service_id"] = out["service_id"].fillna(source_name + "_service_0").astype(str)
    out["is_synthetic"] = 0
    out["is_noisy"] = 0
    out["time_index"] = np.arange(len(out), dtype=np.int64)
    out["congestion_label"] = infer_congestion_label(out)
    out = out[COMMON_SCHEMA].sort_values(["timestamp", "time_index"]).reset_index(drop=True)

    if mapping_out is not None:
        mapping_out[source_name] = mapping
    return out


def write_schema_mapping(mapping: dict, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "schema_mapping.json").write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["| source | target_column | original_column_or_rule |", "|---|---|---|"]
    for source, items in mapping.items():
        for target, original in items.items():
            lines.append(f"| {source} | {target} | {original} |")
    (output / "tables" / "table_02_schema_mapping.md").parent.mkdir(parents=True, exist_ok=True)
    (output / "tables" / "table_02_schema_mapping.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

