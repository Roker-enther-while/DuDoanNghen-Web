from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_noisy_logs(
    external: pd.DataFrame,
    ratio: float = 0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    n = int(len(external) * ratio)
    if n <= 0:
        return external.head(0).copy(), {"synthetic_rows": 0, "ratio": ratio}

    base = external.sort_values(["source_name", "machine_id", "timestamp", "time_index"]).tail(n).copy()
    base = base.reset_index(drop=True)
    numeric = [
        "cpu_usage",
        "memory_usage",
        "disk_io",
        "network_in",
        "network_out",
        "request_rate",
        "throughput",
        "response_time",
        "error_rate",
    ]
    config = {
        "ratio": ratio,
        "seed": seed,
        "synthetic_rows": n,
        "noise_sigma_fraction": 0.18,
        "burst_fraction": 0.08,
        "congestion_label_rule": "cpu>=85 OR memory>=90 OR response_time>=source p90 OR error_rate>=1.0",
        "warning": "Synthetic noisy data is generated for robustness testing and is not real trace data.",
    }
    for col in numeric:
        vals = pd.to_numeric(base[col], errors="coerce")
        scale = float(vals.std(skipna=True) or 1.0)
        base[col] = vals + rng.normal(0, 0.18 * scale, size=n)

    burst_idx = rng.choice(n, size=max(1, int(n * 0.08)), replace=False)
    for col, factor in {
        "cpu_usage": 1.25,
        "memory_usage": 1.18,
        "request_rate": 1.35,
        "response_time": 1.45,
        "error_rate": 2.0,
    }.items():
        base.loc[burst_idx, col] = pd.to_numeric(base.loc[burst_idx, col], errors="coerce") * factor

    base["cpu_usage"] = base["cpu_usage"].clip(lower=0, upper=100)
    base["memory_usage"] = base["memory_usage"].clip(lower=0, upper=100)
    for col in ["disk_io", "network_in", "network_out", "request_rate", "throughput", "response_time", "error_rate"]:
        base[col] = pd.to_numeric(base[col], errors="coerce").clip(lower=0)
    base["source_name"] = base["source_name"].astype(str) + "_synthetic_noisy"
    base["is_synthetic"] = 1
    base["is_noisy"] = 1
    base["time_index"] = np.arange(n, dtype=np.int64)
    rt_thr = base["response_time"].quantile(0.90)
    base["congestion_label"] = (
        (base["cpu_usage"] >= 85)
        | (base["memory_usage"] >= 90)
        | (base["response_time"] >= rt_thr)
        | (base["error_rate"] >= 1.0)
    ).astype(int)
    return base, config


def write_synthetic_config(config: dict, output_dir: str | Path) -> None:
    output = Path(output_dir)
    (output / "synthetic_generation_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = ["| setting | value |", "|---|---|"]
    for k, v in config.items():
        lines.append(f"| {k} | {v} |")
    (output / "tables" / "table_03_synthetic_noise_config.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

