"""Diagnostics for proxy targets and alert thresholds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


QUANTILES = {
    "p50": 0.50,
    "p75": 0.75,
    "p80": 0.80,
    "p85": 0.85,
    "p90": 0.90,
    "p95": 0.95,
    "p99": 0.99,
}


def _finite_float(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else 0.0


def describe_target(y) -> dict:
    values = np.asarray(y, dtype=np.float64).reshape(-1)
    quantiles = {name: _finite_float(np.quantile(values, q)) for name, q in QUANTILES.items()} if values.size else {}
    return {
        "count": int(values.size),
        "min": _finite_float(np.min(values)) if values.size else 0.0,
        "max": _finite_float(np.max(values)) if values.size else 0.0,
        "mean": _finite_float(np.mean(values)) if values.size else 0.0,
        "std": _finite_float(np.std(values)) if values.size else 0.0,
        "quantiles": quantiles,
    }


def threshold_counts(y, thresholds: dict[str, float]) -> dict:
    values = np.asarray(y, dtype=np.float64).reshape(-1)
    return {
        name: {"threshold": _finite_float(threshold), "count": int((values >= threshold).sum())}
        for name, threshold in thresholds.items()
    }


def target_distribution_report(data: dict) -> dict:
    splits = {split: np.asarray(data[f"y_{split}"], dtype=np.float64).reshape(-1) for split in ["train", "val", "test"]}
    summary = {split: describe_target(values) for split, values in splits.items()}
    threshold_values = {
        "fixed_0.50": 0.50,
        "fixed_0.60": 0.60,
        "fixed_0.70": 0.70,
        "fixed_0.80": 0.80,
    }
    for q_name in ["p80", "p85", "p90", "p95"]:
        threshold_values[f"train_{q_name}"] = summary["train"]["quantiles"][q_name]
        threshold_values[f"val_{q_name}"] = summary["val"]["quantiles"][q_name]
    counts = {split: threshold_counts(values, threshold_values) for split, values in splits.items()}
    warnings = []
    if counts["test"]["fixed_0.70"]["count"] == 0:
        warnings.append("threshold_0.70_has_no_positive_cases_in_test")
    suggested = {
        "mode": "quantile",
        "reference_split": "val",
        "value": 0.90,
        "threshold": summary["val"]["quantiles"]["p90"],
    }
    return {"splits": summary, "threshold_counts": counts, "suggested_alert_threshold": suggested, "warnings": warnings}


def autocorrelation(values, lag: int) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if lag <= 0 or arr.size <= lag:
        return 0.0
    left = arr[:-lag]
    right = arr[lag:]
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return _finite_float(np.corrcoef(left, right)[0, 1])


def proxy_target_quality(y_train, y_val=None, y_test=None) -> dict:
    values = np.asarray(y_train, dtype=np.float64).reshape(-1)
    deltas = np.abs(np.diff(values)) if values.size > 1 else np.array([], dtype=np.float64)
    rolling_window = min(60, values.size) if values.size else 0
    rolling_std_mean = 0.0
    if rolling_window > 1:
        rolling_std = [np.std(values[i : i + rolling_window]) for i in range(0, values.size - rolling_window + 1)]
        rolling_std_mean = _finite_float(np.mean(rolling_std)) if rolling_std else 0.0
    quality = {
        "autocorrelation": {f"lag_{lag}": autocorrelation(values, lag) for lag in [1, 5, 15, 30]},
        "rolling_std_mean": rolling_std_mean,
        "spike_count_delta_gt_0.05": int((deltas > 0.05).sum()),
        "delta_abs": describe_target(deltas) if deltas.size else describe_target([0.0]),
        "percent_near_constant": {
            "abs_delta_lt_0.005": _finite_float(np.mean(deltas < 0.005)) if deltas.size else 0.0,
            "abs_delta_lt_0.01": _finite_float(np.mean(deltas < 0.01)) if deltas.size else 0.0,
        },
        "notes": [],
    }
    if quality["autocorrelation"]["lag_1"] > 0.90 and quality["percent_near_constant"]["abs_delta_lt_0.01"] > 0.70:
        quality["notes"].append("proxy target is highly smooth; moving average may be naturally strong")
    if y_test is not None and int((np.asarray(y_test) >= 0.70).sum()) == 0:
        quality["notes"].append("alert threshold should be calibrated by quantile or target formula should be adjusted")
    return quality


def write_target_reports(distribution: dict, quality: dict, output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    reports_dir = output_dir / "reports"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    dist_json = metrics_dir / "target_distribution.json"
    quality_json = metrics_dir / "proxy_target_quality.json"
    dist_md = reports_dir / "target_distribution.md"
    quality_md = reports_dir / "proxy_target_quality.md"
    dist_json.write_text(json.dumps(distribution, indent=2, allow_nan=False), encoding="utf-8")
    quality_json.write_text(json.dumps(quality, indent=2, allow_nan=False), encoding="utf-8")

    lines = ["# Target Distribution", ""]
    for split, stats in distribution["splits"].items():
        q = stats["quantiles"]
        lines.extend(
            [
                f"## {split}",
                f"- min/max/mean/std: {stats['min']:.6f} / {stats['max']:.6f} / {stats['mean']:.6f} / {stats['std']:.6f}",
                f"- p80/p85/p90/p95: {q['p80']:.6f} / {q['p85']:.6f} / {q['p90']:.6f} / {q['p95']:.6f}",
                "",
            ]
        )
    lines.append("## Threshold Counts")
    for split, counts in distribution["threshold_counts"].items():
        lines.append(f"### {split}")
        for name, item in counts.items():
            lines.append(f"- {name}: threshold={item['threshold']:.6f}, count={item['count']}")
    lines.extend(["", "## Suggested Threshold", json.dumps(distribution["suggested_alert_threshold"], indent=2)])
    for warning in distribution["warnings"]:
        lines.append(f"- Warning: {warning}")
    dist_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    q_lines = [
        "# Proxy Target Quality",
        "",
        "## Autocorrelation",
    ]
    for key, value in quality["autocorrelation"].items():
        q_lines.append(f"- {key}: {value:.6f}")
    q_lines.extend(
        [
            "",
            f"- rolling_std_mean: {quality['rolling_std_mean']:.6f}",
            f"- spike_count_delta_gt_0.05: {quality['spike_count_delta_gt_0.05']}",
            f"- abs_delta_lt_0.005: {quality['percent_near_constant']['abs_delta_lt_0.005']:.6f}",
            f"- abs_delta_lt_0.01: {quality['percent_near_constant']['abs_delta_lt_0.01']:.6f}",
            "",
            "## Notes",
        ]
    )
    q_lines.extend([f"- {note}" for note in quality["notes"]] or ["- No automatic warning."])
    quality_md.write_text("\n".join(q_lines) + "\n", encoding="utf-8")
    return {
        "target_distribution_json": str(dist_json),
        "target_distribution_md": str(dist_md),
        "proxy_target_quality_json": str(quality_json),
        "proxy_target_quality_md": str(quality_md),
    }
