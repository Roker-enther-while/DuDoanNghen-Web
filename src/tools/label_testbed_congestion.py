from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def pct(series: pd.Series, q: float, fallback: float) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return fallback
    return float(values.quantile(q))


def label_dataframe(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    for col in ["cpu_usage", "memory_usage", "request_rate", "response_time", "error_rate", "throughput"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    thresholds = {
        "cpu_usage": args.cpu_threshold,
        "memory_usage": args.memory_threshold,
        "response_time": args.latency_threshold if args.latency_threshold > 0 else pct(out["response_time"], 0.90, 500.0),
        "error_rate": args.error_threshold,
        "request_rate_high": pct(out["request_rate"], 0.85, 0.0),
    }

    request_roll = out["request_rate"].rolling(args.trend_window, min_periods=1).mean()
    latency_roll = out["response_time"].rolling(args.trend_window, min_periods=1).mean()
    request_rising = request_roll.diff().fillna(0) > 0
    latency_rising = latency_roll.diff().fillna(0) > 0

    reasons = []
    labels = []
    for pos, (_, row) in enumerate(out.iterrows()):
        row_reasons = []
        if row["cpu_usage"] >= thresholds["cpu_usage"]:
            row_reasons.append("cpu_high")
        if row["memory_usage"] >= thresholds["memory_usage"]:
            row_reasons.append("memory_high")
        if row["response_time"] >= thresholds["response_time"]:
            row_reasons.append("latency_high")
        if row["error_rate"] >= thresholds["error_rate"]:
            row_reasons.append("error_rate_high")
        if row["request_rate"] >= thresholds["request_rate_high"] and bool(request_rising.iloc[pos]) and bool(latency_rising.iloc[pos]):
            row_reasons.append("rising_request_and_latency")
        labels.append(1 if row_reasons else 0)
        reasons.append(",".join(row_reasons) if row_reasons else "normal")

    out["congestion_label"] = labels
    out["label_reason"] = reasons
    out["label_rule_version"] = "testbed_rule_v1"
    out["imputed_fields"] = ""
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    missing_before = out[numeric_cols].isna().sum().to_dict()
    out[numeric_cols] = out[numeric_cols].interpolate(method="linear", limit_direction="both").fillna(0)
    missing_after = out[numeric_cols].isna().sum().to_dict()
    report = {
        "rows": len(out),
        "positive_labels": int(out["congestion_label"].sum()),
        "positive_rate": float(out["congestion_label"].mean()) if len(out) else 0.0,
        "thresholds": thresholds,
        "missing_before": {k: int(v) for k, v in missing_before.items()},
        "missing_after": {k: int(v) for k, v in missing_after.items()},
        "rule_version": "testbed_rule_v1",
    }
    return out, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign congestion labels to testbed Prometheus CSV.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--cpu-threshold", type=float, default=85.0)
    parser.add_argument("--memory-threshold", type=float, default=450.0, help="MB for the default Docker testbed.")
    parser.add_argument("--latency-threshold", type=float, default=0.0, help="ms; <=0 uses p90.")
    parser.add_argument("--error-threshold", type=float, default=2.0, help="percent.")
    parser.add_argument("--trend-window", type=int, default=6)
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    labeled, report = label_dataframe(df, args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(output, index=False)
    report_path = Path(args.report) if args.report else output.with_suffix(".label_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "report": str(report_path), **report}, indent=2))


if __name__ == "__main__":
    main()
