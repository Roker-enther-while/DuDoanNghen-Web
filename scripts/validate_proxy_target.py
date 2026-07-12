"""Validate proxy_congestion_score against real testbed telemetry.

This script computes the proxy congestion score from testbed telemetry features
and correlates it with measured system metrics (latency, error rate, CPU, etc.)
to assess whether the proxy target reflects real congestion signals.

Usage:
    python scripts/validate_proxy_target.py --testbed-csv data/testbed/longrun_20260517_211328/testbed_labeled.csv
    python scripts/validate_proxy_target.py --testbed-csv data/testbed/testbed_labeled.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --- Proxy computation (mirrors src/data/nasa_http.py) ---

def _expanding_minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0).astype(float)
    running_min = values.cummin()
    running_max = values.cummax()
    denom = (running_max - running_min).replace(0, np.nan)
    return ((values - running_min) / denom).fillna(0).clip(0, 1)


def compute_proxy_score(df: pd.DataFrame, rolling_window: int = 20) -> pd.DataFrame:
    """Compute proxy congestion score on testbed data.

    The testbed data has different column names than NASA data, so we map:
    - request_count -> request_rate (requests/sec, but we treat as count proxy)
    - bytes_sum -> throughput (bytes/sec proxy)
    - unique_hosts -> 1 (testbed has single host)
    - error_rate -> error_rate (already a percentage)
    """
    work = df.copy()
    ts = pd.to_datetime(work["timestamp"], utc=True)
    work = work.sort_values("timestamp").reset_index(drop=True)

    # Map testbed columns to NASA-style features
    request_count = pd.to_numeric(work["request_rate"], errors="coerce").fillna(0)
    bytes_sum = pd.to_numeric(work["throughput"], errors="coerce").fillna(0)

    # For testbed: unique_hosts is always 1 (single service)
    unique_hosts = pd.Series(1.0, index=work.index)

    # Error rate: testbed gives it as percentage, NASA pipeline computes it
    error_rate = pd.to_numeric(work.get("error_rate", 0), errors="coerce").fillna(0) / 100.0
    error_rate = error_rate.clip(0, 1)

    # Request spike score
    min_periods = min(rolling_window, max(2, len(request_count) // 3))
    rolling_mean = request_count.rolling(rolling_window, min_periods=min_periods).mean().shift(1)
    rolling_std = request_count.rolling(rolling_window, min_periods=min_periods).std().shift(1).replace(0, np.nan)
    z_score = ((request_count - rolling_mean) / rolling_std).replace([np.inf, -np.inf], 0).fillna(0)
    request_spike_score = (z_score.clip(lower=0, upper=3) / 3.0).clip(0, 1)

    # Weighted composite
    weights = {
        "request_count": 0.35,
        "bytes_sum": 0.20,
        "unique_hosts": 0.15,
        "error_rate": 0.20,
        "request_spike_score": 0.10,
    }
    components = {
        "request_count": _expanding_minmax(request_count),
        "bytes_sum": _expanding_minmax(bytes_sum),
        "unique_hosts": _expanding_minmax(unique_hosts),
        "error_rate": error_rate,
        "request_spike_score": request_spike_score,
    }
    score = sum(components[name] * weight for name, weight in weights.items())
    work["proxy_congestion_score"] = score.clip(0, 1)

    # Also store individual components for analysis
    for name, comp in components.items():
        work[f"component_{name}"] = comp
    work["component_request_spike_score"] = request_spike_score

    return work


def compute_correlations(df: pd.DataFrame) -> dict:
    """Compute Pearson and Spearman correlations between proxy score and real telemetry."""
    proxy = df["proxy_congestion_score"]

    # Real telemetry columns from testbed
    telemetry_cols = {
        "response_time": "Response Time (ms)",
        "error_rate": "Error Rate (%)",
        "cpu_usage": "CPU Usage (%)",
        "memory_usage": "Memory Usage (MB)",
        "request_rate": "Request Rate (req/s)",
        "inflight_requests": "In-flight Requests",
        "throughput": "Throughput (bytes/s)",
    }

    results = {}
    for col, label in telemetry_cols.items():
        if col not in df.columns:
            continue
        real = pd.to_numeric(df[col], errors="coerce")
        valid = real.notna() & proxy.notna()
        if valid.sum() < 5:
            continue

        pearson_r, pearson_p = stats.pearsonr(proxy[valid], real[valid])
        spearman_r, spearman_p = stats.spearmanr(proxy[valid], real[valid])

        results[col] = {
            "label": label,
            "n_valid": int(valid.sum()),
            "pearson_r": round(float(pearson_r), 4),
            "pearson_p": round(float(pearson_p), 6),
            "spearman_r": round(float(spearman_r), 4),
            "spearman_p": round(float(spearman_p), 6),
            "mean_real": round(float(real[valid].mean()), 4),
            "std_real": round(float(real[valid].std()), 4),
            "mean_proxy": round(float(proxy[valid].mean()), 4),
        }
    return results


def compute_lag_analysis(df: pd.DataFrame, max_lag: int = 10) -> dict:
    """Compute cross-correlation at different lags to find optimal alignment."""
    proxy = df["proxy_congestion_score"].values
    results = {}

    for col in ["response_time", "error_rate", "cpu_usage"]:
        if col not in df.columns:
            continue
        real = pd.to_numeric(df[col], errors="coerce").fillna(0).values

        lags = {}
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                p = proxy[lag:]
                r = real[:len(p)] if len(real) >= len(p) else real
            else:
                r = real[-lag:]
                p = proxy[:len(r)]
            min_len = min(len(p), len(r))
            if min_len < 5:
                continue
            corr, _ = stats.pearsonr(p[:min_len], r[:min_len])
            lags[lag] = round(float(corr), 4)

        if lags:
            best_lag = max(lags, key=lags.get)
            results[col] = {
                "correlations_by_lag": lags,
                "best_lag": best_lag,
                "best_correlation": lags[best_lag],
            }
    return results


def compute_quantile_analysis(df: pd.DataFrame) -> dict:
    """Analyze proxy score distribution vs real metrics at different quantile thresholds."""
    proxy = df["proxy_congestion_score"]

    results = {}
    for col in ["response_time", "error_rate", "cpu_usage"]:
        if col not in df.columns:
            continue
        real = pd.to_numeric(df[col], errors="coerce")

        thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
        threshold_results = []
        for t in thresholds:
            proxy_threshold = float(proxy.quantile(t))
            high_proxy = proxy >= proxy_threshold
            if high_proxy.sum() == 0:
                continue
            mean_real_when_high = float(real[high_proxy].mean())
            mean_real_when_low = float(real[~high_proxy].mean())
            ratio = mean_real_when_high / max(mean_real_when_low, 1e-10)
            threshold_results.append({
                "quantile": t,
                "proxy_threshold": round(proxy_threshold, 4),
                "n_high": int(high_proxy.sum()),
                "mean_real_high_proxy": round(mean_real_when_high, 4),
                "mean_real_low_proxy": round(mean_real_when_low, 4),
                "ratio": round(ratio, 4),
            })
        results[col] = threshold_results
    return results


def generate_scatter_data(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate CSV data suitable for scatter/time-series visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_df = df[["timestamp", "proxy_congestion_score"]].copy()
    for col in ["response_time", "error_rate", "cpu_usage", "request_rate"]:
        if col in df.columns:
            viz_df[col] = pd.to_numeric(df[col], errors="coerce")
    viz_df.to_csv(output_dir / "proxy_vs_real_telemetry.csv", index=False)


def run_validation(testbed_csv: str, output_dir: str = "outputs/proxy_validation") -> dict:
    """Main validation pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load testbed data
    df = pd.read_csv(testbed_csv)
    print(f"Loaded {len(df)} rows from {testbed_csv}")
    print(f"Columns: {list(df.columns)}")

    # Compute proxy score
    df = compute_proxy_score(df)
    print(f"Proxy score range: [{df['proxy_congestion_score'].min():.4f}, {df['proxy_congestion_score'].max():.4f}]")
    print(f"Proxy score mean: {df['proxy_congestion_score'].mean():.4f}")

    # Correlations
    correlations = compute_correlations(df)
    print("\n=== Correlation Analysis ===")
    for col, info in correlations.items():
        print(f"  {info['label']}: Pearson r={info['pearson_r']:.4f} (p={info['pearson_p']:.2e}), "
              f"Spearman r={info['spearman_r']:.4f} (p={info['spearman_p']:.2e})")

    # Lag analysis
    lag_analysis = compute_lag_analysis(df)
    print("\n=== Lag Analysis ===")
    for col, info in lag_analysis.items():
        print(f"  {col}: best lag={info['best_lag']}, r={info['best_correlation']:.4f}")

    # Quantile analysis
    quantile_analysis = compute_quantile_analysis(df)

    # Generate visualization data
    generate_scatter_data(df, output_dir)

    # Summary assessment
    # For a congestion proxy, we expect POSITIVE correlation with congestion indicators
    # (response_time, error_rate, cpu_usage). Negative correlations mean the proxy
    # is inversely related to actual congestion — it measures load level, not congestion.
    proxy_validated = False
    proxy_direction_correct = False
    validation_notes = []

    congestion_indicators = ["response_time", "error_rate", "cpu_usage"]
    load_indicators = ["request_rate", "throughput"]

    positive_congestion = 0
    negative_congestion = 0

    for metric in congestion_indicators:
        if metric in correlations:
            r = correlations[metric]["pearson_r"]
            abs_r = abs(r)
            direction = "POSITIVE" if r > 0 else "NEGATIVE"
            strength = "strong" if abs_r >= 0.5 else "moderate" if abs_r >= 0.3 else "weak"
            validation_notes.append(f"{correlations[metric]['label']}: r={r:.4f} ({direction}, {strength})")
            if abs_r >= 0.3:
                if r > 0:
                    positive_congestion += 1
                else:
                    negative_congestion += 1

    for metric in load_indicators:
        if metric in correlations:
            r = correlations[metric]["pearson_r"]
            direction = "POSITIVE" if r > 0 else "NEGATIVE"
            validation_notes.append(f"{correlations[metric]['label']}: r={r:.4f} ({direction})")

    if negative_congestion > 0 and positive_congestion == 0:
        proxy_validated = False
        proxy_direction_correct = False
        validation_notes.insert(0,
            "CRITICAL FINDING: Proxy score is NEGATIVELY correlated with congestion indicators "
            "(response_time, error_rate). HIGH proxy score corresponds to LOW congestion. "
            "The proxy score measures LOAD LEVEL, not CONGESTION LEVEL. "
            "This is a significant limitation that must be reported in the research."
        )
        validation_notes.append(
            "RECOMMENDATION: Revise the proxy formula to include latency/error components directly, "
            "or reinterpret the score as a 'load intensity' metric rather than 'congestion'."
        )
    elif positive_congestion >= 1:
        proxy_validated = True
        proxy_direction_correct = True
        validation_notes.insert(0, "Proxy score shows expected positive correlation with congestion indicators.")
    else:
        validation_notes.insert(0, "Proxy score shows weak or no correlation with congestion indicators.")

    # Build report
    report = {
        "status": "success",
        "testbed_csv": testbed_csv,
        "n_rows": len(df),
        "proxy_score_range": [float(df["proxy_congestion_score"].min()), float(df["proxy_congestion_score"].max())],
        "proxy_score_mean": float(df["proxy_congestion_score"].mean()),
        "proxy_score_std": float(df["proxy_congestion_score"].std()),
        "correlations": correlations,
        "lag_analysis": lag_analysis,
        "quantile_analysis": quantile_analysis,
        "proxy_validated": proxy_validated,
        "validation_notes": validation_notes,
        "output_paths": {
            "scatter_data": str(output_dir / "proxy_vs_real_telemetry.csv"),
            "full_report": str(output_dir / "proxy_validation_report.json"),
        },
    }

    # Save full report
    (output_dir / "proxy_validation_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )

    # Generate markdown report
    md_lines = [
        "# Proxy Congestion Score Validation Report",
        "",
        f"**Testbed data**: `{testbed_csv}` ({len(df)} rows)",
        f"**Proxy score range**: [{df['proxy_congestion_score'].min():.4f}, {df['proxy_congestion_score'].max():.4f}]",
        f"**Proxy score mean**: {df['proxy_congestion_score'].mean():.4f} (std={df['proxy_congestion_score'].std():.4f})",
        "",
        "## Validation Status",
        "",
        f"**{'VALIDATED' if proxy_validated else 'NOT VALIDATED'}**",
        "",
    ]
    for note in validation_notes:
        md_lines.append(f"- {note}")
    md_lines.extend(["", "## Correlation with Real Telemetry", ""])
    md_lines.append("| Metric | Pearson r | p-value | Spearman r | p-value | n |")
    md_lines.append("|---|---|---|---|---|---|")
    for col, info in correlations.items():
        md_lines.append(
            f"| {info['label']} | {info['pearson_r']:.4f} | {info['pearson_p']:.2e} | "
            f"{info['spearman_r']:.4f} | {info['spearman_p']:.2e} | {info['n_valid']} |"
        )

    if lag_analysis:
        md_lines.extend(["", "## Lag Analysis", ""])
        md_lines.append("| Metric | Best Lag (steps) | Correlation at Best Lag |")
        md_lines.append("|---|---|---|")
        for col, info in lag_analysis.items():
            md_lines.append(f"| {col} | {info['best_lag']} | {info['best_correlation']:.4f} |")

    if quantile_analysis:
        md_lines.extend(["", "## Quantile Analysis", ""])
        for col, thresholds in quantile_analysis.items():
            md_lines.append(f"### {col}")
            md_lines.append("| Quantile | Proxy Threshold | Mean Real (High Proxy) | Mean Real (Low Proxy) | Ratio |")
            md_lines.append("|---|---|---|---|---|")
            for t in thresholds:
                md_lines.append(
                    f"| {t['quantile']:.0%} | {t['proxy_threshold']:.4f} | "
                    f"{t['mean_real_high_proxy']:.4f} | {t['mean_real_low_proxy']:.4f} | {t['ratio']:.2f}x |"
                )
            md_lines.append("")

    md_lines.extend([
        "",
        "## Interpretation",
        "",
        "The proxy congestion score is a synthetic composite of HTTP log features. This validation",
        "checks whether it correlates with independently measured system metrics (latency, error rate,",
        "CPU usage) from a real testbed. If correlations are weak, the proxy score may not reflect",
        "actual system congestion and should be interpreted with caution.",
        "",
        "### Component Weights",
        "",
        "| Component | Weight |",
        "|---|---|",
        "| request_count | 0.35 |",
        "| bytes_sum | 0.20 |",
        "| unique_hosts | 0.15 |",
        "| error_rate | 0.20 |",
        "| request_spike_score | 0.10 |",
        "",
        "See `docs/PROXY_TARGET_DEFINITION.md` for the full formula.",
    ])

    (output_dir / "proxy_validation_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nReport saved to: {output_dir / 'proxy_validation_report.md'}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate proxy congestion score against testbed telemetry")
    parser.add_argument("--testbed-csv", required=True, help="Path to testbed labeled CSV")
    parser.add_argument("--output-dir", default="outputs/proxy_validation", help="Output directory")
    args = parser.parse_args(argv)

    report = run_validation(args.testbed_csv, args.output_dir)
    print(json.dumps({"status": report["status"], "proxy_validated": report["proxy_validated"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
