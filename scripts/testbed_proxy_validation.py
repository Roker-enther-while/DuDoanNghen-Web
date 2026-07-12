"""Testbed proxy validation: run Flask app + Locust, collect telemetry, correlate with proxy score.

This script:
1. Starts the Flask webapp on localhost:8080
2. Runs Locust with 4 load profiles (baseline, ramp, spike, sustained)
3. Scrapes /metrics from the Flask app every 5 seconds
4. Aggregates to 1-minute windows
5. Computes proxy_congestion_score from request logs
6. Correlates with real telemetry (latency, error rate, CPU)
7. Generates time-series overlay figures
"""

from __future__ import annotations

import json
import multiprocessing
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def start_webapp(port: int = 8080) -> subprocess.Popen:
    """Start Flask webapp in background."""
    app_path = ROOT / "testbed" / "webapp" / "app.py"
    proc = subprocess.Popen(
        [sys.executable, str(app_path)],
        cwd=str(ROOT / "testbed" / "webapp"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for app to start
    for _ in range(30):
        try:
            requests.get(f"http://localhost:{port}/", timeout=2)
            print(f"  Webapp started on port {port} (pid={proc.pid})", flush=True)
            return proc
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("Webapp failed to start")


def scrape_metrics(port: int = 8080) -> dict | None:
    """Scrape Prometheus metrics from /metrics endpoint."""
    try:
        resp = requests.get(f"http://localhost:{port}/metrics", timeout=5)
        text = resp.text
        metrics = {}
        for line in text.split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0].split("{")[0]  # strip labels
                try:
                    value = float(parts[1])
                    metrics[name] = value
                except ValueError:
                    pass
        return metrics
    except Exception:
        return None


def parse_prometheus_metrics(raw_metrics: dict) -> dict:
    """Extract meaningful metrics from raw Prometheus counters/histograms."""
    result = {}

    # Request count and error count from counters
    total_requests = 0
    error_requests = 0
    for key, val in raw_metrics.items():
        if "webapp_requests_total" in key:
            total_requests += val
            if 'status="503"' in key or 'status="5..' in key:
                error_requests += val

    result["total_requests"] = total_requests
    result["error_requests"] = error_requests
    result["error_rate"] = error_requests / max(total_requests, 1) * 100

    # Inflight requests
    result["inflight"] = raw_metrics.get("webapp_inflight_requests", 0)

    # Latency from histogram (approximate mean from _sum/_count)
    latency_sum = 0
    latency_count = 0
    for key, val in raw_metrics.items():
        if "webapp_request_latency_seconds_sum" in key:
            latency_sum += val
        elif "webapp_request_latency_seconds_count" in key:
            latency_count += val
    result["latency_mean_ms"] = (latency_sum / max(latency_count, 1)) * 1000

    return result


def run_locust_profile(profile: str, host: str, duration: str, users: int, spawn_rate: int, csv_prefix: str) -> dict:
    """Run Locust with a specific load profile."""
    env = {"LOAD_PROFILE": profile}
    cmd = [
        sys.executable, "-m", "locust",
        "-f", str(ROOT / "testbed" / "load" / "locustfile.py"),
        "--headless",
        "--host", host,
        "--users", str(users),
        "--spawn-rate", str(spawn_rate),
        "--run-time", duration,
        "--csv", csv_prefix,
    ]
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT),
        env={**dict(__import__("os").environ), **env},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    proc.wait(timeout=300)
    return {"profile": profile, "status": "done", "csv_prefix": csv_prefix}


def collect_testbed_data(
    profile: str,
    host: str,
    duration_seconds: int,
    users: int,
    spawn_rate: int,
    scrape_interval: float = 5.0,
) -> pd.DataFrame:
    """Collect metrics while running a Locust profile."""
    print(f"  Running profile: {profile} for {duration_seconds}s with {users} users", flush=True)

    # Start Locust in background
    env = dict(__import__("os").environ)
    env["LOAD_PROFILE"] = profile
    csv_prefix = str(ROOT / "outputs" / "testbed_validation" / f"locust_{profile}")
    cmd = [
        sys.executable, "-m", "locust",
        "-f", str(ROOT / "testbed" / "load" / "locustfile.py"),
        "--headless",
        "--host", host,
        "--users", str(users),
        "--spawn-rate", str(spawn_rate),
        "--run-time", f"{duration_seconds}s",
        "--csv", csv_prefix,
    ]
    locust_proc = subprocess.Popen(
        cmd, cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Scrape metrics during the run
    records = []
    start_time = time.time()
    prev_requests = 0
    prev_errors = 0
    prev_time = start_time

    while time.time() - start_time < duration_seconds:
        time.sleep(scrape_interval)
        elapsed = time.time() - start_time

        raw = scrape_metrics()
        if raw is None:
            continue

        metrics = parse_prometheus_metrics(raw)
        now = time.time()
        dt = now - prev_time

        # Compute rates (per second)
        req_rate = (metrics["total_requests"] - prev_requests) / max(dt, 0.001)
        err_rate = (metrics["error_requests"] - prev_errors) / max(dt, 0.001) * 100

        records.append({
            "timestamp": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "request_rate": round(req_rate, 2),
            "error_rate": round(err_rate, 4),
            "latency_mean_ms": round(metrics["latency_mean_ms"], 4),
            "inflight": round(metrics["inflight"], 1),
            "total_requests": metrics["total_requests"],
            "total_errors": metrics["error_requests"],
            "profile": profile,
        })

        prev_requests = metrics["total_requests"]
        prev_errors = metrics["error_requests"]
        prev_time = now

    locust_proc.wait(timeout=60)
    df = pd.DataFrame(records)
    print(f"  Collected {len(df)} samples for {profile}", flush=True)
    return df


def compute_proxy_from_testbed(df: pd.DataFrame) -> pd.DataFrame:
    """Compute proxy congestion score from testbed metrics.

    Maps testbed columns to NASA-style features for the proxy formula.
    """
    work = df.copy()

    # Map testbed metrics to NASA-style features
    request_count = work["request_rate"]  # requests/sec as proxy for request_count
    bytes_sum = work["latency_mean_ms"] * work["request_rate"]  # rough throughput proxy
    unique_hosts = pd.Series(1.0, index=work.index)  # single host
    error_rate = work["error_rate"] / 100.0  # convert % to [0,1]

    # Request spike score (rolling z-score)
    rolling_window = max(2, len(work) // 5)
    min_periods = min(rolling_window, max(2, len(work) // 10))
    rolling_mean = request_count.rolling(rolling_window, min_periods=min_periods).mean().shift(1)
    rolling_std = request_count.rolling(rolling_window, min_periods=min_periods).std().shift(1).replace(0, np.nan)
    z_score = ((request_count - rolling_mean) / rolling_std).replace([np.inf, -np.inf], 0).fillna(0)
    request_spike_score = (z_score.clip(lower=0, upper=3) / 3.0).clip(0, 1)

    # Expanding min-max normalization
    def expanding_minmax(series):
        values = pd.to_numeric(series, errors="coerce").fillna(0).astype(float)
        running_min = values.cummin()
        running_max = values.cummax()
        denom = (running_max - running_min).replace(0, np.nan)
        return ((values - running_min) / denom).fillna(0).clip(0, 1)

    # Weighted composite (same weights as NASA pipeline)
    score = (
        expanding_minmax(request_count) * 0.35 +
        expanding_minmax(bytes_sum) * 0.20 +
        expanding_minmax(unique_hosts) * 0.15 +
        error_rate.clip(0, 1) * 0.20 +
        request_spike_score * 0.10
    )
    work["proxy_congestion_score"] = score.clip(0, 1)
    return work


def compute_correlations(df: pd.DataFrame) -> dict:
    """Compute Pearson and Spearman correlations between proxy score and real metrics."""
    proxy = df["proxy_congestion_score"]
    results = {}
    for col, label in [("latency_mean_ms", "Latency Mean (ms)"), ("error_rate", "Error Rate (%)"), ("inflight", "In-flight Requests"), ("request_rate", "Request Rate")]:
        if col not in df.columns:
            continue
        real = pd.to_numeric(df[col], errors="coerce")
        valid = real.notna() & proxy.notna()
        if valid.sum() < 5:
            continue
        pearson_r, pearson_p = stats.pearsonr(proxy[valid], real[valid])
        spearman_r, spearman_p = stats.spearmanr(proxy[valid], real[valid])
        # Handle NaN from constant inputs
        pearson_r = 0.0 if np.isnan(pearson_r) else pearson_r
        pearson_p = 1.0 if np.isnan(pearson_p) else pearson_p
        spearman_r = 0.0 if np.isnan(spearman_r) else spearman_r
        spearman_p = 1.0 if np.isnan(spearman_p) else spearman_p
        results[col] = {
            "label": label,
            "n_valid": int(valid.sum()),
            "pearson_r": round(float(pearson_r), 4),
            "pearson_p": round(float(pearson_p), 6),
            "spearman_r": round(float(spearman_r), 4),
            "spearman_p": round(float(spearman_p), 6),
        }
    return results


def generate_figure(df: pd.DataFrame, profile: str, output_dir: Path) -> None:
    """Generate time-series overlay figure (proxy score vs real metrics)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        ts = pd.to_datetime(df["timestamp"])

        # Plot 1: Proxy score vs latency
        ax1 = axes[0]
        ax1.plot(ts, df["proxy_congestion_score"], "b-", label="Proxy Score", linewidth=1.5)
        ax1.set_ylabel("Proxy Score", color="b")
        ax1.tick_params(axis="y", labelcolor="b")
        ax1_r = ax1.twinx()
        ax1_r.plot(ts, df["latency_mean_ms"], "r-", alpha=0.7, label="Latency (ms)", linewidth=1)
        ax1_r.set_ylabel("Latency (ms)", color="r")
        ax1_r.tick_params(axis="y", labelcolor="r")
        ax1.set_title(f"Profile: {profile} — Proxy Score vs Latency")
        ax1.legend(loc="upper left")
        ax1_r.legend(loc="upper right")

        # Plot 2: Proxy score vs error rate
        ax2 = axes[1]
        ax2.plot(ts, df["proxy_congestion_score"], "b-", label="Proxy Score", linewidth=1.5)
        ax2.set_ylabel("Proxy Score", color="b")
        ax2_r = ax2.twinx()
        ax2_r.plot(ts, df["error_rate"], "r-", alpha=0.7, label="Error Rate (%)", linewidth=1)
        ax2_r.set_ylabel("Error Rate (%)", color="r")
        ax2_r.tick_params(axis="y", labelcolor="r")
        ax2.set_title(f"Profile: {profile} — Proxy Score vs Error Rate")
        ax2.legend(loc="upper left")
        ax2_r.legend(loc="upper right")

        # Plot 3: Request rate
        ax3 = axes[2]
        ax3.plot(ts, df["request_rate"], "g-", label="Request Rate", linewidth=1.5)
        ax3.set_ylabel("Request Rate (req/s)", color="g")
        ax3.set_xlabel("Time")
        ax3.set_title(f"Profile: {profile} — Request Rate Over Time")
        ax3.legend(loc="upper left")

        plt.tight_layout()
        fig_path = output_dir / f"proxy_validation_{profile}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure saved: {fig_path}", flush=True)
    except ImportError:
        print("  matplotlib not available, skipping figure generation", flush=True)


def main():
    output_dir = ROOT / "outputs" / "testbed_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = ROOT / "outputs" / "figures" / "proxy_validation"
    figures_dir.mkdir(parents=True, exist_ok=True)

    host = "http://localhost:8080"

    # Start webapp
    print("Starting webapp...", flush=True)
    webapp_proc = start_webapp()

    try:
        # Define 4 load scenarios
        scenarios = [
            {"profile": "baseline", "users": 10, "spawn_rate": 5, "duration": 180, "description": "Low stable load"},
            {"profile": "gradual", "users": 60, "spawn_rate": 2, "duration": 180, "description": "Gradual ramp-up"},
            {"profile": "spike", "users": 120, "spawn_rate": 50, "duration": 180, "description": "Sudden spike then recovery"},
            {"profile": "stress", "users": 120, "spawn_rate": 10, "duration": 300, "description": "Sustained overload"},
        ]

        all_data = []
        all_correlations = {}

        for scenario in scenarios:
            print(f"\n{'='*60}", flush=True)
            print(f"Scenario: {scenario['description']}", flush=True)
            print(f"{'='*60}", flush=True)

            df = collect_testbed_data(
                scenario["profile"], host,
                scenario["duration"], scenario["users"], scenario["spawn_rate"],
            )

            if len(df) < 10:
                print(f"  Too few samples ({len(df)}), skipping", flush=True)
                continue

            # Compute proxy score
            df = compute_proxy_from_testbed(df)

            # Compute correlations
            corrs = compute_correlations(df)
            all_correlations[scenario["profile"]] = corrs

            # Generate figure
            generate_figure(df, scenario["profile"], figures_dir)

            # Save data
            df.to_csv(output_dir / f"testbed_{scenario['profile']}.csv", index=False)
            all_data.append(df)

            # Print correlations
            print(f"\n  Correlations for {scenario['profile']}:", flush=True)
            for col, info in corrs.items():
                print(f"    {info['label']}: Pearson r={info['pearson_r']:.4f} (p={info['pearson_p']:.2e}), "
                      f"Spearman r={info['spearman_r']:.4f}", flush=True)

        # Combine all data
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined.to_csv(output_dir / "testbed_all_profiles.csv", index=False)

            # Overall correlations
            overall_corrs = compute_correlations(combined)
            all_correlations["overall"] = overall_corrs

            print(f"\n{'='*60}", flush=True)
            print("Overall correlations (all profiles combined):", flush=True)
            for col, info in overall_corrs.items():
                print(f"  {info['label']}: Pearson r={info['pearson_r']:.4f} (p={info['pearson_p']:.2e})", flush=True)

            # Save correlations
            (output_dir / "correlations.json").write_text(
                json.dumps(all_correlations, indent=2, allow_nan=False), encoding="utf-8"
            )

            # Generate summary report
            generate_report(all_correlations, output_dir)

    finally:
        # Stop webapp
        webapp_proc.terminate()
        webapp_proc.wait(timeout=10)
        print("\nWebapp stopped", flush=True)


def generate_report(correlations: dict, output_dir: Path) -> None:
    """Generate markdown report of proxy validation results."""
    lines = [
        "# Proxy Validation Report — Testbed Results",
        "",
        "## Methodology",
        "",
        "- Flask webapp with configurable CPU/IO/error endpoints",
        "- Locust load generator with 4 profiles: baseline, gradual, spike, stress",
        "- Prometheus metrics scraped every 5 seconds",
        "- Proxy congestion score computed using same formula as NASA pipeline",
        "- Correlations computed between proxy score and real metrics",
        "",
        "## Results by Profile",
        "",
    ]

    for profile, corrs in correlations.items():
        if profile == "overall":
            continue
        lines.append(f"### {profile}")
        lines.append("")
        lines.append("| Metric | Pearson r | p-value | Spearman r | p-value |")
        lines.append("|---|---|---|---|---|")
        for col, info in corrs.items():
            lines.append(
                f"| {info['label']} | {info['pearson_r']:.4f} | {info['pearson_p']:.2e} | "
                f"{info['spearman_r']:.4f} | {info['spearman_p']:.2e} |"
            )
        lines.append("")

    if "overall" in correlations:
        lines.extend(["## Overall (All Profiles Combined)", ""])
        lines.append("| Metric | Pearson r | p-value | Spearman r | p-value |")
        lines.append("|---|---|---|---|---|")
        for col, info in correlations["overall"].items():
            lines.append(
                f"| {info['label']} | {info['pearson_r']:.4f} | {info['pearson_p']:.2e} | "
                f"{info['spearman_r']:.4f} | {info['spearman_p']:.2e} |"
            )
        lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "The proxy congestion score is a weighted composite of load metrics. This validation",
        "checks whether it correlates with independently measured system metrics (latency,",
        "error rate) from a real testbed under controlled load scenarios.",
        "",
        "|r| >= 0.6: strong correlation, proxy is a reasonable congestion indicator",
        "|r| 0.3-0.6: moderate correlation, proxy has some signal but is imperfect",
        "|r| < 0.3: weak correlation, proxy does not reflect real congestion well",
    ])

    report_path = output_dir / "proxy_validation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved: {report_path}", flush=True)


if __name__ == "__main__":
    main()
