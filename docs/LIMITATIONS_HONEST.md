# Limitations — Honest Assessment (Updated 2026-07-11)

This section provides a transparent account of the study's limitations, informed by the research integrity audit conducted as part of the upgrade process.

## 1. Data Limitations

### 1.1 NASA HTTP 1995 is a single legacy dataset
- **What**: The NASA Kennedy Space Center HTTP access log from July-August 1995 is the sole real public data source used for training and evaluation.
- **Why it matters**: This log represents HTTP/1.0 traffic to a single web server during the dial-up era. Modern web infrastructure uses HTTP/2, HTTP/3, CDN, microservices, adaptive TCP congestion control, and load balancers — none of which are present in this dataset. The workload patterns, error distributions, and traffic characteristics of 1995 are fundamentally different from today's networks.
- **Scope of inference**: Results should be interpreted as demonstrating a **methodological approach** (feature engineering + time-series prediction architecture) on a historical dataset, not as evidence of congestion prediction capability on modern networks.

### 1.2 No measured congestion ground truth
- **What**: The NASA dataset contains only HTTP access log fields (host, timestamp, request, status, bytes). It does not include CPU usage, memory usage, response time, queue depth, packet loss, or any other direct measurement of system or network congestion.
- **Why it matters**: Without ground truth congestion labels, we cannot verify that the model is actually predicting congestion rather than learning surface-level traffic patterns.

### 1.3 Proxy congestion score is a synthetic composite
- **What**: The `proxy_congestion_score` is a weighted combination of five HTTP log features (request_count 0.35, bytes_sum 0.20, unique_hosts 0.15, error_rate 0.20, request_spike_score 0.10), normalized with expanding min-max.
- **Critical finding from validation**: When applied to real testbed telemetry (1802 samples from a Flask+Prometheus+Locust system), the proxy score showed **negative correlation** with actual congestion indicators:
  - Response Time: r = -0.54 (high proxy → low latency)
  - Error Rate: r = -0.54 (high proxy → few errors)
  - CPU Usage: r = -0.30 (weak inverse relationship)
- **Interpretation**: The proxy score appears to measure **load intensity** (how much traffic is flowing) rather than **congestion** (how much the system is struggling). During steady-state high traffic, the system handles load efficiently (low latency, few errors). Congestion occurs during overload transitions, which the proxy score does not capture well.
- **See**: `docs/PROXY_VALIDATION_REPORT.md` for full correlation analysis.

### 1.4 Single-source only
- **What**: The Zanbil dataset was planned but raw data is unavailable. All results are NASA-only.
- **Why it matters**: No cross-source validation or generalization claims can be made.

## 2. Model Limitations

### 2.1 R² = 0.33 means 67% of variance is unexplained
- The model explains approximately one-third of the variance in the proxy target. This is a moderate result for time-series prediction, but it means the model's predictions have substantial residual error.

### 2.2 Alert performance depends heavily on threshold
- At the original p90 validation threshold (0.184), recall is extremely low (0.007), meaning the model misses 99.3% of "congestion" events.
- Calibrated threshold (0.05) achieves F1=0.87 but classifies 77% of samples as positive — this is not a practical alerting system.
- The trade-off between precision and recall at different thresholds should be presented as a PR curve, not a single number.

### 2.3 No ablation study was previously conducted
- Prior to this upgrade, only cross-architecture comparisons existed (TCN-Attention-BiLSTM vs. LSTM, GRU, etc.). There was no evidence that each component (TCN, Attention, BiLSTM) actually contributes to performance. The ablation study infrastructure is now implemented but requires training runs to produce results.

### 2.4 No rolling-origin validation
- Previous results used a single chronological split. Rolling-origin cross-validation is now implemented to provide more robust estimates with variance reporting, but requires training runs.

### 2.5 Model checkpoints and prediction artifacts are not reproducible from disk
- **Critical audit finding**: The `data/processed/` directory (containing the windows FP16 NPZ), `outputs/models/` (model checkpoints), and `outputs/predictions/` (prediction CSVs) do not exist on disk. The metric JSON files reference artifacts that are no longer present. This means the reported numbers cannot be independently verified by recomputing from predictions.
- **Mitigation**: All metrics in JSON files are treated as historical records. The training infrastructure is intact and can reproduce results when data is re-downloaded and training is re-run.

## 3. Scope of Practical Applicability

### What this study IS:
- A methodological demonstration of hybrid deep learning architectures (TCN + Self-Attention + BiLSTM) for time-series prediction on a historical HTTP workload dataset.
- An exploration of feature engineering and proxy target construction when direct congestion labels are unavailable.
- A framework for systematic model comparison with statistical testing.

### What this study IS NOT:
- A congestion prediction system ready for deployment on modern networks.
- Evidence that the TCN-Attention-BiLSTM architecture predicts network congestion in any practical sense.
- A claim that the proxy congestion score reflects real-world system stress.

### Recommended use:
The methodology (feature engineering pipeline, hybrid architecture design, ablation study framework, rolling-origin evaluation) can be **reapplied** to modern telemetry data (e.g., from the project's own Flask/Prometheus/Locust testbed, or from production monitoring systems) where ground truth congestion labels are available. The NASA HTTP 1995 results serve as a **baseline proof of concept**, not a production-ready finding.

## 4. References to Supporting Documents

| Document | Path | Content |
|---|---|---|
| Proxy target definition | `docs/PROXY_TARGET_DEFINITION.md` | Exact formula, weights, normalization |
| Proxy validation report | `docs/PROXY_VALIDATION_REPORT.md` | Correlation with real testbed telemetry |
| Research integrity audit | `outputs/reports/research_integrity_audit.md` | Full artifact audit and gap analysis |
| Ablation study results | `outputs/ablation_study/ablation_comparison.md` | Component contribution analysis |
| Rolling CV results | `outputs/rolling_cv/rolling_cv_report.md` | Multi-fold evaluation with statistical tests |
