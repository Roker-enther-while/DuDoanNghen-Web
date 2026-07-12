# Proxy Congestion Score Validation Report — Testbed Results

**Date**: 2026-07-11
**Testbed**: Flask webapp + Locust + Prometheus metrics
**Scenarios**: baseline (10 users), gradual ramp (60 users), spike (120 users), sustained stress (120 users)

## Executive Summary

**The proxy congestion score does NOT correlate with real congestion indicators.**

The proxy score correlates strongly with **request rate** (r=0.82 in ramp/spike scenarios) but has **weak or negative correlation with latency** (overall r=-0.07) and **zero correlation with error rate** (error rate was constant at 0%). This confirms the proxy score measures **load intensity**, not **congestion level**.

## Correlation Results by Scenario

### Baseline (10 users, low stable load)
| Metric | Pearson r | p-value | Spearman r |
|---|---|---|---|
| Latency Mean | -0.391 | 0.048 | 0.154 |
| Error Rate | 0.000 | 1.000 | 0.000 |
| In-flight | 0.100 | 0.627 | 0.014 |
| Request Rate | 0.228 | 0.263 | -0.042 |

### Gradual Ramp (60 users, linear increase)
| Metric | Pearson r | p-value | Spearman r |
|---|---|---|---|
| Latency Mean | -0.072 | 0.728 | -0.058 |
| Error Rate | 0.000 | 1.000 | 0.000 |
| In-flight | 0.331 | 0.099 | 0.452 |
| Request Rate | **0.823** | **0.000** | **0.819** |

### Spike (120 users, sudden spike)
| Metric | Pearson r | p-value | Spearman r |
|---|---|---|---|
| Latency Mean | -0.155 | 0.450 | -0.095 |
| Error Rate | 0.000 | 1.000 | 0.000 |
| In-flight | -0.056 | 0.787 | -0.022 |
| Request Rate | **0.820** | **0.000** | **0.791** |

### Sustained Stress (120 users, prolonged overload)
| Metric | Pearson r | p-value | Spearman r |
|---|---|---|---|
| Latency Mean | -0.174 | 0.271 | -0.275 |
| Error Rate | 0.000 | 1.000 | 0.000 |
| In-flight | 0.205 | 0.193 | 0.174 |
| Request Rate | -0.013 | 0.935 | **0.849** |

### Overall (All Profiles Combined, 120 samples)
| Metric | Pearson r | p-value | Spearman r |
|---|---|---|---|
| Latency Mean | **-0.067** | 0.469 | — |
| Error Rate | 0.000 | 1.000 | — |
| In-flight | **-0.289** | 0.001 | — |
| Request Rate | -0.058 | 0.532 | — |

## Key Findings

### 1. Proxy score measures LOAD, not CONGESTION
- Strong positive correlation with request rate (r=0.82) in ramp/spike scenarios
- The proxy formula weights request_count (35%) and bytes_sum (20%) heavily
- High request rate → high proxy score, regardless of whether the system is struggling

### 2. Proxy score does NOT predict latency (the real congestion signal)
- Overall correlation with latency: r=-0.07 (essentially zero)
- In some scenarios, the correlation is NEGATIVE (high proxy → LOW latency)
- This happens because: during high but manageable load, the system handles requests efficiently (low latency). During真正的 congestion (high latency), request rate may actually DROP as the system slows down.

### 3. Error rate was constant (0%)
- The Flask app's `/maybe-error` endpoint has a low default error probability
- Locust profiles didn't generate enough errors to create variance
- Future validation should increase error probability in stress scenarios

### 4. The proxy score is a load intensity metric, not a congestion metric
- It correctly identifies "how busy is the server" (strong correlation with request rate)
- It does NOT identify "how much is the server struggling" (weak/negative correlation with latency)
- For congestion prediction, the formula needs to be revised to weight latency/error components more heavily, or the metric should be reinterpreted

## Implications for the Research

1. **The proxy target is not validated as a congestion proxy.** All model performance metrics (R², MAE, RMSE) are measured against this unvalidated target. The model's ability to predict the proxy score does NOT imply ability to predict real congestion.

2. **The ablation study results must be interpreted with this caveat.** When we say "Attention+BiLSTM has the best R²", we mean it best predicts the load intensity metric, NOT that it best predicts congestion.

3. **Future work should revise the proxy formula** to include latency/error components from real telemetry, or use a different target altogether.

## Figures

Time-series overlay plots (proxy score vs latency/error rate) for each scenario:
- `outputs/figures/proxy_validation/proxy_validation_baseline.png`
- `outputs/figures/proxy_validation/proxy_validation_gradual.png`
- `outputs/figures/proxy_validation/proxy_validation_spike.png`
- `outputs/figures/proxy_validation/proxy_validation_stress.png`

## Recommendations

1. **Immediate**: Reinterpret the proxy score as a "load intensity metric" in all reports and the dashboard
2. **Short-term**: Revise the proxy formula to include latency/error components from testbed telemetry
3. **Long-term**: Use real telemetry (latency, error rate, queue depth) as the target variable for congestion prediction
