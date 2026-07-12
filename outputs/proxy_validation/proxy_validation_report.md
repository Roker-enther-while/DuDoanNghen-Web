# Proxy Congestion Score Validation Report

**Testbed data**: `data/testbed/longrun_20260517_211328/testbed_labeled.csv` (1802 rows)
**Proxy score range**: [0.0000, 0.6503]
**Proxy score mean**: 0.3259 (std=0.1508)

## Validation Status

**NOT VALIDATED**

- CRITICAL FINDING: Proxy score is NEGATIVELY correlated with congestion indicators (response_time, error_rate). HIGH proxy score corresponds to LOW congestion. The proxy score measures LOAD LEVEL, not CONGESTION LEVEL. This is a significant limitation that must be reported in the research.
- Response Time (ms): r=-0.5436 (NEGATIVE, strong)
- Error Rate (%): r=-0.5440 (NEGATIVE, strong)
- CPU Usage (%): r=-0.2994 (NEGATIVE, weak)
- Request Rate (req/s): r=0.5734 (POSITIVE)
- Throughput (bytes/s): r=0.5793 (POSITIVE)
- RECOMMENDATION: Revise the proxy formula to include latency/error components directly, or reinterpret the score as a 'load intensity' metric rather than 'congestion'.

## Correlation with Real Telemetry

| Metric | Pearson r | p-value | Spearman r | p-value | n |
|---|---|---|---|---|---|
| Response Time (ms) | -0.5436 | 0.00e+00 | -0.4428 | 0.00e+00 | 1802 |
| Error Rate (%) | -0.5440 | 0.00e+00 | -0.4103 | 0.00e+00 | 1802 |
| CPU Usage (%) | -0.2994 | 0.00e+00 | -0.3337 | 0.00e+00 | 1802 |
| Memory Usage (MB) | -0.5719 | 0.00e+00 | -0.6112 | 0.00e+00 | 1802 |
| Request Rate (req/s) | 0.5734 | 0.00e+00 | 0.6780 | 0.00e+00 | 1802 |
| In-flight Requests | -0.4071 | 0.00e+00 | -0.3005 | 0.00e+00 | 1802 |
| Throughput (bytes/s) | 0.5793 | 0.00e+00 | 0.5511 | 0.00e+00 | 1802 |

## Lag Analysis

| Metric | Best Lag (steps) | Correlation at Best Lag |
|---|---|---|
| response_time | -10 | -0.5384 |
| error_rate | -10 | -0.5397 |
| cpu_usage | -2 | -0.2977 |

## Quantile Analysis

### response_time
| Quantile | Proxy Threshold | Mean Real (High Proxy) | Mean Real (Low Proxy) | Ratio |
|---|---|---|---|---|
| 50% | 0.2754 | 32.5986 | 147.4696 | 0.22x |
| 70% | 0.4564 | 12.8680 | 123.1403 | 0.10x |
| 80% | 0.4872 | 15.3542 | 108.7430 | 0.14x |
| 90% | 0.5443 | 21.3134 | 97.7074 | 0.22x |
| 95% | 0.5693 | 23.0391 | 93.5973 | 0.25x |

### error_rate
| Quantile | Proxy Threshold | Mean Real (High Proxy) | Mean Real (Low Proxy) | Ratio |
|---|---|---|---|---|
| 50% | 0.2754 | 0.3490 | 1.3923 | 0.25x |
| 70% | 0.4564 | 0.0845 | 1.2079 | 0.07x |
| 80% | 0.4872 | 0.0990 | 1.0639 | 0.09x |
| 90% | 0.5443 | 0.1344 | 0.9528 | 0.14x |
| 95% | 0.5693 | 0.1436 | 0.9093 | 0.16x |

### cpu_usage
| Quantile | Proxy Threshold | Mean Real (High Proxy) | Mean Real (Low Proxy) | Ratio |
|---|---|---|---|---|
| 50% | 0.2754 | 51.6712 | 70.2736 | 0.74x |
| 70% | 0.4564 | 36.3352 | 71.5424 | 0.51x |
| 80% | 0.4872 | 43.1935 | 65.4264 | 0.66x |
| 90% | 0.5443 | 59.4959 | 61.1372 | 0.97x |
| 95% | 0.5693 | 63.6296 | 60.8311 | 1.05x |


## Interpretation

The proxy congestion score is a synthetic composite of HTTP log features. This validation
checks whether it correlates with independently measured system metrics (latency, error rate,
CPU usage) from a real testbed. If correlations are weak, the proxy score may not reflect
actual system congestion and should be interpreted with caution.

### Component Weights

| Component | Weight |
|---|---|
| request_count | 0.35 |
| bytes_sum | 0.20 |
| unique_hosts | 0.15 |
| error_rate | 0.20 |
| request_spike_score | 0.10 |

See `docs/PROXY_TARGET_DEFINITION.md` for the full formula.