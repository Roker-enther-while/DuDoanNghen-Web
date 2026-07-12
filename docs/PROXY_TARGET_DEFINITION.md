# Proxy Congestion Score — Formal Definition

## Overview

The `proxy_congestion_score` (stored as `congestion_score_proxy` in the dataset and `target_next_congestion_score` as the prediction target) is a **synthetic composite metric** derived from HTTP access log features. It is **not** a measured congestion label, a ground-truth congestion annotation, or a directly observed network/system state.

This document defines the exact computation as implemented in `src/data/nasa_http.py:add_congestion_proxy_target()`.

## Feature Computation

The proxy score is a weighted sum of five components, each normalized to [0, 1]:

### Components

| Component | Weight | Normalization | Source |
|---|---|---|---|
| `request_count` | 0.35 | Expanding min-max | Aggregated 1-min window request count |
| `bytes_sum` | 0.20 | Expanding min-max | Total response bytes in 1-min window |
| `unique_hosts` | 0.15 | Expanding min-max | Unique source IPs in 1-min window |
| `error_rate` | 0.20 | Pre-bounded [0, 1] | (status_4xx + status_5xx) / request_count |
| `request_spike_score` | 0.10 | Rolling z-score → clip → scale | Anomalous request volume indicator |

### Formula

```
proxy_score = 0.35 * minmax(request_count)
            + 0.20 * minmax(bytes_sum)
            + 0.15 * minmax(unique_hosts)
            + 0.20 * error_rate
            + 0.10 * request_spike_score
```

The result is clipped to [0, 1].

### Expanding Min-Max Normalization

```python
def _expanding_minmax(series):
    values = series.cummin()  # running minimum
    running_min = values.cummin()
    running_max = values.cummax()
    denom = (running_max - running_min).replace(0, NaN)
    return ((values - running_min) / denom).fillna(0).clip(0, 1)
```

Note: This uses **expanding** (cumulative) min-max, not global min-max. This means:
- The score at time t depends on all observations from t=0 to t.
- Early values may be noisy (small denominator).
- The score is non-stationary by construction.

### Request Spike Score

```python
rolling_mean = request_count.rolling(60, min_periods=2).mean().shift(1)
rolling_std = request_count.rolling(60, min_periods=2).std().shift(1).replace(0, NaN)
z_score = ((request_count - rolling_mean) / rolling_std).replace([inf, -inf], 0).fillna(0)
request_spike_score = (z_score.clip(lower=0, upper=3) / 3.0).clip(0, 1)
```

This captures sudden increases in request volume relative to the trailing 60-minute window. The shift(1) prevents leakage (only uses past data).

### Target Construction

```python
target_next_congestion_score = congestion_score_proxy.shift(-horizon_steps)
```

Where `horizon_steps = 15` (15 minutes ahead with 1-minute windows).

## Known Limitations of This Target

1. **Self-referential**: The proxy score is computed from the same features used as model inputs. The model has access to `congestion_score_proxy` as feature #19 and predicts its own future value.

2. **No external validation**: The proxy score has not been validated against measured congestion (latency, queue depth, error rate from a real system). See `docs/PROXY_VALIDATION_REPORT.md` for the testbed-based validation attempt.

3. **Expanding normalization non-stationarity**: The expanding min-max makes the score dependent on the full history up to each point, which can distort comparisons across different time periods.

4. **Weight arbitrariness**: The weights (0.35, 0.20, 0.15, 0.20, 0.10) were chosen heuristically, not optimized against any ground truth.

5. **Single-source proxy**: Validated only on NASA HTTP 1995 data, which represents a single web server from the dial-up era.

## Provenance

- Implementation: `src/data/nasa_http.py:198-242`
- Pipeline config: `configs/data/nasa_http_3m.yaml`
- Windowing: `src/data/windowing.py`
- Feature count: 19 (17 raw + 1 derived spike score + 1 proxy score)
- Target: `target_next_congestion_score` (shifted by 15 steps)
