# PHASE 2 — Data Pipeline Report

## Pipeline Overview

```
Raw HTTP Logs → Parse → Aggregate 1-min → Normalize 0-1 → Float16 → Sliding Windows → Train/Val/Test
```

## Data Source

- **Source:** NASA Kennedy Space Center HTTP Server Logs (1995)
- **Files:** NASA_access_log_Jul95.gz, NASA_access_log_Aug95.gz
- **Raw lines:** ~3.46 million
- **Time range:** July-August 1995

## Preprocessing Steps

1. **Parse:** Regex parser cho NASA HTTP log format
2. **Aggregate:** Resample theo cửa sổ 1 phút
3. **Features:** 19 features (request_count, bytes_sum, error_rate, spike_score, ...)
4. **Normalize:** MinMax scaling về [0, 1]
5. **Window:** Sliding windows (lookback=60, horizon=15)
6. **Split:** Chronological (70% train, 15% val, 15% test)
7. **Storage:** float16 NPZ

## Output

| Split | Windows | Shape |
|---|---|---|
| Train | 62,425 | [62425, 60, 19] |
| Validation | 13,330 | [13330, 60, 19] |
| Test | 13,330 | [13330, 60, 19] |

## Features

1. request_count
2. bytes_sum
3. bytes_mean
4. bytes_max
5. bytes_p95
6. unique_hosts
7. status_2xx
8. status_3xx
9. status_4xx
10. status_5xx
11. error_count
12. error_rate
13. method_get
14. method_post
15. method_head
16. method_other
17. throughput_bytes_per_min
18. request_spike_score
19. congestion_score_proxy

## Target

- **Name:** target_next_congestion_score
- **Type:** proxy (derived from HTTP features)
- **Range:** [0, 1]

## Leakage Check

- ✅ Chronological split (no shuffle)
- ✅ Train-only normalization
- ✅ No future data in features
