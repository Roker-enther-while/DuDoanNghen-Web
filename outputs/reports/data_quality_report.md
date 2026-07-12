# Data Quality Report

- Raw rows parsed: 1154
- Raw rows skipped: 1
- Window count: 215
- Time range: 1995-07-01T04:00:00+00:00 to 1995-07-01T07:34:00+00:00
- Missing timestamp count: 0
- Duplicate timestamp count: 0

## Outlier Rates
- request_count: 0.023256
- bytes_sum: 0.018605
- error_rate: 0.000000

## Status Class Distribution
- status_2xx: 911
- status_3xx: 0
- status_4xx: 211
- status_5xx: 8

## Feature Statistics
- request_count: min=2.000000, max=20.000000, mean=5.255814, std=2.715100
- bytes_sum: min=2403.000000, max=48570.000000, mean=9491.586047, std=5743.400031
- bytes_mean: min=1201.500000, max=2428.500000, mean=1796.616279, std=373.776865
- bytes_max: min=1203.000000, max=2457.000000, mean=1803.000000, std=373.995249
- bytes_p95: min=1202.850000, max=2454.150000, mean=1802.361628, std=373.971421
- unique_hosts: min=2.000000, max=20.000000, mean=5.255814, std=2.715100
- status_2xx: min=1.000000, max=18.000000, mean=4.237209, std=2.663188
- status_3xx: min=0.000000, max=0.000000, mean=0.000000, std=0.000000
- status_4xx: min=0.000000, max=2.000000, mean=0.981395, std=0.166015
- status_5xx: min=0.000000, max=3.000000, mean=0.037209, std=0.270248
- error_count: min=1.000000, max=3.000000, mean=1.018605, std=0.166015
- error_rate: min=0.052632, max=0.500000, mean=0.243827, std=0.123756
- method_get: min=1.000000, max=17.000000, mean=4.223256, std=2.569303
- method_post: min=0.000000, max=1.000000, mean=0.023256, std=0.150715
- method_head: min=1.000000, max=2.000000, mean=1.009302, std=0.095999
- method_other: min=0.000000, max=0.000000, mean=0.000000, std=0.000000
- throughput_bytes_per_min: min=2403.000000, max=48570.000000, mean=9491.586047, std=5743.400031
- request_spike_score: min=0.000000, max=1.000000, mean=0.104542, std=0.172946
- congestion_score_proxy: min=0.042857, max=0.820000, mean=0.204776, std=0.116710

## Notes
- Dataset is suitable for minute-level HTTP workload time-series modeling.
- NASA HTTP logs do not include true CPU, memory, or response-time telemetry.
- Current congestion target is a proxy derived from request volume, bytes, hosts, errors, and spikes.
