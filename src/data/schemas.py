"""Shared schema constants for data preparation artifacts."""

NASA_FEATURE_COLUMNS = [
    "request_count",
    "bytes_sum",
    "bytes_mean",
    "bytes_max",
    "bytes_p95",
    "unique_hosts",
    "status_2xx",
    "status_3xx",
    "status_4xx",
    "status_5xx",
    "error_count",
    "error_rate",
    "method_get",
    "method_post",
    "method_head",
    "method_other",
    "throughput_bytes_per_min",
    "request_spike_score",
    "congestion_score_proxy",
]

NASA_TARGET_COLUMN = "target_next_congestion_score"

GOOGLE_CLUSTER_FEATURE_COLUMNS = [
    "cpu_usage",
    "memory_usage",
    "task_count",
    "failed_task_count",
    "evicted_task_count",
    "mean_cpu_request",
    "mean_memory_request",
    "resource_pressure_score",
]

GOOGLE_CLUSTER_TARGET_COLUMN = "target_next_resource_pressure_score"
