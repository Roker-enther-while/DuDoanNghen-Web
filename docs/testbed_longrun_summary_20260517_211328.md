# Testbed Long-Run Summary

Timestamp: 20260517_211328

This summary is generated from the timestamped long-run artifacts. No values below are manually invented.

- Run minutes per profile: 30
- Started UTC: 2026-05-17T14:13:28.3997514Z
- Ended UTC: 2026-05-17T16:43:37.7727759Z
- Data directory: Data/testbed/longrun_20260517_211328
- Artifact directory: paper_artifacts/longrun_20260517_211328
- SQLite table: testbed_longrun_20260517_211328

## Data Outputs

| file/table | rows |
|---|---:|
| Data/testbed/longrun_20260517_211328/prometheus_metrics.csv | 1802 |
| Data/testbed/longrun_20260517_211328/testbed_labeled.csv | 1802 |
| Data/testbed/longrun_20260517_211328/testbed_harmonized.csv | 1802 |
| SQLite table testbed_longrun_20260517_211328 | 1802 |

## Label Distribution

| label | rows |
|---|---:|
| 0 | 977 |
| 1 | 825 |

Positive label rate: 0.45782463928967815.

Label rule version: testbed_rule_v1.

Thresholds read from `Data/testbed/longrun_20260517_211328/testbed_labeled.label_report.json`:

| signal | threshold |
|---|---:|
| cpu_usage | 85.0 |
| memory_usage | 450.0 |
| response_time | 302.7718332486868 |
| error_rate | 2.0 |
| request_rate_high | 114.44111271236474 |

Missing values before and after labeling were 0 for the monitored columns in the label report.

## Locust Load Profiles

Each profile was executed for 30 minutes.

| profile | requests | failures | avg response ms | requests/s | failures/s |
|---|---:|---:|---:|---:|---:|
| normal | 80677 | 29 | 15.46845144343121 | 44.84419441093674 | 0.0161196082888204 |
| gradual | 264464 | 30602 | 44.77397171187247 | 146.99213323956027 | 17.008943604411275 |
| spike | 131471 | 3042 | 1212.7642706246895 | 73.06680342016264 | 1.6906330369749585 |
| stress | 75374 | 2049 | 2423.358668456029 | 41.89882073481073 | 1.138995989142505 |
| recovery | 79828 | 64 | 20.709394761219254 | 44.36622984886152 | 0.0355694582142498 |

## Model Selection

Experiments were run with seeds 42, 123, and 2026 using `--quick-epochs 2`.

| seed | top by F1 | top F1 | top by RMSE | top RMSE | full architecture RMSE | full architecture F1 |
|---:|---|---:|---|---:|---:|---:|
| 42 | moving_average | 0.0 | moving_average | 0.3970318687636076 | 5.697774657009578 | 0.0 |
| 123 | moving_average | 0.0 | moving_average | 0.3970318687636076 | 8.439515567156807 | 0.0 |
| 2026 | moving_average | 0.0 | moving_average | 0.3970318687636076 | 4.639225895431911 | 0.0 |

All F1 values in the model-selection metrics are 0.0 for this validation split. The long-run evidence therefore should not be described as proving classification superiority. By RMSE, `moving_average` is strongest on this long-run table.

## Stability

| model | seed_count | RMSE mean | RMSE std | F1 mean | F1 std |
|---|---:|---:|---:|---:|---:|
| arima | 3 | 4.789169763996206 | 0.0 | 0.0 | 0.0 |
| moving_average | 3 | 0.3970318687636076 | 0.0 | 0.0 | 0.0 |
| persistence | 3 | 0.44739381957985075 | 5.551115123125783e-17 | 0.0 | 0.0 |
| tcn32 | 3 | 14.375371574903318 | 1.5094908355804712 | 0.0 | 0.0 |
| tcn_bilstm32_no_attn | 3 | 7.634848592077406 | 1.3188517503452985 | 0.0 | 0.0 |
| tcn_bilstm32_temporal_attention | 3 | 6.41122515157518 | 1.3662651628186743 | 0.0 | 0.0 |
| tcn_feature_attention_bilstm_temporal_attention | 3 | 6.258838706532765 | 1.6013836290949528 | 0.0 | 0.0 |

## Ablation

| variant | RMSE mean | F1 mean |
|---|---:|---:|
| TCN only | 14.375371574903318 | 0.0 |
| TCN + BiLSTM | 7.634848592077406 | 0.0 |
| TCN + BiLSTM + Temporal Attention | 6.41122515157518 | 0.0 |
| TCN + Feature Attention + BiLSTM + Temporal Attention | 6.258838706532765 | 0.0 |

The ablation suggests the combined architecture lowers RMSE versus the simpler neural variants on this run, but it is still weaker than Moving Average by RMSE and has F1 0.0 on this validation split.

## Other Artifact Checks

Generated top-level tables and figures include:

- `stability_test.csv` and `tables/stability_test.md`
- `ablation_architecture.csv` and `tables/ablation_architecture.md`
- `threshold_search.csv` and `figures/threshold_search_f1.png`
- `imputation_report.csv` and `figures/imputation_missing_before.png`
- `arima_behavior_analysis.csv` and `figures/arima_behavior_analysis.png`
- `recommendation_engine_audit.csv` and `figures/recommendation_alert_counts.png`

Artifact state scan:

- State markers `not_run`, `not_available_or_not_run`, and `failed`: 0 files.
- Broad text matches for `error` and `missing` remain in expected data contexts: Locust failure/error-rate CSV columns, recommendation audit scenario text, and imputation report column names.

## Interpretation

This long-run is a production-like laboratory experiment, not production traffic from a deployed service. It expands the evidence base beyond the 109-row short run, but it does not justify a claim that `TCN + Feature Attention + BiLSTM + Temporal Attention` is the top model overall. On this run, Moving Average is the strongest RMSE baseline, while the full neural architecture is useful mainly as an ablation candidate that improves over simpler neural variants.
