# Paper Summary

## Tentative Title

Early Prediction of Web System Congestion Using Multivariate Time-Series Monitoring Data

## Research Problem

The research problem is early prediction of web system congestion from multivariate monitoring time series. The target signals include resource pressure, request load, latency, throughput, and error-rate behavior. The goal is to identify or forecast congestion before severe latency or availability degradation is observed.

## Motivation

Most web monitoring workflows are reactive: alerts fire after CPU, latency, or error-rate thresholds are already violated. A predictive pipeline can support earlier warning, capacity planning, and safer autoscaling decisions. This repository studies that direction as a reproducible research prototype.

## Data Sources

The repository contains previous/public experimental datasets under `Data/`, including cluster/cloud trace samples and synthetic workload data. It also contains Docker testbed data generated from Prometheus, cAdvisor, Locust, and a small Flask web application exposing request, latency, error, and inflight metrics.

The Docker dataset is production-like laboratory data. It is not a production log from a real deployed web system.

## Testbed Design

The testbed includes:

- Web app: Flask application with `/`, `/cpu`, `/io`, `/maybe-error`, and `/metrics` endpoints.
- Prometheus: scrapes web application metrics and cAdvisor.
- cAdvisor: exposes container and host resource metrics.
- Locust: generates HTTP load against the web application.

Implemented load profiles:

- `normal`
- `gradual`
- `spike`
- `stress`
- `recovery`

The initial short run used 90 seconds per profile. The long-run timestamp `20260517_211328` used 30 minutes per profile.

## Labeling Strategy

The testbed labeler assigns `congestion_label` using rule-based thresholds and trend signals:

- CPU threshold.
- Memory threshold.
- Response-time threshold.
- Error-rate threshold.
- High request-rate threshold combined with rising request and latency trend.

For the short initial run, thresholds in `Data/testbed/testbed_labeled.label_report.json` were:

- CPU: `85.0`
- Memory: `450.0` MB
- Response time: `296.67936933429814` ms
- Error rate: `2.0` percent
- High request-rate threshold: `87.24001158436407`

For long-run `20260517_211328`, thresholds in `Data/testbed/longrun_20260517_211328/testbed_labeled.label_report.json` were:

- CPU: `85.0`
- Memory: `450.0` MB
- Response time: `302.7718332486868` ms
- Error rate: `2.0` percent
- High request-rate threshold: `114.44111271236474`

## Models

The current experiment pipeline includes:

- Persistence.
- Moving Average.
- ARIMA.
- LSTM.
- BiLSTM.
- TCN.
- TCN + BiLSTM.
- TCN + BiLSTM + Temporal Attention.
- TCN + Feature Attention + BiLSTM + Temporal Attention.

## Evaluation Metrics

Generated artifacts include the following metrics where applicable:

- MAE.
- MSE.
- RMSE.
- R2.
- sMAPE.
- WAPE.
- MAPE.
- Accuracy.
- Precision.
- Recall.
- F1.
- ROC-AUC.
- Inference latency.

## Current Results

Short testbed data:

- `Data/testbed/prometheus_metrics.csv`: 109 rows.
- `Data/testbed/testbed_labeled.csv`: 109 rows.
- Label distribution: `0 = 60`, `1 = 49`.
- `Data/testbed/testbed_harmonized.csv`: 109 rows.

Short-run model-selection artifacts exist for seeds `42`, `123`, and `2026` under `paper_artifacts/model_selection_seed_*`. From those seed-level metrics:

- Best RMSE for all three seeds is `moving_average` with RMSE `9.589731235261173`.
- Best F1 for seed `42` is `lstm32` with F1 `0.7274014632995044`.
- Best F1 for seeds `123` and `2026` is `persistence` with F1 `0.7224759005580923`.

Long-run `20260517_211328` data:

- `Data/testbed/longrun_20260517_211328/prometheus_metrics.csv`: 1802 rows.
- `Data/testbed/longrun_20260517_211328/testbed_labeled.csv`: 1802 rows.
- `Data/testbed/longrun_20260517_211328/testbed_harmonized.csv`: 1802 rows.
- SQLite table `testbed_longrun_20260517_211328`: 1802 rows.
- Label distribution: `0 = 977`, `1 = 825`.

Long-run Locust aggregate statistics:

| profile | requests | failures | avg response ms | requests/s | failures/s |
|---|---:|---:|---:|---:|---:|
| normal | 80677 | 29 | 15.46845144343121 | 44.84419441093674 | 0.0161196082888204 |
| gradual | 264464 | 30602 | 44.77397171187247 | 146.99213323956027 | 17.008943604411275 |
| spike | 131471 | 3042 | 1212.7642706246895 | 73.06680342016264 | 1.6906330369749585 |
| stress | 75374 | 2049 | 2423.358668456029 | 41.89882073481073 | 1.138995989142505 |
| recovery | 79828 | 64 | 20.709394761219254 | 44.36622984886152 | 0.0355694582142498 |

Long-run model selection used seeds `42`, `123`, and `2026` with `--quick-epochs 2`. For all three seeds, all F1 values in `model_selection_metrics.csv` are `0.0`. By RMSE, `moving_average` is best for all three seeds with RMSE `0.3970318687636076`. The full `TCN + Feature Attention + BiLSTM + Temporal Attention` model has RMSE values `5.697774657009578`, `8.439515567156807`, and `4.639225895431911` for seeds `42`, `123`, and `2026`.

These results do not support a claim that the hybrid model wins absolutely. On the short artifacts, simple baselines are competitive. On long-run `20260517_211328`, Moving Average is the strongest RMSE model and the classification F1 results are not useful because they are all zero on the validation split.

## Stability

Short-run stability is stored in `paper_artifacts/stability_test.csv`. Selected short-run mean and standard deviation across 3 seeds:

| model | RMSE mean | RMSE std | F1 mean | F1 std |
|---|---:|---:|---:|---:|
| moving_average | 9.589731 | 0.0 | 0.685195 | 0.0 |
| persistence | 11.261423 | 0.0 | 0.722476 | 0.0 |
| tcn32 | 10.505006 | 0.242060 | 0.628756 | 0.002888 |
| tcn_bilstm32_temporal_attention | 10.776463 | 0.903730 | 0.640680 | 0.027659 |
| tcn_feature_attention_bilstm_temporal_attention | 11.672863 | 0.106964 | 0.621097 | 0.010835 |

Long-run stability is stored in `paper_artifacts/longrun_20260517_211328/stability_test.csv`:

| model | RMSE mean | RMSE std | F1 mean | F1 std |
|---|---:|---:|---:|---:|
| moving_average | 0.3970318687636076 | 0.0 | 0.0 | 0.0 |
| persistence | 0.44739381957985075 | 5.551115123125783e-17 | 0.0 | 0.0 |
| arima | 4.789169763996206 | 0.0 | 0.0 | 0.0 |
| tcn32 | 14.375371574903318 | 1.5094908355804712 | 0.0 | 0.0 |
| tcn_feature_attention_bilstm_temporal_attention | 6.258838706532765 | 1.6013836290949528 | 0.0 | 0.0 |

The long-run stability result confirms that the baseline forecast remains strong by RMSE on this particular testbed split. It does not establish reliable classification performance because F1 is zero for all models.

## Ablation

Short-run ablation is stored in `paper_artifacts/ablation_architecture.csv`. It is inconclusive: temporal attention improves over `TCN + BiLSTM`, but the full feature-attention variant does not dominate the simpler TCN or moving-average baseline by RMSE.

Long-run ablation is stored in `paper_artifacts/longrun_20260517_211328/ablation_architecture.csv`:

| variant | RMSE mean | F1 mean |
|---|---:|---:|
| TCN only | 14.375371574903318 | 0.0 |
| TCN + BiLSTM | 7.634848592077406 | 0.0 |
| TCN + BiLSTM + Temporal Attention | 6.41122515157518 | 0.0 |
| TCN + Feature Attention + BiLSTM + Temporal Attention | 6.258838706532765 | 0.0 |

On long-run `20260517_211328`, the full architecture improves RMSE over the simpler neural variants, but it does not beat Moving Average by RMSE and does not produce useful F1 on the validation split.

## Limitations

- No real production log has been integrated yet.
- The Docker testbed is laboratory/production-like and does not fully represent production traffic.
- Short and single-session testbeds can favor inertia-based baselines such as Persistence and Moving Average.
- The long-run covers 30 minutes per profile on one machine, not multiple days or independent deployments.
- More independent workloads and stronger confidence intervals are needed.
- Deployment on Kubernetes or a real cloud environment is needed before making strong operational claims.
- cAdvisor labels vary by platform; Docker Desktop/WSL required fallback queries in the collector.

## Future Work

- Longer and multi-day testbed experiments.
- Production log integration.
- Kubernetes autoscaling integration.
- Online learning and drift detection.
- More robust confidence intervals and statistical tests.
- Conference paper submission after long-run and production-like validation are stronger.

## Paper Readiness

Ready:

- Reproducible pipeline.
- Docker testbed.
- Prometheus/cAdvisor/Locust integration.
- Artifact generator.
- Initial stability, ablation, threshold search, imputation, ARIMA behavior, and recommendation audit artifacts.

Need more:

- Multi-day long-run results.
- Stronger ablation evidence.
- More stable confidence intervals.
- Production or cloud/Kubernetes validation.
