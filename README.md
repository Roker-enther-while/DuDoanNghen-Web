# Web Congestion Prediction with Multivariate Time-Series AI

## Overview

This repository is a research prototype for a student research project on early congestion prediction in web systems using multivariate time-series monitoring data.

The project includes data processing, statistical and neural baselines, a hybrid deep learning model, a Docker-based web testbed, Prometheus/cAdvisor monitoring, Locust load generation, and artifact generation for tables and figures used in reports or paper drafts.

The main hybrid architecture is described neutrally as:

`TCN + Feature Attention + BiLSTM + Temporal Attention`

The repository focuses on reproducible experimentation and honest artifact generation. Results depend on the dataset, testbed duration, random seed, and evaluation metric.

## Key Features

- Multivariate time-series preprocessing for workload and monitoring metrics.
- Baselines including Persistence, Moving Average, ARIMA, LSTM, BiLSTM, and TCN variants.
- Hybrid architecture variants for ablation study.
- Docker Compose testbed with a measurable web application.
- Prometheus and cAdvisor metric collection.
- Locust profiles for `normal`, `gradual`, `spike`, `stress`, and `recovery` traffic.
- Prometheus-to-CSV collector.
- Rule-based congestion labeling with threshold and trend explanations.
- Paper artifact generation for metrics, stability tests, ablation study, threshold search, imputation report, ARIMA behavior analysis, and recommendation-engine audit.

## Repository Structure

```text
Data/                 Input datasets and generated testbed CSV files
configs/              Experiment configuration files
docs/                 Audit notes, run summaries, and pipeline documentation
paper_artifacts/      Generated tables, figures, metrics, and model-selection outputs
src/data/             Data pool, schema harmonization, external data helpers
src/models/           Neural model definitions
src/services/         Inference, monitoring, anomaly, decision, recommendation logic
src/tools/            Training, evaluation, collector, labeler, paper artifact scripts
src/utils/            Data loading, preprocessing, metrics, utilities
testbed/              Docker web app, Prometheus config, Locust load profiles
```

## Testbed Components

- `testbed/webapp/`: Flask web application exposing Prometheus metrics.
- `testbed/prometheus/prometheus.yml`: scrape configuration for web app and cAdvisor.
- `testbed/docker-compose.yml`: starts the web app, Prometheus, and cAdvisor.
- `testbed/load/locustfile.py`: Locust workloads for five load profiles.

The testbed is production-like laboratory data. It is not a production log from a real deployed web service.

## How to Run Testbed

```powershell
docker compose -f testbed/docker-compose.yml up --build
```

For a background run:

```powershell
docker compose -f testbed/docker-compose.yml up --build -d
```

Check targets:

```powershell
Invoke-RestMethod http://localhost:8080/
Invoke-RestMethod http://localhost:9090/api/v1/targets
```

## How to Run Load Profiles

Short smoke run:

```powershell
powershell -ExecutionPolicy Bypass -File testbed/run_load_profiles.ps1 -RunTime 5m
```

Longer research run:

```powershell
powershell -ExecutionPolicy Bypass -File testbed/run_load_profiles.ps1 -RunTime 30m
```

Timestamped end-to-end long-run pipeline:

```powershell
powershell -ExecutionPolicy Bypass -File testbed/run_longrun_pipeline.ps1 -RunMinutesPerProfile 30
```

The five profiles are `normal`, `gradual`, `spike`, `stress`, and `recovery`.

## How to Collect Prometheus Metrics

```powershell
python -m src.tools.collect_prometheus_metrics `
  --minutes 30 `
  --output Data/testbed/prometheus_metrics.csv `
  --load-profile mixed
```

The collector writes both CSV data and a metadata JSON file containing the PromQL queries and time range.

## How to Label Congestion

```powershell
python -m src.tools.label_testbed_congestion `
  --input Data/testbed/prometheus_metrics.csv `
  --output Data/testbed/testbed_labeled.csv
```

The labeler creates:

- `congestion_label`
- `label_reason`
- `label_rule_version`
- a JSON label report with thresholds and missing-value counts

## How to Run Paper Experiments

```powershell
python -m src.tools.prepare_testbed_dataset `
  --input Data/testbed/testbed_labeled.csv `
  --output Data/testbed/testbed_harmonized.csv `
  --db-path Data/processed/nckh_biglogs_training_pool.sqlite `
  --table testbed_pool

python -m src.tools.run_paper_experiments `
  --db-path Data/processed/nckh_biglogs_training_pool.sqlite `
  --testbed-csv Data/testbed/testbed_labeled.csv `
  --raw-testbed-csv Data/testbed/prometheus_metrics.csv `
  --output-dir paper_artifacts `
  --quick-epochs 2 `
  --seeds 42 123 2026
```

The generated artifacts are written under `paper_artifacts/`.

## Current Evidence and Limitations

The repository currently has both a short smoke run and one timestamped long-run with real generated artifacts:

- Docker + Prometheus + cAdvisor + Locust testbed is implemented.
- Five load profiles were run: `normal`, `gradual`, `spike`, `stress`, `recovery`.
- Initial short testbed data:
  - `Data/testbed/prometheus_metrics.csv`: 109 rows.
  - `Data/testbed/testbed_labeled.csv`: 109 rows.
  - labels: `0 = 60`, `1 = 49`.
- Long-run `20260517_211328` used 30 minutes per profile:
  - `Data/testbed/longrun_20260517_211328/prometheus_metrics.csv`: 1802 rows.
  - `Data/testbed/longrun_20260517_211328/testbed_labeled.csv`: 1802 rows.
  - `Data/testbed/longrun_20260517_211328/testbed_harmonized.csv`: 1802 rows.
  - SQLite table `testbed_longrun_20260517_211328`: 1802 rows.
  - labels: `0 = 977`, `1 = 825`.
- Stability, ablation, threshold search, imputation, ARIMA behavior, and recommendation audit artifacts exist under `paper_artifacts/` and `paper_artifacts/longrun_20260517_211328/`.

Important limitations:

- This is a research prototype, not an operational monitoring product.
- The Docker testbed is a laboratory/production-like environment, not real production traffic.
- On the short initial artifacts, simple baselines such as Persistence and Moving Average are competitive.
- On long-run `20260517_211328`, Moving Average is the strongest RMSE model, while classification F1 is 0.0 for all models on the validation split.
- The current artifacts do not prove that `TCN + Feature Attention + BiLSTM + Temporal Attention` wins under every metric or seed.
- Multi-day runs and independent workloads are still needed for stronger stability claims.

## Reproducibility Notes

- Do not report metrics unless they come from generated CSV, JSON, Markdown, or figure artifacts.
- Use timestamped output directories for long-run experiments to avoid overwriting previous evidence.
- Keep model claims tied to a specific dataset, seed set, and metric.
- cAdvisor labels can differ across Docker Desktop, WSL, and Linux. The collector includes fallback PromQL for Docker Desktop/WSL.

## Citation / Paper Artifacts

Use `paper_artifacts/paper_summary.md` as the living summary for a paper draft. The tables and figures under `paper_artifacts/` are generated from code and should be preferred over manually typed metrics.

Recommended neutral title:

`Early Prediction of Web System Congestion Using Multivariate Time-Series Monitoring Data`

## License / Academic Use

This repository is intended for academic research and student experimentation. Validate all results in the target environment before using the pipeline for operational decisions.
