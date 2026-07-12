# Data Pipeline

This repository prepares public time-series data for web-system congestion forecasting. It does not train models, run APIs, or build a web UI.

## Sources

The default small public source is NASA Kennedy Space Center HTTP access logs from the Internet Traffic Archive:

- `nasa_jul95`: `https://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz`
- `nasa_aug95`: `https://ita.ee.lbl.gov/traces/NASA_access_log_Aug95.gz`

Google Cluster Trace support is a skeleton for later work. The 2019 trace is very large and should be accessed through BigQuery or a small export/sample, not downloaded by this pipeline.

## NASA Feature Set

NASA HTTP logs do not include CPU usage, memory usage, or response time. The pipeline therefore derives practical access-log features: request count, bytes, unique hosts, status classes, error rate, HTTP method counts, throughput, request spike score, and a proxy congestion score.

`congestion_score_proxy` is not a true congestion label. It is a temporary target proxy built from request volume, bytes, unique hosts, error rate, and request spikes. With real telemetry, replace it with measured congestion score, future response time, CPU/memory pressure, or a binary/multiclass congestion label.

## Float16 Rule

The pipeline never casts raw count or byte values directly to `float16`. It uses:

`raw int/float -> float32 -> train-only min-max normalize -> float16`

The scaler is fit on train rows only to avoid leakage into validation or test.

## Commands

List sources:

```powershell
python scripts/fetch_public_data.py --list-sources
```

Run tests:

```powershell
python -m pytest -q
```

Run offline smoke pipeline:

```powershell
python scripts/run_data_pipeline.py --config configs/data/nasa_http_smoke.yaml
```

Run NASA pipeline with public downloads:

```powershell
python scripts/run_data_pipeline.py --config configs/data/nasa_http.yaml
```

Prepare Google Cluster from a local sample:

```powershell
python scripts/prepare_google_cluster.py --input data/raw/google_cluster/sample.csv --window-minutes 5
```

## Outputs

- Raw manifest: `outputs/metrics/raw_data_manifest.json`
- Time series: `data/processed/nasa_http/timeseries_1min.csv`
- Splits: `data/processed/nasa_http/splits/`
- Scaler: `data/processed/nasa_http/scaler.json`
- Normalized arrays: `data/processed/nasa_http/normalized/`
- Model windows: `data/processed/nasa_http/windows/windows_fp16.npz`
- Quality report: `outputs/reports/data_quality_report.md`
- Pipeline manifest: `outputs/metrics/data_pipeline_manifest.json`

Check dtype quickly:

```powershell
@'
import numpy as np
d=np.load('data/processed/nasa_http/windows/windows_fp16.npz', allow_pickle=True)
print(d['X_train'].dtype, d['y_train'].dtype, d['X_train'].shape, d['y_train'].shape)
'@ | python -
```
