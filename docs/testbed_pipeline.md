# Testbed Pipeline

Kien truc output moi phai ghi la `TCN + Feature Attention + BiLSTM + Temporal Attention`.

## Chay testbed

```powershell
docker compose -f testbed/docker-compose.yml up --build
```

## Sinh tai bang Locust

```powershell
powershell -ExecutionPolicy Bypass -File testbed/run_load_profiles.ps1 -RunTime 5m
```

Hoac tren Linux/macOS:

```bash
bash testbed/run_load_profiles.sh
```

## Thu metrics Prometheus thanh CSV

```powershell
python -m src.tools.collect_prometheus_metrics --minutes 30 --output Data/testbed/prometheus_metrics.csv
```

## Gan nhan nghe va harmonize dataset

```powershell
python -m src.tools.label_testbed_congestion --input Data/testbed/prometheus_metrics.csv --output Data/testbed/testbed_labeled.csv
python -m src.tools.prepare_testbed_dataset --input Data/testbed/testbed_labeled.csv --output Data/testbed/testbed_harmonized.csv --db-path Data/processed/nckh_biglogs_training_pool.sqlite --table testbed_pool
```

## Tao artifact cho paper

Script nay chi doc ket qua that tu CSV/JSON/SQLite. Neu thieu du lieu, artifact se ghi `not_run`.

```powershell
python -m src.tools.run_paper_experiments --db-path Data/processed/nckh_biglogs_training_pool.sqlite --testbed-csv Data/testbed/testbed_labeled.csv --raw-testbed-csv Data/testbed/prometheus_metrics.csv --output-dir paper_artifacts --quick-epochs 2 --seeds 42 123 2026
```

Ket qua bang/hinh nam trong `paper_artifacts/tables`, `paper_artifacts/figures` va cac file CSV/JSON cung cap so lieu nguon.

