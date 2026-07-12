# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Web congestion prediction system using a TCN-Attention-BiLSTM model on NASA HTTP 1995 log data. The project is a research/thesis deliverable (Vietnamese academic context) that includes a full ML pipeline: data ingestion, time-series feature engineering, model training with baseline comparisons, threshold calibration, synthetic stress benchmarking, and an early-warning demo dashboard.

**Primary model:** TCN-Attention-BiLSTM (Temporal Convolutional Network + Self-Attention + Bidirectional LSTM)
**Baselines:** Naive Last Value, Moving Average, LSTM, GRU, TCN, Transformer, TCN-LSTM
**Target:** `proxy_congestion_score` — a synthetic composite score derived from NASA HTTP log features, NOT a measured congestion label. This distinction must be maintained in all outputs and reports.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest -q

# Run a single test file
python -m pytest tests/test_nasa_http_parser.py -v

# Data pipeline (smoke — uses generated sample data)
python scripts/run_data_pipeline.py --config configs/data/nasa_http_smoke.yaml

# Data pipeline (full NASA HTTP)
python scripts/run_data_pipeline.py --config configs/data/nasa_http.yaml

# Train a specific model (smoke)
python scripts/train_model.py --model tcn_attention_bilstm --config configs/training/smoke.yaml

# Train TCN-Attention-BiLSTM full 120 epochs (requires CUDA GPU)
python scripts/train_model.py --model tcn_attention_bilstm --config configs/training/tcn_attention_bilstm_full_120.yaml

# List available models
python scripts/train_model.py --model x --list-models

# Evaluate model predictions
python scripts/evaluate_model.py --predictions outputs/predictions/full_120_tcn_attention_bilstm/test_predictions.csv --threshold 0.183838

# Calibrate alert threshold
python scripts/calibrate_alert_threshold.py --predictions outputs/predictions/full_120_tcn_attention_bilstm/test_predictions.csv

# Run synthetic stress benchmark
python scripts/evaluate_synthetic_stress.py --data data/processed/synthetic_stress/windows/windows_fp16.npz --labels data/processed/synthetic_stress/labels/synthetic_stress_labels.csv --model-path outputs/models/full_120_tcn_attention_bilstm/best_model.pt --output-dir outputs/synthetic_stress_eval/full_120_tcn_attention_bilstm

# Run early-warning demo
python scripts/run_demo.py --sample synthetic --explain --save-report

# Open dashboard
start outputs/web/research_defense_dashboard.html
```

## Architecture

### Data Pipeline (`src/data/` + `scripts/run_data_pipeline.py`)

Raw NASA HTTP access logs (gzip) are parsed into request-level records (`nasa_http.py`), aggregated into 1-minute time-series windows with 19 features (request_count, bytes_sum, error_rate, unique_hosts, etc.), then a **proxy congestion score** is computed as a weighted composite of expanding min-max normalized features. The target is shifted by `horizon_steps` (default 15 minutes) to create `target_next_congestion_score`. Data is split chronologically into train/val/test, normalized per-split, stored as float16 NPZ sliding windows (`windowing.py`). Config is YAML-driven (`configs/data/`).

### Model Registry (`src/training/registry.py`)

All models are registered in `MODEL_REGISTRY` with metadata (name, category, module path, recommended config). `get_model_builder()` dynamically imports the `build_model` function from the model's module. Categories: `baseline`, `rnn`, `convolutional`, `attention`, `hybrid`, `proposed`.

### Dual Backend Training (`src/training/trainer.py` + `torch_trainer.py`)

The trainer dispatches based on `backend` in config:
- **`torch`** (primary for production training): Uses `src/training/torch_models.py` PyTorch implementations with CUDA, mixed precision, gradient clipping, early stopping, checkpointing, and resume support.
- **`tensorflow`**: Uses Keras models defined in `src/models/*.py` (e.g., `tcn_attention_bilstm.py`). Used for the original model definition but PyTorch is the active backend.

Training outputs: `outputs/models/{name}/best_model.pt`, `outputs/predictions/{name}/test_predictions.csv`, `outputs/metrics/{name}/final_metrics.json`.

### Services Layer (`src/services/`)

- `anomaly_detector.py`: Hybrid anomaly detection (Isolation Forest + statistical deviation)
- `recommendation_engine.py`: Rule-based alert and recommendation engine (Normal/Warning/Critical)
- `decision_engine.py`, `infer_service.py`, `monitor_service.py`: Supporting inference and monitoring services

### Testbed (`testbed/`)

A Flask app (`testbed/webapp/app.py`) with Prometheus metrics that simulates a web service under load. Used with Locust (`testbed/load/locustfile.py`) to generate real telemetry for validation against the model.

## Key Constraints

- **Proxy target honesty:** The NASA HTTP target is a proxy congestion score. Never call it "measured congestion" or "real congestion label." All reports/dashboards must state this.
- **Synthetic separation:** Synthetic stress benchmark results must be kept separate from real public data results.
- **No data fabrication:** All metrics must come from actual training artifacts. No fake checkpoints, no fake numbers, no claiming multi-source when only NASA data is used.
- **Float16 storage, float32 training:** Data is stored as float16 for efficiency, upcast to float32 during training via `convert_to_train_dtype()`.
- **Chronological splits only:** Train/val/test splits are strictly time-ordered to prevent data leakage.
- **JSON outputs must not contain NaN/Infinity:** All JSON is written with `allow_nan=False`.

## Project State Files

- `NEXT_STEP.md`: Current task queue — read before starting work
- `AGENT_REPORT.md`: Session work log — update at end of each session
- `PHASE_LOG.md`: Phase completion tracking — update when phases complete
- `outputs/metrics/`: All experiment metrics as JSON
- `outputs/web/research_defense_dashboard.html`: Main defense/presentation dashboard
