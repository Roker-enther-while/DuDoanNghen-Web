# Research Integrity Audit Report

**Date**: 2026-07-11
**Auditor**: Automated (MiMoCode agent)
**Scope**: Full project artifact verification

## 1. Executive Summary

The project has a complete ML pipeline (data ingestion, training, evaluation, dashboard) but several **critical integrity gaps** were identified:

1. **No reproducible artifacts on disk**: The processed data, model checkpoints, and prediction CSVs referenced by metric JSONs do not exist.
2. **Three inconsistent metric sets** for the proposed model, none traceable to artifacts on disk.
3. **Proxy congestion score validation failed**: The proxy score is negatively correlated with real congestion indicators — it measures load, not congestion.
4. **No ablation study**: Component contributions were never evaluated.
5. **No rolling-origin validation**: Results rely on a single train/val/test split.

These gaps do not indicate fraud — the pipeline code is sound and the training infrastructure works. But the reported numbers cannot be independently verified from artifacts on disk, and the scientific claims require significant revision.

## 2. Artifact Inventory

### 2.1 Data Artifacts

| Artifact | Expected Path | Exists? | Status |
|---|---|---|---|
| Raw NASA Jul95 | `data/raw/nasa_http/NASA_access_log_Jul95.gz` | NO | Must re-download |
| Raw NASA Aug95 | `data/raw/nasa_http/NASA_access_log_Aug95.gz` | NO | Must re-download |
| Windows FP16 | `data/processed/nasa_http_3m/windows/windows_fp16.npz` | NO | Must re-run pipeline |
| Testbed telemetry | `data/testbed/longrun_20260517_211328/testbed_labeled.csv` | YES | 1802 rows |

### 2.2 Model Artifacts

| Model | Expected Path | Exists? |
|---|---|---|
| TCN-Attention-BiLSTM (full_120) | `outputs/models/full_120_tcn_attention_bilstm/best_model.pt` | NO |
| TCN-Attention-BiLSTM (v2) | `outputs/models/full_120_v2_tcn_attention_bilstm/best_model.pt` | NO |
| LSTM | `outputs/models/lstm/model.pt` | NO |
| GRU | `outputs/models/gru/model.pt` | NO |
| TCN | `outputs/models/tcn/model.pt` | NO |
| Transformer | `outputs/models/transformer/model.pt` | NO |
| TCN-LSTM | `outputs/models/tcn_lstm/model.pt` | NO |
| Baselines | `outputs/models/naive_last_value/baseline.json` | NO |

### 2.3 Prediction Artifacts

| Model | Expected Path | Exists? |
|---|---|---|
| All models | `outputs/predictions/*.csv` | NO |

### 2.4 Metric Artifacts

| File | Exists? | Content |
|---|---|---|
| `full_120/final_metrics.json` | YES | MAE=0.046073, R²=0.257794, train_time=1.5µs |
| `full_120_v2/final_metrics.json` | YES | MAE=0.043053, R²=0.339994, train_time=1092s |
| `final_experiment_summary.json` | YES | MAE=0.042792, R²=0.331430 |
| `balanced_model_comparison.json` | YES | Proposed MAE=0.043916, R²=0.270956 |

## 3. Metric Inconsistency Analysis

The proposed model (TCN-Attention-BiLSTM) has four different reported metric sets:

| Source | MAE | RMSE | R² | train_time | Notes |
|---|---|---|---|---|---|
| full_120 (v1) | 0.046073 | 0.059423 | 0.257794 | 1.5µs | **Did not actually train** |
| full_120_v2 | 0.043053 | 0.056036 | 0.339994 | 1092s | Actually trained 120 epochs |
| final_experiment_summary | 0.042792 | 0.056399 | 0.331430 | Unknown | Source unclear |
| balanced_comparison | 0.043916 | 0.058894 | 0.270956 | 50s | 20 epochs only |

**Critical finding**: The v1 `full_120` has `train_time_seconds: 1.5e-06` (1.5 microseconds). This is physically impossible for 120 epochs of training on 62K samples. The model likely loaded a pre-existing checkpoint and only ran inference, or the training was interrupted and the metrics reflect an incomplete/untrained state.

The numbers used in NEXT_STEP.md (MAE: 0.042792, R²: 0.331430) match `final_experiment_summary.json` but this file's provenance is unclear — it doesn't correspond to either v1 or v2 training runs.

## 4. Proxy Target Validation

### 4.1 Methodology
The proxy congestion score was computed on real testbed telemetry (Flask app + Prometheus + Locust, 1802 samples across 5 load profiles) and correlated with independently measured system metrics.

### 4.2 Results

| Metric | Pearson r | Direction | Interpretation |
|---|---|---|---|
| Response Time | -0.544 | NEGATIVE | High proxy → low latency (WRONG direction) |
| Error Rate | -0.544 | NEGATIVE | High proxy → few errors (WRONG direction) |
| CPU Usage | -0.299 | NEGATIVE | Weak inverse relationship |
| Request Rate | +0.573 | POSITIVE | High proxy → high traffic (expected) |
| Throughput | +0.579 | POSITIVE | High proxy → high throughput (expected) |

### 4.3 Conclusion
The proxy score measures **load intensity**, not **congestion**. When the system is under high but manageable load, the proxy score is high but the system is not congested (low latency, few errors). Congestion occurs during overload transitions that the proxy score does not capture.

## 5. Gaps Between Claims and Reality

| Claim in README/CLAUDE.md | Reality |
|---|---|
| "MAE: 0.042792, RMSE: 0.056399, R²: 0.331430" | Numbers exist in JSON but no artifacts to verify |
| "86 tests pass" | Tests verify code logic, not scientific claims |
| "Dashboard with real numbers" | Dashboard HTML exists with numbers from JSON files |
| "Threshold calibration: F1 0.866" | Calibration code exists but no prediction CSVs to verify |
| "Synthetic stress: 6 scenarios" | Synthetic data generation code exists |

## 6. Recommendations

### Immediate (P0)
1. **Re-run data pipeline** to regenerate `data/processed/nasa_http_3m/windows/windows_fp16.npz`
2. **Re-train the proposed model** from scratch to produce fresh checkpoints and predictions
3. **Fix or remove v1 metrics** (the 1.5µs training run is clearly invalid)
4. **Update all reports** to use a single, consistent set of metrics from a verified training run

### Scientific (P0)
5. **Acknowledge proxy target limitation** in all reports and the dashboard
6. **Run ablation study** to determine component contributions
7. **Run rolling-origin CV** to get robust performance estimates with variance

### Documentation (P1)
8. **Update limitations section** (see `docs/LIMITATIONS_HONEST.md`)
9. **Update CLAUDE.md** to reflect current state honestly
10. **Write final integrity report** (see `docs/RESEARCH_INTEGRITY_UPDATE.md`)

## 7. Files Modified in This Audit

| File | Action |
|---|---|
| `docs/PROXY_TARGET_DEFINITION.md` | Created — formal proxy formula definition |
| `docs/PROXY_VALIDATION_REPORT.md` | Created — correlation with real telemetry |
| `docs/LIMITATIONS_HONEST.md` | Created — transparent limitations section |
| `docs/RESEARCH_INTEGRITY_UPDATE.md` | Created — before/after comparison |
| `scripts/validate_proxy_target.py` | Created — proxy validation script |
| `scripts/run_ablation_study.py` | Created — ablation study runner |
| `scripts/run_rolling_cv.py` | Created — rolling-origin CV runner |
| `src/training/torch_models.py` | Modified — added ablation variants |
| `src/training/registry.py` | Modified — registered ablation models |
| `configs/training/ablation/*.yaml` | Created — ablation training configs |
