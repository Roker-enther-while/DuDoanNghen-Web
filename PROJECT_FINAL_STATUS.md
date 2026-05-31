# PROJECT FINAL STATUS

## Dự án
Dự đoán nghẽn hệ thống web bằng mô hình TCN-Attention-BiLSTM dựa trên chuỗi thời gian.

## Trạng thái: READY FOR DEMO

## DONE

### Data Pipeline
- ✅ NASA HTTP 1995 → windows_fp16.npz
- ✅ 19 features, proxy congestion score
- ✅ Chronological split (70/15/15)
- ✅ MinMax normalization [0, 1]
- ✅ Float16 storage

### Models (8 models)
- ✅ Naive Last Value
- ✅ Moving Average
- ✅ LSTM
- ✅ GRU
- ✅ TCN
- ✅ Transformer
- ✅ TCN-LSTM
- ✅ TCN-Attention-BiLSTM (full 120 epochs)

### Training
- ✅ PyTorch CUDA + RTX 4060
- ✅ Mixed precision AMP
- ✅ Best epoch 30
- ✅ Best val RMSE 0.055146

### Evaluation (v2 — final verified)
- ✅ MAE: 0.043053
- ✅ RMSE: 0.056036
- ✅ R²: 0.339994
- ✅ Train time: 1092.5s
- ✅ Threshold calibration: F1 0.865596

### Synthetic Stress
- ✅ 6 scenarios, 1800 samples
- ✅ Best: periodic_spike (F1 0.757576)
- ✅ Worst: error_surge (F1 0.142857)

### System
- ✅ Dashboard HTML (Research Defense)
- ✅ Recommendation Engine (prototype)
- ✅ Test suite (83 tests)
- ✅ Source governance

### Figures
- ✅ prediction_vs_actual.png
- ✅ error_distribution.png
- ✅ model_comparison_rmse.png
- ✅ training_curves.png
- ✅ early_warning_timeline.png
- ✅ synthetic_stress_scenarios.png

### Demo
- ✅ Terminal demo (scripts/run_demo.py)
- ✅ Dashboard HTML
- ✅ JSON + MD reports

### Documentation
- ✅ README.md
- ✅ 5 chương báo cáo
- ✅ Presentation talking points
- ✅ Verification report
- ✅ Final acceptance checklist

## EVIDENCE

```
python -m pytest -q → 83 passed ✅
python scripts/generate_figures.py → 6 figures ✅
python scripts/run_demo.py --sample synthetic --explain --save-report → OK ✅
```

## RESULT SUMMARY

| Item | Value |
|---|---|
| Dataset | NASA HTTP 1995 |
| Model | TCN-Attention-BiLSTM |
| MAE | 0.043053 |
| RMSE | 0.056036 |
| R² | 0.339994 |
| Train time | 1092.5s |
| Calibrated F1 | 0.865596 |
| Best model | TCN-Attention-BiLSTM |
| Demo | Terminal + Dashboard HTML |
| Tests | 83 passed |
| Figures | 6 PNG |

## UPDATED STATUS

| Item | Previous | Current |
|---|---|---|
| Figures PNG | BLOCKED | ✅ FULL (6 figures) |
| Terminal demo | BLOCKED | ✅ FULL (working) |
| ARIMA baseline | PARTIAL | OPTIONAL (not blocking) |
| Zanbil raw | BLOCKED | BLOCKED (guide created) |
| Hook error | Unknown | NOT BLOCKING |

## STILL MISSING / BLOCKED

| Item | Status | Reason |
|---|---|---|
| Zanbil raw | BLOCKED | Pipeline đã chuẩn bị để mở rộng sang Zanbil, nhưng raw log chưa được cung cấp; kết quả hiện tại dựa trên NASA HTTP 1995 và synthetic stress benchmark. |
| ARIMA baseline | OPTIONAL | Naive/Moving Average đủ baseline |

## NEXT COMMANDS

```bash
# 1. Xem dashboard
start outputs/web/research_defense_dashboard.html

# 2. Xem figures
start outputs/figures/prediction_vs_actual.png
start outputs/figures/model_comparison_rmse.png

# 3. Chạy terminal demo
python scripts/run_demo.py --sample synthetic --explain --save-report

# 4. Chạy tests
python -m pytest -q

# 5. Xem kết quả
cat outputs/metrics/full_120_v2_tcn_attention_bilstm/final_metrics.json

# 6. Đặt Zanbil raw (khi có)
# data/raw/zanbil/access.log
# python scripts/check_zanbil_readiness.py
```
