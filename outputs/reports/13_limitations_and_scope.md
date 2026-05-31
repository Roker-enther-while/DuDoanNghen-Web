# Limitations and Scope

## Scope

### In Scope
- Web congestion prediction từ log data
- Time series forecasting
- TCN-Attention-BiLSTM và baselines
- MAE, RMSE, R² evaluation
- Precision, Recall, F1 cho alert
- Threshold calibration
- Synthetic stress benchmark
- Recommendation Engine prototype
- Dashboard minh bạch

### Out of Scope
- Measured congestion (CPU/RAM/response time)
- Production auto-scaling
- Multi-source (chưa có Zanbil)
- Real-time system
- Cross-domain generalization

## Limitations

### Data Limitations
1. **NASA HTTP 1995 only** — data cũ, không representative cho web hiện đại
2. **No system telemetry** — không có CPU/RAM/response time
3. **Proxy target** — không phải ground truth congestion
4. **Limited time range** — chỉ 2 tháng

### Model Limitations
1. **R² = 0.331** — chỉ giải thích 33% phương sai
2. **Low recall at p90** — threshold gốc quá cao
3. **Error surge weakness** — F1 chỉ 0.14
4. **No ARIMA comparison** — baseline truyền thống chưa implement

### System Limitations
1. **Static dashboard** — không real-time
2. **Prototype recommendation** — rule-based, chưa production
3. **No terminal demo** — chỉ có HTML dashboard
4. **No deployment** — chỉ research prototype

## Honest Assessment

### What Works
- Pipeline dữ liệu chạy được
- Model train được
- Threshold calibration cải thiện recall
- Synthetic stress benchmark hữu ích
- Dashboard minh bạch

### What Doesn't Work
- Alert ở threshold gốc (recall 0.7%)
- Error surge detection
- Cross-source generalization
- Production deployment

## Future Work Required

### Immediate
1. Thêm Zanbil dataset
2. Data augmentation
3. Cải thiện error_surge

### Medium-term
1. Measured telemetry
2. ARIMA/Prophet baseline
3. Real-time dashboard

### Long-term
1. Production deployment
2. Cross-domain transfer
3. AutoML optimization
