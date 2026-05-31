# Final Project Summary

## Dự án
Dự đoán nghẽn hệ thống web bằng mô hình TCN-Attention-BiLSTM dựa trên chuỗi thời gian.

## Kết quả chính

| Metric | Giá trị |
|---|---|
| MAE | 0.042792 |
| RMSE | 0.056399 |
| R² | 0.331430 |
| Calibrated F1 | 0.865596 |
| Calibrated Recall | 0.979049 |

## Mô hình tốt nhất
- **Model:** TCN-Attention-BiLSTM
- **Epochs:** 120 (best epoch 30)
- **Backend:** PyTorch CUDA + RTX 4060

## Hệ thống đã xây dựng
1. ✅ Data pipeline (NASA HTTP → windows)
2. ✅ 8 models (baseline + deep learning)
3. ✅ Training framework
4. ✅ Evaluation & comparison
5. ✅ Threshold calibration
6. ✅ Synthetic stress benchmark
7. ✅ Dashboard HTML
8. ✅ Recommendation Engine (prototype)
9. ✅ Test suite (83 tests)

## Giới hạn
- NASA target là proxy, không phải measured congestion
- Không có CPU/RAM/response time
- Zanbil raw missing → no multi-source
- Synthetic stress là controlled benchmark

## Hướng phát triển
1. Thêm Zanbil → multi-source
2. Thêm measured telemetry
3. Real-time dashboard
4. Production Recommendation Engine
