# Chương 5: Kết Luận Và Hướng Phát Triển

## 5.1 Kết quả đạt được

### Pipeline dữ liệu
- ✅ Xây dựng pipeline từ NASA HTTP logs đến model training
- ✅ Xử lý 3.46 triệu raw log lines
- ✅ Tạo 89,085 sliding windows
- ✅ Chuẩn hóa MinMax [0, 1]
- ✅ Float16 storage

### Mô hình
- ✅ Implement 8 mô hình (baseline + deep learning)
- ✅ TCN-Attention-BiLSTM train full 120 epoch
- ✅ Best epoch 30, best val RMSE 0.055146

### Đánh giá
- ✅ MAE: 0.042792
- ✅ RMSE: 0.056399
- ✅ R²: 0.331430
- ✅ Threshold calibration: F1 0.865596, Recall 0.979049

### Synthetic stress benchmark
- ✅ 6 scenarios, 1800 samples
- ✅ Best scenario: periodic_spike (F1 0.757576)
- ✅ Worst scenario: error_surge (F1 0.142857)

### Hệ thống
- ✅ Dashboard HTML minh bạch
- ✅ Recommendation Engine prototype
- ✅ Source governance/license manifest
- ✅ Test suite (83 tests pass)

## 5.2 Hạn chế

### Dữ liệu
- Chỉ có NASA HTTP 1995 (cũ)
- Không có CPU/RAM/response time telemetry
- Proxy target, không phải measured congestion
- Thiếu Zanbil → không có multi-source

### Mô hình
- R² = 0.331 (giải thích 33% phương sai)
- Alert recall thấp ở threshold gốc
- Chưa có ARIMA baseline

### Hệ thống
- Recommendation Engine là prototype
- Dashboard tĩnh (không real-time)
- Chưa có terminal demo

## 5.3 Hướng phát triển

### Ngắn hạn
1. Thêm Zanbil dataset → multi-source
2. Thêm Calgary/ClarkNet/WorldCup datasets
3. Data augmentation (time warping, SMOTE)
4. Cải thiện error_surge detection

### Trung hạn
1. Thêm measured telemetry (CPU/RAM/response time)
2. Implement ARIMA/Prophet baseline
3. Real-time dashboard
4. Production Recommendation Engine

### Dài hạn
1. Cross-domain generalization
2. Transfer learning
3. AutoML cho hyperparameter tuning
4. Deploy production

## 5.4 Kết luận

Mô hình TCN-Attention-BiLSTM đã học được proxy congestion score từ NASA HTTP logs với kết quả khả quan. Threshold calibration cải thiện đáng kể hiệu suất cảnh báo. Synthetic stress benchmark cho thấy model phản ứng tốt với periodic spike nhưng yếu với error surge.

Để ứng dụng thực tế, cần:
1. Thêm dữ liệu đa nguồn
2. Có measured telemetry
3. Cải thiện specificity
4. Deploy production-ready system

Dự án đã xây dựng được pipeline hoàn chỉnh, minh bạch, có thể mở rộng khi có thêm dữ liệu.
