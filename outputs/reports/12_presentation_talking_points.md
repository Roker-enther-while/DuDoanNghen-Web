# Presentation Talking Points

## Mở đầu (2 phút)
- Giới thiệu đề tài: Dự đoán nghẽn hệ thống web
- Vấn đề: Hệ thống web cần cảnh báo sớm
- Giải pháp: AI dựa trên chuỗi thời gian

## Pipeline dữ liệu (3 phút)
- Nguồn data: NASA HTTP 1995 (3.46M log lines)
- Xử lý: Parse → Aggregate → Normalize → Windows
- 19 features, proxy congestion score target
- Chronological split, không data leakage

## Mô hình đề xuất (5 phút)
- TCN-Attention-BiLSTM kết hợp 3 thành phần:
  - TCN: trích xuất đặc trưng thời gian
  - Attention: học trọng số quan trọng
  - BiLSTM: học phụ thuộc hai chiều
- Mixed precision training trên RTX 4060
- 120 epochs, best epoch 30

## Kết quả (5 phút)
- Regression: MAE 0.043, RMSE 0.056, R² 0.331
- Alert: Threshold p90 recall rất thấp (0.7%)
- Calibration: Threshold 0.05 → recall 97.9%
- Synthetic stress: periodic_spike F1 0.76, error_surge F1 0.14

## Demo (3 phút)
- Dashboard HTML minh bạch
- Recommendation Engine prototype
- Rule-based risk levels

## Giới hạn (2 phút)
- Proxy target, không phải measured congestion
- Chỉ có NASA data
- Không có CPU/RAM/response time
- Synthetic stress là controlled benchmark

## Hướng phát triển (2 phút)
- Thêm Zanbil → multi-source
- Thêm measured telemetry
- Real-time dashboard
- Production deployment

## Câu hỏi dự kiến
1. Tại sao R² chỉ 0.331? → Proxy target, limited data
2. Tại sao recall thấp? → Threshold p90 quá cao, calibration fix
3. Có thể deploy production? → Cần thêm data và real-time infra
4. So với ARIMA? → Chưa implement, planned
