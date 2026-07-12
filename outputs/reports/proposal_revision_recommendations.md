# Proposal Revision Recommendations — Sau Phản Biện

## 1. Những mục giữ nguyên

- Hướng đề tài: dự đoán nghẽn hệ thống web bằng AI chuỗi thời gian
- Mô hình TCN-Attention-BiLSTM
- Mixed Precision Float16 + chuẩn hóa 0-1
- Synthetic scenarios để kiểm thử có kiểm soát
- Dashboard prototype demo cảnh báo
- Source governance/license manifest
- So sánh với baseline (Moving Average, LSTM, GRU, TCN, Transformer)

## 2. Những mục phải sửa

### 2.1 Measured congestion → Proxy congestion risk

**Lý do:** NASA HTTP log không có CPU, RAM, response time. Target hiện tại là proxy congestion score được xây dựng từ traffic features.

**Sửa thành:** "Dự đoán nguy cơ nghẽn dựa trên proxy congestion score từ đặc trưng log web (request count, bytes, error rate, spike score), không phải đo lường nghẽn thật từ system telemetry."

### 2.2 TurboQuant → Planned/Optional

**Lý do:** Không có code hay implementation TurboQuant.

**Sửa thành:** "Tối ưu bằng Mixed Precision Float16 + chuẩn hóa 0-1. TurboQuant là planned extension nếu có bằng chứng implementation."

### 2.3 Recommendation Engine → Rule-based prototype

**Lý do:** Không có production recommendation engine.

**Sửa thành:** "Prototype cảnh báo rule-based dựa trên risk score và threshold calibration, không phải production auto-scaling system."

### 2.4 Model comparison → Chỉ dùng verified results

**Lý do:** ARIMA/Holt-Winters chưa train. Không dùng số demo giả.

**Sửa thành:** "So sánh các mô hình đã train trên cùng split: Moving Average, LSTM, GRU, TCN, Transformer, TCN-Attention-BiLSTM. ARIMA/Holt-Winters là planned extension."

### 2.5 Synthetic → Controlled benchmark

**Lý do:** Synthetic không phải real-world.

**Sửa thành:** "Kiểm thử bằng synthetic stress benchmark (6 scenarios, 1800 samples) là controlled simulation, không phải real-world performance."

### 2.6 Multi-source → Pending Zanbil

**Lý do:** Zanbil raw chưa có.

**Sửa thành:** "Multi-source (NASA + Zanbil) là mục tiêu mở rộng, cần Zanbil raw hợp lệ tại data/raw/zanbil/access.log."

## 3. Mục tiêu đề cương viết lại

### Mục tiêu tổng quát (viết lại)

Nghiên cứu và xây dựng pipeline AI minh bạch dựa trên chuỗi thời gian từ dữ liệu log web công khai để dự đoán sớm nguy cơ nghẽn hệ thống web. Mô hình TCN-Attention-BiLSTM được đánh giá trên proxy congestion score từ NASA HTTP logs, với threshold calibration và synthetic stress benchmark tách riêng.

### Mục tiêu cụ thể (viết lại)

1. Xây dựng data pipeline từ NASA HTTP 1995 logs thành chuỗi thời gian với proxy congestion target.
2. Chuẩn hóa dữ liệu 0-1, lưu trữ float16, training float32 với Mixed Precision.
3. Huấn luyện và so sánh 6 mô hình: Moving Average, LSTM, GRU, TCN, Transformer, TCN-Attention-BiLSTM.
4. Đánh giá bằng MAE, RMSE, R² cho regression; Precision, Recall, F1, confusion matrix cho alert.
5. Calibrate threshold để cải thiện recall cho cảnh báo.
6. Kiểm thử bằng synthetic stress benchmark (6 scenarios) tách riêng khỏi real public result.
7. Xây dựng dashboard/report minh bạch dùng số liệu thật.
8. Đặt governance cho nguồn dữ liệu và ghi rõ giới hạn.

## 4. Câu kết luận trung thực

Mô hình TCN-Attention-BiLSTM học được proxy congestion score từ NASA HTTP logs với MAE 0.043, RMSE 0.056, R² 0.331. Cảnh báo ở threshold p90 gốc có recall rất thấp (0.007); threshold calibration giảm xuống 0.05 để đạt recall 0.979 nhưng phải giải thích tách biệt. Synthetic stress benchmark cho thấy model phản ứng tốt với periodic spike (F1 0.758) nhưng yếu với error surge (F1 0.143). Kết quả không phải measured congestion, không có cross-source claim vì thiếu Zanbil raw. Cần thêm system telemetry và multi-source data để mở rộng real-world applicability.
