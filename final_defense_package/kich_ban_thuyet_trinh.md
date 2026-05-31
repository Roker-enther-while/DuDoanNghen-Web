# KỊCH BẢN THUYẾT TRÌNH

**Đề tài:** Dự đoán nghẽn hệ thống web bằng mô hình trí tuệ nhân tạo dựa trên chuỗi thời gian

**Thời lượng:** 7–10 phút

**Nhóm:** Đinh Hữu Phong, Đặng Văn Tuyển, Nguyễn Đức Thịnh

---

## PHẦN 1: MỞ ĐẦU (1 phút)

### Lời nói:

"Kính chào thầy và các bạn. Nhóm em xin trình bày đề tài: Dự đoán nghẽn hệ thống web bằng mô hình trí tuệ nhân tạo dựa trên chuỗi thời gian.

Nhóm gồm 3 thành viên: Đinh Hữu Phong, Đặng Văn Tuyển và Nguyễn Đức Thịnh, dưới sự hướng dẫn của ThS. Nguyễn Ngọc Thận."

---

## PHẦN 2: VẤN ĐỀ NGHIÊN CỨU (1 phút)

### Lời nói:

"Hệ thống web hiện đại phục vụ hàng triệu người dùng đồng thời. Khi lưu lượng truy cập tăng đột ngột, nguy cơ nghẽn có thể gây mất dữ liệu và thiệt hại kinh tế.

Các phương pháp giám sát truyền thống dựa trên ngưỡng tĩnh có hạn chế: không phát hiện được pattern phức tạp và thường phản ứng quá muộn.

Vì vậy, nhóm em nghiên cứu sử dụng trí tuệ nhân tạo để dự đoán sớm nguy cơ nghẽn, giúp chủ động ứng phó trước khi sự cố xảy ra."

### Slide:
- Vấn đề: Nghẽn hệ thống web
- Hạn chế: Giám sát truyền thống
- Giải pháp: AI dự đoán sớm

---

## PHẦN 3: DATASET VÀ PIPELINE (1.5 phút)

### Lời nói:

"Nhóm sử dụng dataset NASA HTTP 1995 từ Internet Traffic Archive, bao gồm 3.46 triệu log lines từ web server của NASA.

Dữ liệu được xử lý qua pipeline:
1. Parse HTTP logs
2. Tổng hợp theo cửa sổ 1 phút
3. Tạo 19 đặc trưng như request count, bytes, error rate
4. Chuẩn hóa MinMax về khoảng [0, 1]
5. Tạo sliding windows 60 time steps
6. Chia train/val/test theo thứ tự thời gian

Kết quả: 62,425 mẫu train, 13,330 mẫu validation, 13,330 mẫu test."

### Slide:
- Dataset: NASA HTTP 1995
- 3.46 triệu log lines
- 19 đặc trưng
- 62,425 train windows

### Hình:
- prediction_vs_actual.png

---

## PHẦN 4: MÔ HÌNH TCN-ATTENTION-BILSTM (2 phút)

### Lời nói:

"Mô hình chính là TCN-Attention-BiLSTM, kết hợp 3 thành phần:

Thứ nhất, TCN (Temporal Convolutional Network) trích xuất đặc trưng thời gian bằng convolution có dilation, giúp học pattern cục bộ hiệu quả.

Thứ hai, Attention Mechanism học trọng số cho từng time step, giúp tập trung vào thông tin quan trọng.

Thứ ba, BiLSTM (Bidirectional LSTM) học phụ thuộc chuỗi hai chiều, bắt được context cả trước và sau.

Mô hình được train 120 epochs trên GPU RTX 4060 với mixed precision FP16, thời gian train khoảng 18 phút."

### Slide:
- TCN: Đặc trưng thời gian
- Attention: Trọng số quan trọng
- BiLSTM: Phụ thuộc hai chiều
- 120 epochs, RTX 4060

### Hình:
- training_curves.png

---

## PHẦN 5: KẾT QUẢ THỰC NGHIỆM (1.5 phút)

### Lời nói:

"Kết quả đánh giá trên test set:
- MAE: 0.043
- RMSE: 0.056
- R²: 0.34 — mô hình giải thích được 34% phương sai

So sánh với 7 mô hình baseline, TCN-Attention-BiLSTM có R² cao nhất.

Về cảnh báo: threshold gốc 0.184 có recall rất thấp chỉ 0.7%. Sau khi calibration giảm xuống 0.05, recall tăng lên 97.9% với F1 đạt 0.87.

Với synthetic stress benchmark gồm 6 kịch bản, mô hình phản ứng tốt nhất với periodic spike (F1=0.76) và yếu nhất với error surge (F1=0.14)."

### Slide:
- MAE: 0.043 | RMSE: 0.056 | R²: 0.34
- Calibrated F1: 0.87 | Recall: 97.9%
- Best scenario: periodic_spike
- Worst scenario: error_surge

### Hình:
- model_comparison_rmse.png
- synthetic_stress_scenarios.png

---

## PHẦN 6: DEMO (2 phút)

### Chuyển cảnh:

"Giờ em xin demo hệ thống. Trước tiên, em mở dashboard HTML..."

### Lời nói:

"Đây là dashboard Research Defense, hiển thị đầy đủ:
- Metrics chính với số liệu thật
- Confusion matrix
- Threshold calibration
- Synthetic stress benchmark
- Recommendation Engine

Tiếp theo, em chạy terminal demo..."

### Chạy demo:

```
python scripts/run_demo.py --sample synthetic --model best --explain --save-report
```

### Giải thích:

"Demo mô phỏng luồng hoạt động:
1. Đọc dữ liệu telemetry
2. Model dự đoán congestion score
3. Tính risk level
4. Đề xuất hành động: scale up, rate limit, enable cache

Với dữ liệu synthetic ngẫu nhiên, model dự đoán score 0.64, vượt threshold 0.05, nên risk level là CRITICAL."

### Hình:
- research_defense_dashboard.html

---

## PHẦN 7: HẠN CHẾ (1 phút)

### Lời nói:

"Nhóm em nhận thức rõ các hạn chế:

Thứ nhất, NASA HTTP 1995 là dữ liệu cũ, không có CPU/RAM/response time. Target là proxy congestion score, không phải measured congestion thật.

Thứ hai, synthetic stress benchmark chỉ dùng để kiểm thử có kiểm soát, không phải dữ liệu real-world.

Thứ ba, pipeline đã chuẩn bị cho Zanbil dataset nhưng raw log chưa được cung cấp, nên chưa có multi-source validation.

Thứ tư, R²=0.34 cho thấy mô hình còn dư địa cải thiện lớn."

### Slide:
- NASA 1995: dữ liệu cũ
- Proxy target: không phải measured congestion
- Synthetic: không phải real-world
- Zanbil: chưa có raw log

---

## PHẦN 8: KẾT LUẬN (1 phút)

### Lời nói:

"Tóm lại, nhóm em đã:

1. Xây dựng pipeline dữ liệu chuỗi thời gian từ log web
2. Thiết kế mô hình TCN-Attention-BiLSTM kết hợp TCN, Attention, BiLSTM
3. Đánh giá với MAE=0.043, RMSE=0.056, R²=0.34
4. Xây dựng hệ thống cảnh báo sớm với calibrated F1=0.87
5. Tạo dashboard demo minh bạch

Hướng phát triển: bổ sung dataset mới hơn, multi-source với Zanbil, tối ưu hyperparameter, và triển khai online monitoring.

Em xin cảm ơn thầy và các bạn. Nhóm em sẵn sàng trả lời câu hỏi."

### Slide:
- Đã hoàn thành: Pipeline, Model, Evaluation, Demo, Recommendation
- Hướng phát triển: Multi-source, Online monitoring, Auto-scaling

---

## GHI CHÚ

- Thời gian mỗi phần có thể điều chỉnh ±30 giây
- Nếu demo lỗi, dùng ảnh/screenshots đã chụp sẵn
- Nói chậm, rõ ràng, nhìn thầy khi trả lời câu hỏi
- Chuẩn bị trước câu trả lời cho 20 câu hỏi thường gặp
