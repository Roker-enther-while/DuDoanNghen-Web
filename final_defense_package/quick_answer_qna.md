# QUICK Q&A — 20 CÂU HỎI THƯỜNG GẶP

---

## CÂU 1: Vì sao chọn TCN-Attention-BiLSTM?

**Trả lời:** TCN trích xuất đặc trưng thời gian hiệu quả bằng dilated convolution. Attention giúp tập trung vào time step quan trọng. BiLSTM học phụ thuộc hai chiều. Kết hợp 3 thành phần giúp mô hình học được pattern phức tạp trong chuỗi thời gian.

---

## CÂU 2: TCN khác LSTM ở điểm nào?

**Trả lời:** TCN dùng convolution 1D causal, có thể parallelize nên training nhanh hơn. LSTM dùng recurrence, tuần tự từng step. TCN có receptive field cố định dựa trên dilation, LSTM có thể học dependency dài hạn tự nhiên. Trong đề tài này, nhóm kết hợp cả hai để tận dụng ưu điểm của mỗi mô hình.

---

## CÂU 3: Attention dùng để làm gì?

**Trả lời:** Attention học trọng số cho từng time step trong cửa sổ dữ liệu. Các time step quan trọng (có tín hiệu nghẽn) sẽ được chú trọng hơn. Điều này giúp mô hình tập trung vào thông tin liên quan thay vì đối xử đều tất cả.

---

## CÂU 4: Vì sao dùng NASA HTTP 1995?

**Trả lời:** Đây là dataset công khai, có sẵn, không vi phạm bản quyền. Phù hợp cho nghiên cứu ban đầu. Tuy nhiên, nhóm nhận thức đây là dữ liệu cũ và đã chuẩn bị pipeline để mở rộng sang Zanbil khi có raw log.

---

## CÂU 5: Dataset cũ có ảnh hưởng không?

**Trả lời:** Có. NASA 1995 không đại diện cho web hiện đại (HTTP/1.0, không HTTPS, không CDN). Kết quả chỉ mang tính minh chứng cho pipeline và mô hình, không khái quát hóa được cho production hiện đại.

---

## CÂU 6: Synthetic stress benchmark có phải dữ liệu thật không?

**Trả lời:** Không. Synthetic được tạo từ public baseline để kiểm thử có kiểm soát. Có 6 kịch bản: flash crowd, burst traffic, error surge, slow ramp, periodic spike, mixed incident. Chỉ dùng để đánh giá phản ứng model, không phải performance thật.

---

## CÂU 7: R² 0.34 có thấp không?

**Trả lời:** R²=0.34 nghĩa là mô hình giải thích được 34% phương sai. Đây là kết quả chấp nhận được cho bài toán dự đoán proxy congestion score từ dữ liệu cũ. Còn dư địa cải thiện lớn khi có dữ liệu tốt hơn.

---

## CÂU 8: F1 0.87 có ý nghĩa gì?

**Trả lời:** F1=0.87 là kết quả sau threshold calibration (giảm threshold từ 0.184 xuống 0.05). Recall tăng từ 0.7% lên 97.9%. Điều này cho thấy model học được tín hiệu tốt, nhưng cần calibration để phát huy.

---

## CÂU 9: Threshold 0.184 lấy từ đâu?

**Trả lời:** Threshold 0.183838 được lấy từ quantile 90% của validation set. Đây là ngưỡng phân biệt normal vs congestion. Tuy nhiên, ngưỡng này quá cao nên recall rất thấp, cần calibration.

---

## CÂU 10: Recommendation Engine hoạt động thế nào?

**Trả lời:** Engine nhận predicted score và threshold, phân loại risk level (Normal/Watch/Warning/Critical), đề xuất hành động (scale up, rate limit, enable cache). Đây là rule-based prototype, chưa phải production auto-scaling.

---

## CÂU 11: Nếu không có Zanbil thì đề tài có còn hợp lệ không?

**Trả lời:** Có. Đề tài vẫn hợp lệ với NASA HTTP 1995 + synthetic stress benchmark. Zanbil là mở rộng, không phải yêu cầu bắt buộc. Pipeline đã sẵn sàng, chỉ cần đặt raw log vào đúng vị trí.

---

## CÂU 12: ARIMA vì sao optional?

**Trả lời:** ARIMA là baseline truyền thống nhưng có hạn chế với dữ liệu phi tuyến tính. Nhóm đã có Naive, Moving Average làm baseline bắt buộc. ARIMA cần dependency statsmodels, có thể gây conflict. Nếu thời gian cho phép, nhóm sẽ bổ sung.

---

## CÂU 13: Mixed Precision giúp gì?

**Trả lời:** Mixed Precision FP16 tăng tốc training trên GPU modern (RTX 4060) và giảm bộ nhớ. AMP tự động chọn FP16/FP32 phù hợp. Kết quả: train time 18 phút cho 120 epochs.

---

## CÂU 14: Làm sao biết demo không phải hard-code?

**Trả lời:** Demo chạy từ script `run_demo.py`, đọc metrics từ file JSON thật, sinh dữ liệu synthetic ngẫu nhiên (seed=42 cho reproducible). Metrics trong JSON khớp với metrics trong report. Có test pytest verify.

---

## CÂU 15: Mô hình có triển khai thực tế được không?

**Trả lời:** Hiện tại là prototype nghiên cứu. Để triển khai thực tế cần: dữ liệu real-time, hệ thống monitoring, tích hợp auto-scaling, và validation trên production traffic.

---

## CÂU 16: Có phát hiện nghẽn real-time không?

**Trả lời:** Chưa. Demo hiện tại chạy offline trên batch data. Để real-time cần streaming pipeline và model serving infrastructure.

---

## CÂU 17: Khác gì monitoring ngưỡng truyền thống?

**Trả lời:** Monitoring ngưỡng truyền thống dùng threshold tĩnh, không học từ dữ liệu. Mô hình AI học pattern phức tạp, dự báo tương lai, và thích ứng với dữ liệu mới. Tuy nhiên, cần calibration để hoạt động tốt.

---

## CÂU 18: Hạn chế lớn nhất là gì?

**Trả lời:** Dữ liệu. NASA 1995 cũ, không có CPU/RAM/response time. Proxy target không phải measured congestion. Cần dữ liệu mới hơn và đa dạng hơn để đánh giá đầy đủ.

---

## CÂU 19: Nếu làm tiếp sẽ cải thiện gì?

**Trả lời:** 1) Thêm dataset mới (Zanbil, Calgary). 2) Multi-source validation. 3) Tối ưu hyperparameter. 4) Online monitoring. 5) Auto-scaling integration. 6) Explainability.

---

## CÂU 20: Đóng góp chính của nhóm là gì?

**Trả lời:** 1) Pipeline dữ liệu chuỗi thời gian cho web congestion. 2) Mô hình TCN-Attention-BiLSTM. 3) Hệ thống cảnh báo sớm với calibration. 4) Recommendation Engine. 5) Dashboard demo minh bạch với số liệu thật.

---

## GHI CHÚ

- Trả lời ngắn gọn, đúng trọng tâm
- Nếu không biết: "Em sẽ tìm hiểu thêm ạ"
- Nhìn thầy khi trả lời
- Không nói dài dòng
