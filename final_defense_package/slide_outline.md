# SLIDE OUTLINE

**Tổng số slide:** 12

---

## SLIDE 1: TIÊU ĐỀ

**Tiêu đề:** Dự đoán nghẽn hệ thống web bằng mô hình trí tuệ nhân tạo dựa trên chuỗi thời gian

**Nội dung:**
- Tên đề tài
- Nhóm sinh viên: Đinh Hữu Phong, Đặng Văn Tuyển, Nguyễn Đức Thịnh
- Giảng viên hướng dẫn: ThS. Nguyễn Ngọc Thận
- Trường Đại học Thủ Dầu Một
- Năm học 2025–2026

**Ghi chú:** Giới thiệu ngắn gọn, nhanh chóng chuyển sang slide tiếp.

---

## SLIDE 2: VẤN ĐỀ NGHIÊN CỨU

**Tiêu đề:** Vấn đề nghẽn hệ thống web

**Bullet:**
- Hệ thống web phục vụ hàng triệu người dùng đồng thời
- Lưu lượng đột biến → nguy cơ nghẽn → mất dữ liệu, thiệt hại
- Giám sát truyền thống: ngưỡng tĩnh, phản ứng muộn
- Cần giải pháp dự đoán sớm

**Hình gợi ý:** Hình minh họa hệ thống web congestion

**Ghi chú:** Nhấn mạnh hậu quả của nghẽn và hạn chế của phương pháp cũ.

---

## SLIDE 3: MỤC TIÊU ĐỀ TÀI

**Tiêu đề:** Mục tiêu nghiên cứu

**Bullet:**
- Xây dựng pipeline dữ liệu chuỗi thời gian từ log web
- Thiết kế mô hình TCN-Attention-BiLSTM
- Đánh giá với MAE, RMSE, R²
- Cảnh báo sớm với threshold calibration
- Recommendation Engine đề xuất hành động

**Ghi chú:** Đây là roadmap cho phần trình bày.

---

## SLIDE 4: DỮ LIỆU VÀ TELEMETRY

**Tiêu đề:** Dataset NASA HTTP 1995

**Bullet:**
- Nguồn: NASA Kennedy Space Center
- 3.46 triệu log lines (tháng 7-8/1995)
- 19 đặc trưng: request_count, bytes_sum, error_rate, ...
- Target: proxy congestion score
- Synthetic stress: 6 kịch bản, 1800 mẫu

**Hình gợi ý:** Bảng liệt kê features

**Ghi chú:** Giải thích tại sao dùng NASA HTTP và giới hạn của proxy target.

---

## SLIDE 5: PIPELINE XỬ LÝ DỮ LIỆU

**Tiêu đề:** Pipeline dữ liệu

**Bullet:**
- Parse HTTP logs → Aggregate 1 phút → Feature Engineering
- Normalize MinMax [0, 1] → Sliding Windows (60 steps)
- Split: Train 62,425 / Val 13,330 / Test 13,330
- Float16 storage, float32 training

**Hình gợi ý:** Sơ đồ pipeline flow

**Ghi chú:** Nhấn mạnh chronological split, không shuffle.

---

## SLIDE 6: KIẾN TRÚC TCN-ATTENTION-BILSTM

**Tiêu đề:** Mô hình đề xuất TCN-Attention-BiLSTM

**Bullet:**
- TCN: Trích xuất đặc trưng thời gian (dilated convolution)
- Attention: Học trọng số quan trọng (2 heads)
- BiLSTM: Phụ thuộc chuỗi hai chiều (32 units)
- Output: Congestion score prediction [0, 1]

**Hình gợi ý:** Sơ đồ kiến trúc model

**Ghi chú:** Giải thích tại sao kết hợp 3 thành phần.

---

## SLIDE 7: RECOMMENDATION ENGINE

**Tiêu đề:** Hệ thống khuyến nghị

**Bullet:**
- Risk levels: Normal → Watch → Warning → Critical
- Input: predicted score, threshold, phase
- Output: risk_level, recommended_actions, explanation
- Ví dụ: score 0.64 → CRITICAL → scale_up_cpu, rate_limit

**Hình gợi ý:** Bảng risk levels

**Ghi chú:** Đây là rule-based prototype, chưa phải production.

---

## SLIDE 8: KẾT QUẢ THỰC NGHIỆM

**Tiêu đề:** Kết quả đánh giá

**Bullet:**
- MAE: 0.043 | RMSE: 0.056 | R²: 0.34
- So sánh 8 mô hình: TCN-Attention-BiLSTM R² cao nhất
- Threshold calibration: F1 0.87, Recall 97.9%
- Train time: 18 phút trên RTX 4060

**Hình gợi ý:** Bảng so sánh models

**Ghi chú:** Giải thích R²=0.34 một cách trung thực.

---

## SLIDE 9: BIỂU ĐỒ ĐÁNH GIÁ

**Tiêu đề:** Phân tích kết quả

**Hình:**
- prediction_vs_actual.png
- model_comparison_rmse.png
- synthetic_stress_scenarios.png

**Bullet:**
- Model học được xu hướng chung
- RMSE thấp nhất trong deep learning models
- Phản ứng tốt với periodic spike, yếu với error surge

**Ghi chú:** Để hình tự nói, chỉ giải thích ngắn gọn.

---

## SLIDE 10: DEMO HỆ THỐNG

**Tiêu đề:** Demo hệ thống

**Nội dung:**
- Dashboard HTML: metrics, confusion matrix, calibration
- Terminal demo: prediction → risk → recommendation
- Figures: 6 biểu đồ đánh giá

**Ghi chú:** Chuyển sang demo thực tế tại đây.

---

## SLIDE 11: HẠN CHẾ

**Tiêu đề:** Giới hạn và hạn chế

**Bullet:**
- NASA 1995: dữ liệu cũ, không có CPU/RAM/response time
- Proxy target: không phải measured congestion
- Synthetic: controlled benchmark, không phải real-world
- Zanbil: pipeline đã chuẩn bị, raw log chưa có
- R²=0.34: còn dư địa cải thiện

**Ghi chú:** Trung thực, không giấu hạn chế.

---

## SLIDE 12: KẾT LUẬN

**Tiêu đề:** Kết luận và hướng phát triển

**Bullet:**
- Đã hoàn thành: Pipeline, Model, Evaluation, Demo, Recommendation
- Metrics: MAE 0.043, RMSE 0.056, R² 0.34, Calibrated F1 0.87
- Hướng phát triển: Multi-source, Online monitoring, Auto-scaling

**Ghi chú:** Cảm ơn và sẵn sàng trả lời câu hỏi.

---

## THIẾT KẾ GỢI Ý

- Font: Sans-serif (Arial, Helvetica)
- Màu chủ đạo: Xanh dương (#2563EB)
- Nền: Trắng hoặc xám nhạt
- Hình: Đặt bên phải hoặc dưới bullet
- Không quá nhiều text trên mỗi slide
