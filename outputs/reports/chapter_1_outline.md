# Chương 1: Tổng Quan Đề Tài

## 1.1 Lý do chọn đề tài

Hệ thống web hiện đại đối mặt với thách thức lớn về hiệu năng và khả năng mở rộng. Khi lưu lượng truy cập tăng đột ngột hoặc hệ thống gặp lỗi, nguy cơ nghẽn có thể gây ra hậu quả nghiêm trọng như mất dữ liệu, giảm trải nghiệm người dùng, và thiệt hại kinh tế.

Việc dự đoán sớm nguy cơ nghẽn giúp:
- Chủ động ứng phó trước khi sự cố xảy ra
- Tối ưu hóa tài nguyên hệ thống
- Cải thiện trải nghiệm người dùng
- Giảm thiểu thiệt hại kinh doanh

## 1.2 Mục tiêu nghiên cứu

### Mục tiêu tổng quát
Xây dựng mô hình trí tuệ nhân tạo dựa trên chuỗi thời gian để dự đoán sớm nguy cơ nghẽn hệ thống web.

### Mục tiêu cụ thể
1. Xây dựng pipeline dữ liệu chuỗi thời gian từ log web
2. Thiết kế mô hình TCN-Attention-BiLSTM
3. Đánh giá mô hình trên dữ liệu công khai
4. Xây dựng hệ thống cảnh báo sớm
5. Đề xuất hành động phòng ngừa

## 1.3 Đối tượng và phạm vi

### Đối tượng nghiên cứu
- Dữ liệu log web (HTTP access logs)
- Chuỗi thời gian các chỉ số hiệu năng

### Phạm vi
- Dữ liệu: NASA HTTP 1995 (public dataset)
- Mô hình: TCN-Attention-BiLSTM và các baseline
- Đánh giá: MAE, RMSE, R², Precision, Recall, F1

## 1.4 Phương pháp nghiên cứu

1. **Phương pháp thực nghiệm:** Xây dựng và đánh giá mô hình trên dữ liệu thực
2. **Phương pháp so sánh:** So sánh TCN-Attention-BiLSTM với các baseline
3. **Phương pháp thống kê:** Phân tích kết quả bằng metrics chuẩn

## 1.5 Đóng góp dự kiến

1. Pipeline dữ liệu chuỗi thời gian cho web congestion
2. Mô hình TCN-Attention-BiLSTM kết hợp TCN, Attention, BiLSTM
3. Hệ thống cảnh báo sớm với threshold calibration
4. Recommendation Engine đề xuất hành động phòng ngừa
5. Đánh giá toàn diện trên synthetic stress benchmark
