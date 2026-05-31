# Chương 2: Cơ Sở Lý Thuyết

## 2.1 Hệ thống web và hiệu năng

### Các thành phần hệ thống web
- Web server (Apache, Nginx, IIS)
- Application server
- Database server
- Load balancer
- Cache layer

### Các chỉ số hiệu năng
- **Request count:** Số lượng request mỗi đơn vị thời gian
- **Response time:** Thời gian phản hồi
- **Throughput:** Lượng dữ liệu xử lý được
- **Error rate:** Tỷ lệ lỗi
- **CPU/Memory usage:** Sử dụng tài nguyên

## 2.2 Nghẽn hệ thống web

### Định nghĩa
Nghẽn (congestion) xảy ra khi hệ thống không thể xử lý kịp lưu lượng truy cập, dẫn đến:
- Tăng response time
- Tăng error rate
- Giảm throughput
- Rớt kết nối

### Nguyên nhân
- Flash crowd (lưu lượng đột biến)
- Resource exhaustion (hết tài nguyên)
- Application errors
- Database bottleneck
- Network issues

## 2.3 Chuỗi thời gian

### Định nghĩa
Chuỗi thời gian là tập hợp các quan sát được sắp xếp theo thứ tự thời gian.

### Đặc điểm
- **Trend:** Xu hướng dài hạn
- **Seasonality:** Tính chu kỳ
- **Noise:** Nhiễu ngẫu nhiên

### Ứng dụng trong dự đoán nghẽn
- Phân tích xu hướng lưu lượng
- Phát hiện pattern bất thường
- Dự đoán giá trị tương lai

## 2.4 Các mô hình deep learning

### LSTM (Long Short-Term Memory)
- Giải quyết vấn đề vanishing gradient
- Học phụ thuộc dài hạn
- Có gating mechanism

### GRU (Gated Recurrent Unit)
- Phiên bản đơn giản hóa của LSTM
- Ít tham số hơn
- Training nhanh hơn

### TCN (Temporal Convolutional Network)
- Sử dụng convolution 1D causal
- Dilated convolution để tăng receptive field
- Parallelizable hơn RNN

### Attention Mechanism
- Học trọng số cho từng time step
- Tập trung vào thông tin quan trọng
- Multi-head attention cho nhiều representation

### BiLSTM (Bidirectional LSTM)
- Học cả forward và backward
- Bắt context hai chiều
- Cải thiện representation

## 2.5 Mixed Precision Training

### FP16 vs FP32
- FP16: 16-bit floating point, nhanh hơn, ít bộ nhớ
- FP32: 32-bit floating point, chính xác hơn

### AMP (Automatic Mixed Precision)
- Tự động chọn FP16/FP32
- Loss scaling để tránh underflow
- Tăng tốc training trên GPU modern

## 2.6 Chuẩn hóa dữ liệu

### Min-Max Normalization
```
X_norm = (X - X_min) / (X_max - X_min)
```
- Đưa dữ liệu về khoảng [0, 1]
- Giữ nguyên phân phối
- Phù hợp cho neural network
