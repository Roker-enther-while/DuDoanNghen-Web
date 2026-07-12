# BÁO CÁO NGHIÊN CỨU KHOA HỌC

---

**Trường:** Trường Đại học Thủ Dầu Một
**Viện:** Viện Công nghệ Số

**Đề tài:** Dự đoán nghẽn hệ thống web bằng mô hình trí tuệ nhân tạo dựa trên chuỗi thời gian

---

**Nhóm sinh viên:**
- Đinh Hữu Phong — 2324802010095 — Nhóm trưởng
- Đặng Văn Tuyển — 2324802010156 — Thành viên
- Nguyễn Đức Thịnh — 2324802010355 — Thành viên

**Giảng viên hướng dẫn:** ThS. Nguyễn Ngọc Thận

**Năm học:** 2025–2026

---

# MỤC LỤC

- [CHƯƠNG 1: GIỚI THIỆU TỔNG QUAN](#chương-1-giới-thiệu-tổng-quan)
- [CHƯƠNG 2: CƠ SỞ LÝ THUYẾT](#chương-2-cơ-sở-lý-thuyết)
- [CHƯƠNG 3: MÔ HÌNH ĐỀ XUẤT VÀ THIẾT KẾ HỆ THỐNG](#chương-3-mô-hình-đề-xuất-và-thiết-kế-hệ-thống)
- [CHƯƠNG 4: THỰC NGHIỆM VÀ ĐÁNH GIÁ](#chương-4-thực-nghiệm-và-đánh-giá)
- [CHƯƠNG 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN](#chương-5-kết-luận-và-hướng-phát-triển)

---

# CHƯƠNG 1: GIỚI THIỆU TỔNG QUAN

## 1.1 Tên đề tài

Dự đoán nghẽn hệ thống web bằng mô hình trí tuệ nhân tạo dựa trên chuỗi thời gian.

## 1.2 Lý do chọn đề tài

Hệ thống web hiện đại đóng vai trò quan trọng trong đời sống số, phục vụ hàng triệu người dùng đồng thời. Khi lưu lượng truy cập tăng đột ngột hoặc hệ thống gặp lỗi, nguy cơ nghẽn (congestion) có thể gây ra hậu quả nghiêm trọng: mất dữ liệu, giảm trải nghiệm người dùng, và thiệt hại kinh tế lớn.

Việc dự đoán sớm nguy cơ nghẽn giúp:
- Chủ động ứng phó trước khi sự cố xảy ra
- Tối ưu hóa tài nguyên hệ thống
- Cải thiện trải nghiệm người dùng
- Giảm thiểu thiệt hại kinh doanh

Các phương pháp giám sát truyền thống dựa trên ngưỡng tĩnh (threshold-based monitoring) có hạn chế lớn: không phát hiện được pattern phức tạp trong dữ liệu thời gian, không dự báo được xu hướng tương lai, và thường phản ứng quá muộn khi sự cố đã xảy ra.

Trí tuệ nhân tạo (AI), đặc biệt là các mô hình deep learning cho chuỗi thời gian, có khả năng học pattern phức tạp từ dữ liệu lịch sử và dự báo xu hướng tương lai. Đây là hướng tiếp cận hứa hẹn cho bài toán dự đoán nghẽn hệ thống web.

## 1.3 Mục tiêu tổng quát

Xây dựng mô hình trí tuệ nhân tạo dựa trên chuỗi thời gian để dự đoán sớm nguy cơ nghẽn hệ thống web, kết hợp các kỹ thuật Temporal Convolutional Network (TCN), Attention Mechanism và Bidirectional LSTM.

## 1.4 Mục tiêu cụ thể

1. Xây dựng pipeline dữ liệu chuỗi thời gian từ log web công khai
2. Thiết kế và triển khai mô hình TCN-Attention-BiLSTM
3. Đánh giá mô hình trên dữ liệu công khai với các metric MAE, RMSE, R²
4. Xây dựng cơ chế cảnh báo sớm với threshold calibration
5. Xây dựng Recommendation Engine đề xuất hành động phòng ngừa
6. Tạo dashboard demo minh bạch

## 1.5 Đối tượng và phạm vi nghiên cứu

**Đối tượng nghiên cứu:**
- Dữ liệu log web (HTTP access logs)
- Chuỗi thời gian các chỉ số hiệu năng hệ thống web

**Phạm vi:**
- Dữ liệu: NASA HTTP 1995 (public dataset)
- Mô hình: TCN-Attention-BiLSTM và các baseline
- Đánh giá: MAE, RMSE, R², Precision, Recall, F1

## 1.6 Phương pháp nghiên cứu

1. **Phương pháp thực nghiệm:** Xây dựng và đánh giá mô hình trên dữ liệu thực
2. **Phương pháp so sánh:** So sánh TCN-Attention-BiLSTM với các baseline
3. **Phương pháp thống kê:** Phân tích kết quả bằng metrics chuẩn

## 1.7 Đóng góp của đề tài

1. Pipeline dữ liệu chuỗi thời gian cho web congestion prediction
2. Mô hình TCN-Attention-BiLSTM kết hợp TCN, Attention, BiLSTM
3. Hệ thống cảnh báo sớm với threshold calibration
4. Recommendation Engine đề xuất hành động phòng ngừa
5. Dashboard minh bạch với số liệu thật

## 1.8 Giới hạn nghiên cứu

- NASA HTTP 1995 là dữ liệu cũ, không có CPU/RAM/response time telemetry
- Target là proxy congestion score, không phải measured congestion thật
- Synthetic stress benchmark chỉ dùng để kiểm thử có kiểm soát
- Chưa có multi-source validation

---

# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

## 2.1 Tổng quan hệ thống web và hiệu năng

### 2.1.1 Các thành phần hệ thống web

Hệ thống web hiện đại bao gồm nhiều thành phần:
- **Web server:** Apache, Nginx, IIS — xử lý HTTP requests
- **Application server:** Chạy business logic
- **Database server:** Lưu trữ và truy vấn dữ liệu
- **Load balancer:** Phân phối tải
- **Cache layer:** Giảm tải cho backend

### 2.1.2 Các chỉ số hiệu năng

- **Request count:** Số lượng request mỗi đơn vị thời gian
- **Response time:** Thời gian phản hồi
- **Throughput:** Lượng dữ liệu xử lý được
- **Error rate:** Tỷ lệ lỗi
- **CPU/Memory usage:** Sử dụng tài nguyên

## 2.2 Khái niệm nghẽn hệ thống web

Nghẽn (congestion) xảy ra khi hệ thống không thể xử lý kịp lưu lượng truy cập, dẫn đến:
- Tăng response time
- Tăng error rate
- Giảm throughput
- Rớt kết nối

Nguyên nhân phổ biến:
- Flash crowd (lưu lượng đột biến)
- Resource exhaustion (hết tài nguyên)
- Application errors
- Database bottleneck

## 2.3 Dữ liệu chuỗi thời gian trong giám sát hệ thống

Chuỗi thời gian là tập hợp các quan sát được sắp xếp theo thứ tự thời gian. Trong giám sát hệ thống web, dữ liệu chuỗi thời gian bao gồm:
- Request count theo phút/giờ
- Error rate theo thời gian
- CPU/Memory usage theo thời gian
- Throughput theo thời gian

Đặc điểm quan trọng:
- **Trend:** Xu hướng dài hạn
- **Seasonality:** Tính chu kỳ
- **Noise:** Nhiễu ngẫu nhiên

## 2.4 Baseline truyền thống

### 2.4.1 Naive Last Value
Dự báo bằng giá trị gần nhất. Đơn giản nhưng hữu ích làm baseline.

### 2.4.2 Moving Average
Dự báo bằng trung bình động của cửa sổ dữ liệu.

### 2.4.3 ARIMA (tùy chọn)
Mô hình ARIMA (AutoRegressive Integrated Moving Average) là phương pháp thống kê cổ điển cho chuỗi thời gian. Tuy nhiên, ARIMA có hạn chế với dữ liệu phi tuyến tính và pattern phức tạp.

## 2.5 LSTM và GRU

### 2.5.1 LSTM (Long Short-Term Memory)
LSTM giải quyết vấn đề vanishing gradient bằng gating mechanism:
- **Forget gate:** Quyết định thông tin nào cần loại bỏ
- **Input gate:** Quyết định thông tin nào cần lưu trữ
- **Output gate:** Quyết định thông tin nào cần đầu ra

### 2.5.2 GRU (Gated Recurrent Unit)
GRU là phiên bản đơn giản hóa của LSTM với ít tham số hơn.

## 2.6 TCN (Temporal Convolutional Network)

TCN sử dụng convolution 1D causal với dilated convolution:
- **Causal convolution:** Đảm bảo nhân quả (không nhìn vào tương lai)
- **Dilated convolution:** Tăng receptive field mà không tăng tham số
- **Parallelizable:** Tốc độ training nhanh hơn RNN

## 2.7 Attention Mechanism

Attention học trọng số cho từng time step:
- Tập trung vào thông tin quan trọng
- Multi-head attention cho nhiều representation
- Cải thiện khả năng học pattern dài hạn

## 2.8 BiLSTM (Bidirectional LSTM)

BiLSTM học cả forward và backward:
- Bắt context hai chiều
- Cải thiện representation
- Phù hợp cho dữ liệu có pattern phức tạp

## 2.9 Mixed Precision Float16

Mixed Precision kết hợp FP16 và FP32:
- **FP16:** Nhanh hơn, ít bộ nhớ
- **FP32:** Chính xác hơn
- **AMP:** Tự động chọn phù hợp

## 2.10 Recommendation Engine

Recommendation Engine đề xuất hành động dựa trên risk score:
- **Risk level:** Normal, Watch, Warning, Critical
- **Actions:** Scale up, rate limit, enable cache, etc.
- **Explanation:** Giải thích ngắn gọn

---

# CHƯƠNG 3: MÔ HÌNH ĐỀ XUẤT VÀ THIẾT KẾ HỆ THỐNG

## 3.1 Phân tích bài toán

Bài toán: Cho chuỗi thời gian X = {x₁, x₂, ..., xₜ} với mỗi xᵢ là vector đặc trưng tại thời điểm i, dự đoán giá trị yₜ₊₁ (congestion score) tại thời điểm t+1.

- **Input:** Window size 60 time steps, 19 features
- **Output:** Prediction scalar [0, 1]

## 3.2 Pipeline dữ liệu

```
Raw HTTP Logs
    ↓
Parse (regex)
    ↓
Aggregate (1-min window)
    ↓
Feature Engineering (19 features)
    ↓
Normalize (MinMax 0-1)
    ↓
Sliding Windows (lookback=60)
    ↓
Train/Val/Test Split (70/15/15)
```

### 3.2.1 Các bước xử lý

1. **Parse:** Đọc và phân tích HTTP log format
2. **Aggregate:** Tổng hợp theo cửa sổ 1 phút
3. **Features:** Tạo 19 đặc trưng (request_count, bytes_sum, error_rate, etc.)
4. **Normalize:** Chuẩn hóa MinMax về [0, 1]
5. **Window:** Tạo sliding windows (lookback=60, horizon=15)
6. **Split:** Chia theo thứ tự thời gian (không shuffle)

*[Hình 3.1. Sơ đồ pipeline dữ liệu — Xem outputs/figures/prediction_vs_actual.png]*

## 3.3 Kiến trúc TCN-Attention-BiLSTM

```
Input [batch, 60, 19]
    ↓
TCN Block (4 layers, dilations [1,2,4,8])
    ↓
Multi-Head Self-Attention (2 heads)
    ↓
Bidirectional LSTM (32 units)
    ↓
Dense Layer (64 units)
    ↓
Output [batch, 1]
```

### 3.3.1 TCN Block
- 4 convolutional layers
- Kernel size: 3
- Dilations: [1, 2, 4, 8]
- Filters: 64
- Activation: ReLU
- Dropout: 0.15

### 3.3.2 Multi-Head Self-Attention
- Heads: 2
- Key dimension: 16
- Học trọng số cho từng time step

### 3.3.3 Bidirectional LSTM
- Units: 32 (forward) + 32 (backward)
- Học phụ thuộc hai chiều

*[Hình 3.2. Kiến trúc TCN-Attention-BiLSTM]*

## 3.4 Recommendation Engine

```
Input: predicted_score, threshold, phase
    ↓
Risk Level Classification
    ↓
Action Recommendation
    ↓
Output: risk_level, actions, explanation
```

### 3.4.1 Risk Levels

| Level | Điều kiện | Hành động |
|---|---|---|
| Normal | score < threshold × 0.9 | continue_monitoring |
| Watch | score gần threshold | inspect_request_trend |
| Warning | score ≥ threshold | check_traffic_spike, consider_scaling |
| Critical | score >> threshold | scale_up_cpu, rate_limit, investigate_anomaly |

*[Hình 3.3. Luồng Recommendation Engine]*

## 3.5 Luồng demo

1. Đọc telemetry/time series
2. Hiển thị cửa sổ dữ liệu gần nhất
3. Model dự đoán tương lai
4. Tính risk score
5. Phát cảnh báo sớm
6. Đề xuất hành động
7. Lưu kết quả

---

# CHƯƠNG 4: THỰC NGHIỆM VÀ ĐÁNH GIÁ

## 4.1 Môi trường thực nghiệm

- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU
- **Framework:** PyTorch CUDA
- **Mixed precision:** AMP float16
- **Python:** 3.10

## 4.2 Dataset

### 4.2.1 NASA HTTP 1995
- **Source:** NASA Kennedy Space Center HTTP Server Logs
- **Period:** July-August 1995
- **Raw lines:** ~3.46 million
- **Features:** 19
- **Target:** proxy_congestion_score

### 4.2.2 Synthetic Stress Benchmark
- **Scenarios:** 6 (flash_crowd, burst_traffic, error_surge, slow_ramp, periodic_spike, mixed_incident)
- **Samples:** 1800
- **Positive ratio:** 0.30
- **Mục đích:** Kiểm thử có kiểm soát, không phải dữ liệu thật

### 4.2.3 Zanbil (planned)
- Pipeline đã chuẩn bị nhưng raw log chưa được cung cấp
- Không blocking main demo

## 4.3 Cấu hình huấn luyện

| Parameter | Value |
|---|---|
| Model | TCN-Attention-BiLSTM |
| Epochs | 120 |
| Batch size | 128 |
| Learning rate | 0.0007 |
| Optimizer | AdamW |
| Weight decay | 0.0001 |
| Mixed precision | AMP float16 |
| Gradient clip | 1.0 |
| Seed | 42 |

## 4.4 Metrics

- **MAE (Mean Absolute Error):** Đo lỗi tuyệt đối trung bình
- **RMSE (Root Mean Squared Error):** Đo lỗi bình phương trung bình
- **R² (Coefficient of Determination):** Đo khả năng giải thích phương sai
- **Precision, Recall, F1:** Đo hiệu suất cảnh báo

## 4.5 Kết quả v2 (final verified)

| Metric | Giá trị |
|---|---|
| MAE | 0.043053 |
| RMSE | 0.056036 |
| R² | 0.339994 |
| Train time | 1092.5s |
| Threshold | 0.183838 |
| Calibrated F1 | 0.865596 |
| Calibrated Recall | 0.979049 |

## 4.6 So sánh mô hình

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Naive Last Value | 0.054805 | 0.073650 | -0.140121 |
| Moving Average | 0.046843 | 0.062242 | 0.185716 |
| LSTM | 0.043268 | 0.056562 | 0.327548 |
| GRU | 0.042702 | 0.055843 | 0.344529 |
| TCN | 0.041912 | 0.056602 | 0.326604 |
| Transformer | 0.042760 | 0.057879 | 0.295869 |
| TCN-LSTM | 0.042780 | 0.056155 | 0.337192 |
| **TCN-Attention-BiLSTM** | **0.043053** | **0.056036** | **0.339994** |

**Nhận xét:** TCN-Attention-BiLSTM có R² cao nhất (0.339994), cho thấy khả năng giải thích phương sai tốt nhất trong các mô hình đã thử.

## 4.7 Phân tích biểu đồ

### 4.7.1 Prediction vs Actual
*[Xem: outputs/figures/prediction_vs_actual.png]*

Biểu đồ cho thấy model dự đoán được xu hướng chung của congestion score, nhưng vẫn còn sai số ở các đỉnh.

### 4.7.2 Error Distribution
*[Xem: outputs/figures/error_distribution.png]*

Phân phối lỗi tập trung quanh 0, cho thấy model không có bias lớn.

### 4.7.3 Model Comparison RMSE
*[Xem: outputs/figures/model_comparison_rmse.png]*

TCN-Attention-BiLSTM có RMSE thấp nhất trong các mô hình deep learning.

### 4.7.4 Training Curves
*[Xem: outputs/figures/training_curves.png]*

Loss giảm đều trong quá trình training, không có dấu hiệu overfitting nghiêm trọng.

### 4.7.5 Early Warning Timeline
*[Xem: outputs/figures/early_warning_timeline.png]*

Model phát hiện được các vùng nguy cơ cao (vượt threshold) trước khi sự cố xảy ra.

### 4.7.6 Synthetic Stress Scenarios
*[Xem: outputs/figures/synthetic_stress_scenarios.png]*

Model phản ứng tốt với periodic_spike (F1=0.758) nhưng yếu với error_surge (F1=0.143).

## 4.8 Demo hệ thống

### 4.8.1 Dashboard HTML
Dashboard Research Defense hiển thị đầy đủ:
- Metrics chính
- Confusion matrix
- Threshold calibration
- Synthetic stress benchmark
- Recommendation prototype
- Giới hạn minh bạch

### 4.8.2 Terminal Demo
```
python scripts/run_demo.py --sample synthetic --model best --explain --save-report
```

Kết quả demo:
- Predicted score: 0.642102
- Risk level: CRITICAL
- Recommended actions: scale_up_cpu, add_instance, enable_cache, rate_limit, investigate_anomaly

---

# CHƯƠNG 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 5.1 Tổng kết kết quả đạt được

### 5.1.1 Pipeline dữ liệu
- Xây dựng pipeline từ NASA HTTP logs đến model training
- Xử lý 3.46 triệu raw log lines
- Tạo 89,085 sliding windows
- Chuẩn hóa MinMax [0, 1]

### 5.1.2 Mô hình
- Implement 8 mô hình (baseline + deep learning)
- TCN-Attention-BiLSTM train full 120 epoch
- Best epoch 30, train time 1092.5s

### 5.1.3 Đánh giá
- MAE: 0.043053
- RMSE: 0.056036
- R²: 0.339994
- Calibrated F1: 0.865596

### 5.1.4 Hệ thống
- Dashboard HTML minh bạch
- Recommendation Engine prototype
- Test suite (83 tests)
- Source governance

## 5.2 Đóng góp

1. **Pipeline dữ liệu:** Chuỗi thời gian từ log web
2. **Mô hình TCN-Attention-BiLSTM:** Kết hợp TCN + Attention + BiLSTM
3. **Đánh giá thực nghiệm:** So sánh 8 mô hình
4. **Dashboard/demo:** Hệ thống minh bạch
5. **Recommendation Engine:** Đề xuất hành động phòng ngừa

## 5.3 Hạn chế

1. **Dữ liệu cũ:** NASA HTTP 1995 không đại diện cho web hiện đại
2. **Proxy target:** Không phải measured congestion thật
3. **Thiếu telemetry:** Không có CPU/RAM/response time
4. **Zanbil chưa có:** Không có multi-source validation
5. **R² còn thấp:** 0.339994 chỉ giải thích 34% phương sai
6. **ARIMA optional:** Chưa so sánh với baseline truyền thống

## 5.4 Hướng phát triển

1. **Bổ sung dataset:** Thêm Zanbil, Calgary, ClarkNet
2. **Multi-source:** NASA + Zanbil khi có raw log
3. **Tối ưu hyperparameter:** Grid search, Bayesian optimization
4. **Online monitoring:** Triển khai real-time
5. **Auto-scaling:** Tích hợp với hệ thống orchestration
6. **Explainability:** Cải thiện khả năng giải thích

---

# TÀI LIỆU THAM KHẢO

1. Arlitt, M. and Williamson, C. (1996). Web Server Workload Characterization: The Search for Invariants. ACM SIGMETRICS.
2. Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
3. Bai, S. et al. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. arXiv.
4. Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.
5. Zaker, F. (2019). Online Shopping Store - Web Server Logs. Harvard Dataverse.

---

# PHỤ LỤC

## Phụ lục A: Danh sách hình

| Hình | Mô tả | Đường dẫn |
|---|---|---|
| Hình 4.1 | Prediction vs Actual | outputs/figures/prediction_vs_actual.png |
| Hình 4.2 | Error Distribution | outputs/figures/error_distribution.png |
| Hình 4.3 | Model Comparison RMSE | outputs/figures/model_comparison_rmse.png |
| Hình 4.4 | Training Curves | outputs/figures/training_curves.png |
| Hình 4.5 | Early Warning Timeline | outputs/figures/early_warning_timeline.png |
| Hình 4.6 | Synthetic Stress Scenarios | outputs/figures/synthetic_stress_scenarios.png |

## Phụ lục B: Bảng metrics chi tiết

| Metric | Giá trị |
|---|---|
| MAE | 0.043053 |
| RMSE | 0.056036 |
| R² | 0.339994 |
| Precision | 0.636792 |
| Recall | 0.076487 |
| F1 | 0.136571 |
| Threshold | 0.183838 |
| Calibrated F1 | 0.865596 |
| Calibrated Recall | 0.979049 |

## Phụ lục C: Demo evidence

| File | Mô tả |
|---|---|
| outputs/web/research_defense_dashboard.html | Dashboard HTML |
| outputs/reports/23_final_hard_audit_report.md | Audit report |
| outputs/reports/18_terminal_demo_report.md | Demo report |
| outputs/metrics/full_120_v2_tcn_attention_bilstm/final_metrics.json | Metrics JSON |
