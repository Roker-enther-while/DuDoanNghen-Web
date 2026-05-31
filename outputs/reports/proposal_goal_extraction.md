# Proposal Goal Extraction — TCN-Attention-BiLSTM

## Đề cương gốc

**File đề cương:** Không tìm thấy file `.docx` trong repo. Nội dung khôi phục từ CLAUDE.md, NEXT_STEP.md, AGENT_REPORT.md và context dự án.

**NEED_CONFIRMATION:** Các mục tiêu dưới đây được suy luận từ lịch sử dự án. Nếu có đề cương gốc, cần đối chiếu lại.

## 1. Tên đề tài

Dự đoán nghẽn hệ thống web bằng mô hình trí tuệ nhân tạo dựa trên chuỗi thời gian.

## 2. Mục tiêu tổng quát

Nghiên cứu và đề xuất mô hình AI dựa trên dữ liệu chuỗi thời gian để dự đoán sớm nguy cơ nghẽn hệ thống web.

## 3. Chỉ số dữ liệu đề cương mong muốn

| Chỉ số đề cương | Có trong NASA HTTP log? | Ghi chú |
|---|---|---|
| Request/lưu lượng truy cập | ✅ Có | request_count |
| CPU usage | ❌ Không có | Không có trong HTTP log |
| Memory usage | ❌ Không có | Không có trong HTTP log |
| Response time | ❌ Không có | Không có trong HTTP log |
| Throughput | ✅ Có (approx) | throughput_bytes_per_min |

**Kết luận:** NASA HTTP log chỉ có traffic-level features. Không có system telemetry (CPU, RAM, response time). Target hiện tại là **proxy congestion score** được xây dựng từ features log, không phải measured congestion.

## 4. Mô hình cần nghiên cứu/so sánh

| Mô hình | Trạng thái | Ghi chú |
|---|---|---|
| Moving Average | ✅ Đã train | Baseline |
| ARIMA/Holt-Winters | ❌ Chưa có | Planned |
| LSTM | ✅ Đã train | |
| GRU | ✅ Đã train | |
| TCN | ✅ Đã train | |
| Transformer (Attention) | ✅ Đã train | |
| TCN-Attention-BiLSTM | ✅ Full 120 epoch | Model chính |

## 5. Tối ưu dữ liệu/hardware

| Yêu cầu | Trạng thái | Ghi chú |
|---|---|---|
| Chuẩn hóa 0-1 | ✅ Đã làm | Min-max normalization trên train split |
| Mixed Precision Float16 | ✅ Đã làm | PyTorch AMP, storage float16 |
| TurboQuant | ❌ Chưa có | Không có code/implementation |
| GPU NVIDIA Ada Lovelace | ✅ Có | RTX 4060 Laptop GPU (Ada Lovelace) |

## 6. Đánh giá

| Metric | Có | Giá trị |
|---|---|---|
| MAE | ✅ | 0.042792 |
| RMSE | ✅ | 0.056399 |
| R² | ✅ | 0.331430 |
| Precision | ✅ | 0.812500 |
| Recall | ✅ | 0.007365 |
| F1 | ✅ | 0.014599 |
| Confusion matrix | ✅ | TP=13, FP=3, TN=11562, FN=1752 |

## 7. Sản phẩm

| Sản phẩm | Trạng thái | Ghi chú |
|---|---|---|
| Báo cáo nghiên cứu | ✅ | final_state_b_research_summary.md |
| Mô hình TCN-Attention-BiLSTM | ✅ | best_model.pt (366KB) |
| Bộ dữ liệu chuỗi thời gian | ✅ | windows_fp16.npz |
| Mã nguồn Python | ✅ | src/, scripts/ |
| Prototype/dashboard demo | ✅ | final_state_b_dashboard_payload.json |
| Kết quả thực nghiệm/biểu đồ | ✅ | metrics, reports, synthetic eval |

## 8. Dữ liệu

| Nguồn | Trạng thái | Ghi chú |
|---|---|---|
| NASA HTTP 1995 | ✅ Đã dùng | Real public dataset |
| Google Trace | ❌ Chưa dùng | Declared/disabled |
| Zanbil | ❌ Raw missing | data/raw/zanbil/access.log chưa có |
| Synthetic scenarios | ✅ Đã tạo | 6 scenarios, 1800 samples |
