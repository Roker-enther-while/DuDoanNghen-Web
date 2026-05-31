# Chương 4: Thực Nghiệm Và Đánh Giá

## 4.1 Dataset

### NASA HTTP 1995
- **Source:** NASA Kennedy Space Center
- **Period:** July-August 1995
- **Raw lines:** ~3.46 million
- **Features:** 19
- **Target:** proxy_congestion_score

### Synthetic Stress Benchmark
- **Scenarios:** 6 (flash_crowd, burst_traffic, error_surge, slow_ramp, periodic_spike, mixed_incident)
- **Samples:** 1800
- **Positive ratio:** 0.30

## 4.2 Preprocessing

1. Parse HTTP logs
2. Aggregate theo 1 phút
3. Tạo 19 features
4. Normalize MinMax [0, 1]
5. Sliding windows (lookback=60, horizon=15)
6. Chronological split (70/15/15)
7. Float16 storage

## 4.3 Cấu hình train

| Parameter | Value |
|---|---|
| Model | TCN-Attention-BiLSTM |
| Backend | PyTorch CUDA |
| GPU | RTX 4060 Laptop |
| Epochs | 120 |
| Batch size | 128 |
| Learning rate | 0.0007 |
| Optimizer | AdamW |
| Mixed precision | AMP float16 |

## 4.4 Metrics

### Regression
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)

### Alert
- Precision
- Recall
- F1-score
- Confusion Matrix

## 4.5 Bảng kết quả

### Model Comparison
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Naive Last Value | 0.054805 | 0.073650 | -0.140121 |
| Moving Average | 0.046843 | 0.062242 | 0.185716 |
| LSTM | 0.043268 | 0.056562 | 0.327548 |
| GRU | 0.042702 | 0.055843 | 0.344529 |
| TCN | 0.041912 | 0.056602 | 0.326604 |
| Transformer | 0.042760 | 0.057879 | 0.295869 |
| TCN-LSTM | 0.042780 | 0.056155 | 0.337192 |
| TCN-Attention-BiLSTM | 0.042792 | 0.056399 | 0.331430 |

### TCN-Attention-BiLSTM Alert Results
| Metric | Original (p90) | Calibrated |
|---|---|---|
| Threshold | 0.183838 | 0.05 |
| F1 | 0.014599 | 0.865596 |
| Recall | 0.007365 | 0.979049 |
| Precision | 0.812500 | 0.775706 |

## 4.6 Phân tích so sánh

### Regression
- GRU có R² cao nhất (0.345)
- TCN-Attention-BiLSTM R² = 0.331 (gần GRU)
- Baselines (Naive, Moving Average) có R² thấp

### Alert
- Threshold p90 quá cao → recall rất thấp
- Calibration cải thiện đáng kể
- TCN-Attention-BiLSTM cần calibration để hoạt động

## 4.7 Demo cảnh báo sớm

### Flow
1. Đọc data window gần nhất
2. Model dự đoán score
3. So sánh với threshold
4. Nếu vượt → cảnh báo
5. Recommendation Engine đề xuất hành động

### Example
```
Input: window [60, 19]
Prediction: 0.08
Threshold: 0.05
Risk Level: Warning
Actions: [check_traffic_spike, check_backend_errors, consider_scaling]
```
