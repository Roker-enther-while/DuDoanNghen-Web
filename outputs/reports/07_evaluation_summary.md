# PHASE 5 — Evaluation Summary

## Test Set

- **Size:** 13,330 windows
- **Shape:** [13330, 60, 19]
- **Source:** NASA HTTP 1995 (chronological split)

## TCN-Attention-BiLSTM Results

### Regression Metrics
| Metric | Value |
|---|---|
| MAE | 0.042792 |
| RMSE | 0.056399 |
| R² | 0.331430 |

### Alert Metrics (threshold p90 = 0.183838)
| Metric | Value |
|---|---|
| Precision | 0.812500 |
| Recall | 0.007365 |
| F1 | 0.014599 |
| TP | 13 |
| FP | 3 |
| TN | 11,562 |
| FN | 1,752 |

### Threshold Calibration
| Metric | Original (p90) | Calibrated |
|---|---|---|
| Threshold | 0.183838 | 0.05 |
| F1 | 0.014599 | 0.865596 |
| Recall | 0.007365 | 0.979049 |
| Precision | 0.812500 | 0.775706 |

## Analysis

### Regression
- R² = 0.331 cho thấy model giải thích 33% phương sai
- MAE và RMSE ở mức chấp nhận được cho proxy score

### Alert
- Threshold p90 quá cao → recall rất thấp (0.7%)
- Calibration giảm threshold xuống 0.05 → recall tăng lên 97.9%
- Calibration phải được trình bày tách biệt

### Key Findings
1. Model học được xu hướng proxy score
2. Alert cần calibration để hoạt động
3. Không nên dùng F1 gốc (0.015) để đánh giá
