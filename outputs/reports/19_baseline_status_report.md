# PHASE D — Baseline Status Report

## Baselines Implemented

| Baseline | Status | Artifact |
|---|---|---|
| Naive Last Value | ✅ FULL | baseline.json |
| Moving Average | ✅ FULL | baseline.json |
| LSTM | ✅ FULL | model.pt |
| GRU | ✅ FULL | model.pt |
| TCN | ✅ FULL | model.pt |
| Transformer | ✅ FULL | model.pt |
| TCN-LSTM | ✅ FULL | model.pt |

## ARIMA Status

**Status:** OPTIONAL / PARTIAL

**Reason:**
- statsmodels dependency có thể gây conflict
- Naive/Moving Average đã làm baseline chính thức
- ARIMA không phải yêu cầu bắt buộc trong đề cương

**Decision:** Giữ ARIMA là OPTIONAL. Baseline chính thức dùng Naive/Moving Average để đảm bảo reproducibility.

## Baseline Comparison (Verified)

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

## Conclusion

- ✅ 7 baselines đã train và đánh giá
- ⚠️ ARIMA là OPTIONAL (không blocking)
- ✅ Baseline comparison đã verified
