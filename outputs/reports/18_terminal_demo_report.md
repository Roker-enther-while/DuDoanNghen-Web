# Terminal Demo Report

**Timestamp:** 20260531_161943
**Sample:** synthetic
**Model:** tcn_attention_bilstm

## Prediction

| Metric | Value |
|---|---|
| Predicted Score | 0.642102 |
| Calibrated Threshold | 0.05 |
| Risk Level | CRITICAL |

## Recommendation

- **Reason:** Score significantly above threshold
- **Actions:** scale_up_cpu, add_instance, enable_cache, rate_limit, investigate_anomaly
- **Explanation:** Nguy co nghen cao! Can scale up ngay, bat cache, rate limit va dieu tra anomaly.

## Model Metrics

| Metric | Value |
|---|---|
| MAE | 0.043053 |
| RMSE | 0.056036 |
| R² | 0.339994 |
