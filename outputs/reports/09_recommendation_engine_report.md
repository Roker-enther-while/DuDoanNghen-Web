# PHASE 6 — Recommendation Engine Report

## Overview

Recommendation Engine dựa trên rule-based prototype, sử dụng risk score và threshold để đề xuất hành động phòng ngừa.

## Logic

### Risk Levels

| Level | Điều kiện | Màu sắc |
|---|---|---|
| Normal | score < calibrated_threshold | 🟢 Xanh lá |
| Watch | score gần threshold (±10%) | 🟡 Vàng |
| Warning | score ≥ calibrated_threshold | 🟠 Cam |
| Critical | score cao + persistent incident | 🔴 Đỏ |

### Recommended Actions

| Risk Level | Actions |
|---|---|
| Normal | continue_monitoring |
| Watch | inspect_request_trend, inspect_error_trend |
| Warning | check_traffic_spike, check_backend_errors, consider_scaling |
| Critical | scale_up_cpu, add_instance, enable_cache, rate_limit, investigate_anomaly |

### Rule-Based Mapping

```python
def get_recommendation(score, threshold, phase=None):
    if score < threshold * 0.9:
        return {
            "risk_level": "normal",
            "risk_score": score,
            "reason": "Score below threshold",
            "recommended_actions": ["continue_monitoring"],
            "explanation": "Hệ thống hoạt động bình thường"
        }
    elif score < threshold:
        return {
            "risk_level": "watch",
            "risk_score": score,
            "reason": "Score near threshold",
            "recommended_actions": ["inspect_request_trend", "inspect_error_trend"],
            "explanation": "Gần ngưỡng cảnh báo, cần theo dõi thêm"
        }
    elif score < threshold * 1.5:
        return {
            "risk_level": "warning",
            "risk_score": score,
            "reason": "Score above threshold",
            "recommended_actions": ["check_traffic_spike", "check_backend_errors", "consider_scaling"],
            "explanation": "Vượt ngưỡng cảnh báo, cần kiểm tra và cân nhắc scale"
        }
    else:
        return {
            "risk_level": "critical",
            "risk_score": score,
            "reason": "Score significantly above threshold",
            "recommended_actions": ["scale_up_cpu", "add_instance", "enable_cache", "rate_limit", "investigate_anomaly"],
            "explanation": "Nguy cơ nghẽn cao, cần hành động ngay"
        }
```

## Limitations

- ⚠️ Đây là rule-based prototype, chưa phải production auto-scaling
- ⚠️ Chưa có uncertainty estimation
- ⚠️ Chưa tích hợp với hệ thống orchestration thực

## Example

```
Input: score=0.08, threshold=0.05
Output:
  risk_level: warning
  risk_score: 0.08
  reason: Score above threshold
  recommended_actions: [check_traffic_spike, check_backend_errors, consider_scaling]
  explanation: Vượt ngưỡng cảnh báo, cần kiểm tra và cân nhắc scale
```
