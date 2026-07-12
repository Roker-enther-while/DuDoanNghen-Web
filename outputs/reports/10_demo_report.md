# PHASE 7 — Demo Report

## Demo Components

### 1. Research Defense Dashboard (HTML)
- **Path:** outputs/web/research_defense_dashboard.html
- **Features:**
  - Hero section với title và warning
  - Research scope cards
  - Data pipeline evidence
  - Target construction & limitation
  - Model training evidence
  - Real public proxy test result
  - Confusion matrix & alert failure analysis
  - Threshold calibration
  - Synthetic stress benchmark
  - Scenario-level analysis
  - Phase-level analysis
  - Recommendation prototype
  - What this result proves/doesn't prove
  - Next work
  - Minh bạch & giới hạn

### 2. Dashboard Payload (JSON)
- **Path:** outputs/web/final_state_b_dashboard_payload.json
- **Features:**
  - run_state
  - project
  - real_public_proxy_result
  - threshold_calibration
  - synthetic_stress_result
  - data_context
  - governance
  - warnings

## Demo Flow

1. **Đọc telemetry/time series** → Data pipeline
2. **Hiển thị cửa sổ dữ liệu gần nhất** → Dashboard
3. **Model dự đoán tương lai** → TCN-Attention-BiLSTM
4. **Tính risk score** → Alert metrics
5. **Phát cảnh báo sớm** → Threshold calibration
6. **Đề xuất hành động** → Recommendation Engine
7. **Lưu biểu đồ/kết quả** → Dashboard HTML

## Cách chạy demo

```bash
# Mở dashboard HTML
start outputs/web/research_defense_dashboard.html

# Xem dashboard payload
cat outputs/web/final_state_b_dashboard_payload.json
```

## Limitations

- ⚠️ Dashboard tĩnh HTML (không có real-time update)
- ⚠️ Chưa có terminal demo script
- ⚠️ Recommendation Engine là prototype
