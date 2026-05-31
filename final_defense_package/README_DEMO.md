# README DEMO

## Mô tả

Đây là gói demo cho đề tài "Dự đoán nghẽn hệ thống web bằng mô hình trí tuệ nhân tạo dựa trên chuỗi thời gian".

**Trạng thái:** READY FOR DEMO

## Cách mở Dashboard

```bash
# Mở dashboard HTML trong trình duyệt
start outputs/web/research_defense_dashboard.html
```

Dashboard hiển thị:
- Metrics chính (MAE, RMSE, R²)
- Confusion matrix
- Threshold calibration
- Synthetic stress benchmark
- Recommendation Engine
- Figures
- Giới hạn minh bạch

## Cách chạy Demo Terminal

```bash
# Chạy demo với synthetic data
python scripts/run_demo.py --sample synthetic --model best --explain --save-report
```

Demo sẽ:
1. Load metrics từ artifact
2. Tạo synthetic data
3. Dự đoán congestion score
4. Tính risk level
5. Đề xuất hành động
6. Lưu report

## Cách xem Metrics

```bash
# Xem metrics v2 (final verified)
cat outputs/metrics/full_120_v2_tcn_attention_bilstm/final_metrics.json
```

Metrics chính:
- MAE: 0.043053
- RMSE: 0.056036
- R²: 0.339994
- Train time: 1092.5s

## Cách xem Audit Report

```bash
# Xem audit report
cat outputs/reports/23_final_hard_audit_report.md
```

## Figures

```bash
# Xem figures
start outputs/figures/prediction_vs_actual.png
start outputs/figures/model_comparison_rmse.png
start outputs/figures/training_curves.png
start outputs/figures/early_warning_timeline.png
start outputs/figures/synthetic_stress_scenarios.png
start outputs/figures/error_distribution.png
```

## Trạng thái Dữ liệu

| Dataset | Trạng thái |
|---|---|
| NASA HTTP 1995 | ✅ Có sẵn |
| Synthetic stress | ✅ Có sẵn |
| Zanbil | ⚠️ Pipeline sẵn, raw log chưa có |

## Trạng thái Giới hạn

- NASA target là proxy congestion score, không phải measured congestion
- Synthetic stress là controlled benchmark, không phải real-world
- Zanbil raw chưa cung cấp
- R²=0.34 còn dư địa cải thiện

## Files Demo Chính

| File | Mô tả |
|---|---|
| outputs/web/research_defense_dashboard.html | Dashboard HTML |
| outputs/reports/23_final_hard_audit_report.md | Audit report |
| outputs/reports/18_terminal_demo_report.md | Demo report |
| outputs/metrics/full_120_v2_tcn_attention_bilstm/final_metrics.json | Metrics JSON |
| outputs/figures/*.png | 6 figures |
