# PHASE 1 — Reproducibility Report

## Cấu trúc dự án đã chuẩn hóa

### Files đã tạo/cập nhật
- `README.md` — Mô tả đề tài, mục tiêu, cách chạy
- `data/README.md` — Giải thích dữ liệu và cách đặt data

### Cấu hình hiện có
| Config | Mô tả |
|---|---|
| configs/data/nasa_http_smoke.yaml | Smoke test data pipeline |
| configs/data/nasa_http_3m.yaml | Full NASA 3-month data |
| configs/data/synthetic_stress_benchmark.yaml | Synthetic stress |
| configs/training/smoke.yaml | Smoke training (2 epochs) |
| configs/training/tcn_attention_bilstm_full_120.yaml | Full 120 epochs |

### Scripts hiện có
| Script | Mô tả |
|---|---|
| scripts/run_data_pipeline.py | Chạy data pipeline |
| scripts/train_model.py | Train single model |
| scripts/evaluate_model.py | Đánh giá model |
| scripts/calibrate_alert_threshold.py | Calibrate threshold |
| scripts/evaluate_synthetic_stress.py | Synthetic stress benchmark |
| scripts/generate_synthetic_stress_benchmark.py | Tạo synthetic data |

### Outputs tự động
- `outputs/metrics/` — Metrics JSON files
- `outputs/reports/` — Report markdown files
- `outputs/models/` — Model checkpoints
- `outputs/predictions/` — Test predictions
- `outputs/web/` — Dashboard HTML/JSON

## Cách chạy lại

```bash
# 1. Chạy tests
python -m pytest -q

# 2. Chạy data pipeline
python scripts/run_data_pipeline.py --config configs/data/nasa_http_smoke.yaml

# 3. Train model
python scripts/train_model.py --config configs/training/smoke.yaml

# 4. Đánh giá
python scripts/evaluate_model.py --config configs/training/smoke.yaml

# 5. Xem dashboard
start outputs/web/research_defense_dashboard.html
```

## Trạng thái

- ✅ Cấu trúc thư mục chuẩn hóa
- ✅ README.md mô tả đề tài
- ✅ data/README.md giải thích data
- ✅ configs/ có smoke/quick/balanced/full
- ✅ scripts/ có entrypoint rõ ràng
- ✅ outputs/ được tạo tự động
- ✅ pytest pass (83 tests)
