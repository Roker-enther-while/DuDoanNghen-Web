# Dự Đoán Nghẽn Hệ Thống Web Bằng Mô Hình TCN-Attention-BiLSTM

## Mục tiêu

Xây dựng pipeline AI minh bạch để dự đoán sớm nguy cơ nghẽn hệ thống web dựa trên chuỗi thời gian từ dữ liệu log web công khai.

## Phạm vi

- **Model chính:** TCN-Attention-BiLSTM (Temporal Convolutional Network + Attention + Bidirectional LSTM)
- **Data:** NASA HTTP 1995 (proxy congestion score)
- **Target:** proxy_congestion_score (không phải measured congestion thật)
- **Baseline:** Moving Average, LSTM, GRU, TCN, Transformer

## Cách chạy nhanh

```bash
# Chạy tests
python -m pytest -q

# Chạy data pipeline (smoke)
python scripts/run_data_pipeline.py --config configs/data/nasa_http_smoke.yaml

# Train model (smoke)
python scripts/train_model.py --config configs/training/smoke.yaml

# Đánh giá tất cả models
python scripts/evaluate_model.py --config configs/training/smoke.yaml

# Xem dashboard
start outputs/web/research_defense_dashboard.html
```

## Cách chạy train/evaluate/demo

```bash
# Train TCN-Attention-BiLSTM full 120 epoch
python scripts/train_model.py --config configs/training/tcn_attention_bilstm_full_120.yaml

# Đánh giá model
python scripts/evaluate_model.py --predictions outputs/predictions/full_120_tcn_attention_bilstm/test_predictions.csv --threshold 0.183838

# Calibrate threshold
python scripts/calibrate_alert_threshold.py --predictions outputs/predictions/full_120_tcn_attention_bilstm/test_predictions.csv

# Chạy synthetic stress benchmark
python scripts/evaluate_synthetic_stress.py --data data/processed/synthetic_stress/windows/windows_fp16.npz --labels data/processed/synthetic_stress/labels/synthetic_stress_labels.csv --model-path outputs/models/full_120_tcn_attention_bilstm/best_model.pt --output-dir outputs/synthetic_stress_eval/full_120_tcn_attention_bilstm
```

## Kết quả chính

| Metric | Giá trị |
|---|---|
| MAE | 0.042792 |
| RMSE | 0.056399 |
| R² | 0.331430 |
| Precision | 0.812500 |
| Recall | 0.007365 |
| F1 | 0.014599 |

## Giới hạn

- NASA target là **proxy congestion score**, không phải measured congestion thật
- Không có CPU/RAM/response time telemetry trong NASA HTTP log
- Synthetic stress là **controlled benchmark**, không phải real-world data
- Zanbil raw chưa có → không có multi-source claim

## Cấu trúc thư mục

```
├── configs/          # Cấu hình data pipeline và training
├── data/             # Raw và processed data
├── docs/             # Tài liệu kỹ thuật
├── outputs/          # Metrics, reports, models, dashboard
├── scripts/          # Python scripts
├── src/              # Source code (models, training, data)
└── tests/            # Pytest test suite
```

## License

NASA HTTP 1995: Internet Traffic Archive redistributable trace permission
