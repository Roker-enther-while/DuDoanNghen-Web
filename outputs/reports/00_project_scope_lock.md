# PHASE 0 — Project Scope Lock Report

## Dự án đang có gì

### Cấu trúc thư mục
| Thư mục | Số files | Mô tả |
|---|---|---|
| configs/ | 21 | Cấu hình data pipeline và training |
| data/ | 32 | Raw và processed data |
| docs/ | 4 | Tài liệu kỹ thuật |
| outputs/ | 244 | Metrics, reports, models, dashboard |
| scripts/ | 25 | Python scripts chạy pipeline |
| src/ | 34 | Source code (models, training, data) |
| tests/ | 35 | Pytest test suite |

### Framework chính
- **Ngôn ngữ:** Python 3.10
- **Framework:** PyTorch CUDA
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU
- **Mixed precision:** AMP float16

### Data hiện có
- **NASA HTTP 1995:** ✅ Đã xử lý thành windows_fp16.npz
- **Synthetic stress:** ✅ 6 scenarios, 1800 samples
- **Zanbil:** ❌ Raw missing
- **Google Cluster:** ❌ Declared/disabled

### Models đã train
| Model | Status | Artifact |
|---|---|---|
| Naive Last Value | ✅ | baseline.json |
| Moving Average | ✅ | baseline.json |
| LSTM | ✅ | model.pt |
| GRU | ✅ | model.pt |
| TCN | ✅ | model.pt |
| Transformer | ✅ | model.pt |
| TCN-LSTM | ✅ | model.pt |
| TCN-Attention-BiLSTM | ✅ Full 120 | best_model.pt |

### Reports đã có
- final_state_b_research_summary.md
- final_artifact_manifest.md
- proposal_goal_extraction.md
- proposal_vs_current_gap_analysis.md
- proposal_revision_recommendations.md
- data_quantity_analysis.md
- research_defense_dashboard.html

### Tests
- **pytest:** 85 passed (sau khi xóa test_cleanup_small_test_artifacts.py)

## Còn thiếu gì so với đề cương

| Yêu cầu đề cương | Trạng thái | Ghi chú |
|---|---|---|
| Request count | ✅ Có | NASA HTTP log |
| Response time | ❌ Không có | NASA HTTP không có |
| Throughput | ✅ Có (approx) | throughput_bytes_per_min |
| CPU usage | ❌ Không có | NASA HTTP không có |
| Memory usage | ❌ Không có | NASA HTTP không có |
| Error rate | ✅ Có | error_rate feature |
| Moving Average baseline | ✅ Có | |
| ARIMA baseline | ❌ Chưa có | Optional |
| LSTM/GRU | ✅ Có | |
| TCN | ✅ Có | |
| TCN-Attention-BiLSTM | ✅ Full 120 | |
| MAE/RMSE/R² | ✅ Có | |
| Precision/Recall/F1 | ✅ Có | |
| Recommendation Engine | ⚠️ Prototype | Rule-based |
| Demo cảnh báo sớm | ⚠️ Dashboard HTML | Chưa có terminal demo |
| Tài liệu 5 chương | ❌ Chưa có | Cần tạo |
| Synthetic stress benchmark | ✅ Có | 6 scenarios |

## Phần nào đã hoàn thành

1. ✅ Data pipeline (NASA HTTP → windows_fp16.npz)
2. ✅ Source governance/license manifest
3. ✅ Training framework (PyTorch CUDA)
4. ✅ Baseline models (Moving Average, LSTM, GRU, TCN, Transformer)
5. ✅ TCN-Attention-BiLSTM (full 120 epoch)
6. ✅ Threshold calibration
7. ✅ Synthetic stress benchmark
8. ✅ Dashboard (HTML + JSON payload)
9. ✅ Reports (research summary, artifact manifest, gap analysis)
10. ✅ Tests (85 passed)

## Phần nào cần bổ sung

1. ⚠️ Recommendation Engine (chỉ rule-based prototype)
2. ⚠️ Demo terminal (chưa có scripts/run_demo.py)
3. ❌ Tài liệu báo cáo 5 chương
4. ❌ ARIMA baseline (optional)
5. ❌ Zanbil raw → multi-source
6. ❌ Figures (prediction_vs_actual.png, etc.)

## Phần nào không được làm vì ngoài phạm vi

- Không train multi-source khi chưa có Zanbil
- Không claim measured congestion
- Không claim cross-source
- Không dùng số demo giả
- Không gọi synthetic là real-world

## Kế hoạch triển khai từng phase

| Phase | Mục tiêu | Trạng thái |
|---|---|---|
| PHASE 0 | Inspect & Lock Scope | ✅ Đang làm |
| PHASE 1 | Clean Architecture | ⚠️ Cần cập nhật README |
| PHASE 2 | Data Pipeline | ✅ Đã hoàn thành |
| PHASE 3 | Model Implementation | ✅ Đã hoàn thành |
| PHASE 4 | Training Pipeline | ✅ Đã hoàn thành |
| PHASE 5 | Evaluation & Comparison | ✅ Đã hoàn thành |
| PHASE 6 | Recommendation Engine | ⚠️ Prototype |
| PHASE 7 | Demo/Prototype | ⚠️ Dashboard HTML only |
| PHASE 8 | Scientific Report | ❌ Chưa có |
| PHASE 9 | Testing & Verification | ✅ pytest pass |
| PHASE 10 | Final Acceptance | ⚠️ Cần checklist |

## Acceptance checklist cuối cùng

1. ✅ Có pipeline dữ liệu chuỗi thời gian
2. ✅ Có xử lý/chuẩn hóa dữ liệu 0-1
3. ✅ Có synthetic stress benchmark
4. ✅ Có data governance manifest
5. ✅ Có mô hình baseline
6. ✅ Có mô hình TCN-Attention-BiLSTM
7. ✅ Có training pipeline
8. ✅ Có evaluation MAE/RMSE/R²
9. ✅ Có so sánh mô hình
10. ⚠️ Có Recommendation Engine (prototype)
11. ⚠️ Có demo cảnh báo sớm (dashboard HTML)
12. ❌ Có biểu đồ kết quả (figures)
13. ❌ Có tài liệu báo cáo 5 chương
14. ✅ Có verification report (pytest)
15. ✅ Không có kết quả giả
16. ⚠️ Ghi rõ phần nào FULL/PARTIAL/BLOCKED
