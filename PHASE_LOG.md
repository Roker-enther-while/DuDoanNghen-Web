# PHASE_LOG.md — TCN-Attention-BiLSTM Phase History

| Date | Phase | Status | Evidence | Commit | Relation to Project Goal | Notes |
|---|---|---:|---|---|---|---|
| 2026-05-31 | Proposal goal extraction | DONE | proposal_goal_extraction.md/json | NONE (no git) | Xác định mục tiêu gốc từ đề cương/project history | NEED_CONFIRMATION vì không tìm thấy file đề cương gốc |
| UNKNOWN | Public data pipeline NASA | DONE | data/processed/nasa_http_3m/ | UNKNOWN | Tạo dataset chuỗi thời gian từ NASA HTTP logs | |
| UNKNOWN | Float16/windowing | DONE | X_train [62425,60,19] float16 | UNKNOWN | Tối ưu storage và tạo sliding windows | |
| UNKNOWN | Training framework | DONE | src/training/ | UNKNOWN | PyTorch CUDA, AMP mixed precision | |
| UNKNOWN | Baseline/diagnostic models | DONE | Moving Average, LSTM, GRU, TCN, Transformer | UNKNOWN | So sánh với model chính | ARIMA/Holt-Winters chưa có |
| UNKNOWN | TCN-Attention-BiLSTM full 120 | DONE | best_model.pt, final_metrics.json | UNKNOWN | Model chính, 120 epoch, best epoch 30 | |
| UNKNOWN | Threshold calibration | DONE | threshold_calibration.json | UNKNOWN | Calibrated threshold 0.05, F1 0.865596 | Phải tách biệt với p90 gốc |
| UNKNOWN | Source governance | DONE | source_license_manifest.json | UNKNOWN | NASA license/citation/provenance | |
| UNKNOWN | Synthetic stress benchmark | DONE | synthetic_stress_eval/ | UNKNOWN | 6 scenarios, 1800 samples, controlled benchmark | Không phải real-world |
| UNKNOWN | Zanbil readiness/multi-source gate | DONE | zanbil_readiness.json | UNKNOWN | Zanbil raw missing, no cross-source claim | |
| UNKNOWN | Final State B packaging | DONE | final_state_b_dashboard_payload.json | UNKNOWN | Dashboard/report minh bạch với số thật | |
| UNKNOWN | Research Defense Dashboard | DONE | research_defense_dashboard.html | UNKNOWN | 16 sections, dùng số thật, không số demo giả | |
| 2026-05-31 | Dashboard truthfulness check | DONE | test_dashboard_no_fake_claims.py | NONE (no git) | Xác nhận dashboard dùng số thật | |
| UNKNOWN | Proposal goal alignment | DONE | proposal_goal_extraction.json, gap_analysis.json | NONE (no git) | Đối chiếu đề cương với thực nghiệm | |
| NEXT | PHASE 0 — Inspect & Lock Scope | TODO | outputs/reports/00_project_scope_lock.md | TARGET | Kiểm tra toàn bộ repository | Bắt đầu quy trình 11 phases |
| NEXT | PHASE 1 — Clean Architecture | TODO | README.md, requirements.txt, configs/ | TARGET | Chuẩn hóa cấu trúc dự án | |
| NEXT | PHASE 2 — Data Pipeline | TODO | scripts/build_dataset.py, synthetic stress | TARGET | Hoàn thiện pipeline dữ liệu | |
| NEXT | PHASE 3 — Model Implementation | TODO | src/models/ | TARGET | Kiểm tra/bổ sung mô hình | |
| NEXT | PHASE 4 — Training Pipeline | TODO | scripts/train_model.py | TARGET | Pipeline train nhiều mức | |
| NEXT | PHASE 5 — Evaluation & Comparison | TODO | outputs/metrics/evaluation_summary.json | TARGET | Đánh giá và so sánh mô hình | |
| NEXT | PHASE 6 — Recommendation Engine | TODO | src/recommendation/engine.py | TARGET | Hệ thống khuyến nghị phòng ngừa | |
| NEXT | PHASE 7 — Demo/Prototype | TODO | scripts/run_demo.py | TARGET | Demo cảnh báo sớm | |
| NEXT | PHASE 8 — Scientific Report | TODO | reports/chapter_*.md | TARGET | Tài liệu báo cáo 5 chương | |
| NEXT | PHASE 9 — Testing & Verification | TODO | outputs/reports/14_verification_report.md | TARGET | Kiểm thử thực tế | |
| NEXT | PHASE 10 — Final Acceptance | TODO | PROJECT_FINAL_STATUS.md | TARGET | Checklist cuối cùng | |
