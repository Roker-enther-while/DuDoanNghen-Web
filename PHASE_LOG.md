# PHASE_LOG.md — TCN-Attention-BiLSTM Phase History

| Date | Phase | Status | Evidence | Commit | Relation to Project Goal | Notes |
|---|---|---:|---|---|---|---|
| 2026-05-31 | Proposal goal extraction | DONE | proposal_goal_extraction.md/json | NONE (no git) | Xác định mục tiêu gốc từ đề cương/project history | NEED_CONFIRMATION vì không tìm thấy file đề cương gốc |
| UNKNOWN | Public data pipeline NASA | DONE | data/processed/nasa_http_3m/ | UNKNOWN | Tạo dataset chuỗi thời gian từ NASA HTTP logs | Artifacts now missing from disk |
| UNKNOWN | Float16/windowing | DONE | X_train [62425,60,19] float16 | UNKNOWN | Tối ưu storage và tạo sliding windows | |
| UNKNOWN | Training framework | DONE | src/training/ | UNKNOWN | PyTorch CUDA, AMP mixed precision | |
| UNKNOWN | Baseline/diagnostic models | DONE | Moving Average, LSTM, GRU, TCN, Transformer | UNKNOWN | So sánh với model chính | ARIMA/Holt-Winters chưa có |
| UNKNOWN | TCN-Attention-BiLSTM full 120 | DONE | best_model.pt, final_metrics.json | UNKNOWN | Model chính, 120 epoch, best epoch 30 | Artifacts now missing from disk |
| UNKNOWN | Threshold calibration | DONE | threshold_calibration.json | UNKNOWN | Calibrated threshold 0.05, F1 0.865596 | Phải tách biệt với p90 gốc |
| UNKNOWN | Source governance | DONE | source_license_manifest.json | UNKNOWN | NASA license/citation/provenance | |
| UNKNOWN | Synthetic stress benchmark | DONE | synthetic_stress_eval/ | UNKNOWN | 6 scenarios, 1800 samples, controlled benchmark | Không phải real-world |
| UNKNOWN | Zanbil readiness/multi-source gate | DONE | zanbil_readiness.json | UNKNOWN | Zanbil raw missing, no cross-source claim | |
| UNKNOWN | Final State B packaging | DONE | final_state_b_dashboard_payload.json | UNKNOWN | Dashboard/report minh bạch với số thật | |
| UNKNOWN | Research Defense Dashboard | DONE | research_defense_dashboard.html | UNKNOWN | 16 sections, dùng số thật, không số demo giả | |
| 2026-05-31 | Dashboard truthfulness check | DONE | test_dashboard_no_fake_claims.py | NONE (no git) | Xác nhận dashboard dùng số thật | |
| 2026-05-31 | Proposal goal alignment | DONE | proposal_goal_extraction.json, gap_analysis.json | NONE (no git) | Đối chiếu đề cương với thực nghiệm | |
| 2026-07-11 | Research Integrity Audit | DONE | research_integrity_audit.md | NONE | Audit toàn bộ artifacts, xác định gaps | CRITICAL: no data/models on disk, 3 inconsistent metric sets |
| 2026-07-11 | Proxy Target Validation | DONE | proxy_validation_report.md | NONE | Kiểm định proxy score với testbed telemetry | CRITICAL: proxy negatively correlated with congestion (r=-0.54) |
| 2026-07-11 | Ablation Infrastructure | DONE | torch_models.py, registry.py, ablation configs | NONE | Implement ablation variants + training configs | BiLSTM, Attn+BiLSTM, TCN+Attn variants added |
| 2026-07-11 | Rolling CV Infrastructure | DONE | run_rolling_cv.py | NONE | Implement rolling-origin CV with stat tests | 5-fold, Wilcoxon, Cohen's d |
| 2026-07-11 | Honest Limitations | DONE | LIMITATIONS_HONEST.md | NONE | Viết lại limitations trung thực | Covers data, model, scope limitations |
| 2026-07-11 | Integrity Update Report | DONE | RESEARCH_INTEGRITY_UPDATE.md | NONE | Before/after comparison documentation | |
| NEXT | Re-download NASA data | TODO | data/raw/nasa_http/ | TARGET | Fetch NASA Jul+Aug 1995 raw logs | `python scripts/fetch_public_data.py` |
| NEXT | Re-run data pipeline | TODO | data/processed/nasa_http_3m/ | TARGET | Generate windows FP16 NPZ | `python scripts/run_data_pipeline.py` |
| NEXT | Re-train proposed model | TODO | outputs/models/full_120_tcn_attention_bilstm/ | TARGET | Full 120 epoch training with fresh data | `python scripts/train_model.py` |
| NEXT | Run ablation study | TODO | outputs/ablation_study/ | TARGET | 6 variants x 3 seeds | `python scripts/run_ablation_study.py` |
| NEXT | Run rolling CV | TODO | outputs/rolling_cv/ | TARGET | 5-fold rolling origin | `python scripts/run_rolling_cv.py` |
| NEXT | Revise proxy formula | TODO | docs/PROXY_TARGET_DEFINITION.md | TARGET | Based on validation results | May need latency/error components |
| NEXT | Update dashboard | TODO | outputs/web/ | TARGET | Reflect new validated metrics | |
