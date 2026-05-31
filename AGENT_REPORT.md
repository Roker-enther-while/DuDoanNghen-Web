# AGENT_REPORT.md — TCN-Attention-BiLSTM Current Agent Report

## PROJECT GOAL SUMMARY

Dự đoán nghẽn hệ thống web bằng mô hình TCN-Attention-BiLSTM dựa trên chuỗi thời gian từ NASA HTTP logs. Target là proxy congestion score, không phải measured congestion thật.

Mục tiêu cuối cùng: biến repository này thành một hệ thống nghiên cứu + thực nghiệm + demo hoàn chỉnh, đúng phạm vi đề cương, có thể bảo vệ/trình bày được trước giảng viên.

## CURRENT STATE

Final State B — NASA-only real public proxy result + calibrated threshold + synthetic stress benchmark + transparent limitations + Research Defense Dashboard.

## DONE

- Data pipeline: NASA HTTP → windows_fp16.npz
- Source governance/license manifest
- Training framework: PyTorch CUDA
- Baseline models: Moving Average, LSTM, GRU, TCN, Transformer
- TCN-Attention-BiLSTM: full 120 epoch, best epoch 30
- Threshold calibration: threshold 0.05, F1 0.865596, Recall 0.979049
- Synthetic stress benchmark: 6 scenarios, 1800 samples
- Dashboard payload: số thật, không số demo giả
- Research Defense Dashboard HTML: 16 sections, dùng số thật
- Reports: research summary, artifact manifest, gap analysis, revision recommendations
- Runbook
- Agent files: CLAUDE.md, NEXT_STEP.md, AGENT_REPORT.md, PHASE_LOG.md

## CHANGED FILES

- outputs/reports/proposal_goal_extraction.md
- outputs/metrics/proposal_goal_extraction.json
- outputs/reports/proposal_vs_current_gap_analysis.md
- outputs/metrics/proposal_vs_current_gap_analysis.json
- outputs/reports/proposal_revision_recommendations.md
- outputs/reports/data_quantity_analysis.md
- outputs/web/final_state_b_dashboard_payload.json
- outputs/web/research_defense_dashboard.html
- CLAUDE.md
- NEXT_STEP.md
- AGENT_REPORT.md
- PHASE_LOG.md

## COMMANDS RUN

- python -m pytest -q → 86 passed

## VERIFICATION RESULT

- pytest: 86 passed
- JSON validation: no NaN/Inf
- Dashboard payload: real numbers confirmed
- Research Defense Dashboard: all sections present, no fake metrics
- All reports exist

## COMMIT

No git repo in project directory.

## REMAINING ISSUES

- Zanbil raw missing: data/raw/zanbil/access.log
- Multi-source not available
- ARIMA/Holt-Winters not implemented
- TurboQuant not implemented
- Recommendation Engine: rule-based prototype only
- Demo terminal script chưa tạo

## RISKS

- Nếu dùng số demo giả sẽ vi phạm tính minh bạch
- Nếu claim measured congestion sẽ sai vì target là proxy
- Nếu claim cross-source khi chưa có Zanbil sẽ误导

## NEXT SMALL STEP

Chạy PHASE 0: Inspect & Lock Scope. Kiểm tra toàn bộ repository, tạo scope lock report.

## BLOCKER

Current blocker: NONE.
