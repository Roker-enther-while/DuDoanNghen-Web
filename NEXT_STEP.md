# NEXT_STEP.md — TCN-Attention-BiLSTM Next Execution Context

## PROJECT GOAL SNAPSHOT

Mục tiêu cuối cùng: biến repository này thành một hệ thống nghiên cứu + thực nghiệm + demo hoàn chỉnh, đúng phạm vi đề cương, có thể bảo vệ/trình bày được trước giảng viên.

## Current State

- Data: NASA HTTP 1995, proxy congestion score
- Model: TCN-Attention-BiLSTM, full 120 epoch
- MAE: 0.042792, RMSE: 0.056399, R²: 0.331430
- Threshold calibration: 0.05, F1: 0.865596, Recall: 0.979049
- Synthetic stress: 6 scenarios, best F1: 0.545455
- Dashboard: Research Defense HTML + JSON payload với số thật
- Reports: research summary, artifact manifest, gap analysis, revision recommendations
- pytest: 86 passed

## QUY TRÌNH THỰC HIỆN BẮT BUỘC

### PHASE 0 — INSPECT & LOCK_SCOPE
- Liệt kê cấu trúc thư mục
- Xác định ngôn ngữ/framework chính
- Xác định scripts/configs/data/outputs/tests/models/dashboard
- Tạo outputs/reports/00_project_scope_lock.md
- Tạo outputs/metrics/00_project_scope_lock.json

### PHASE 1 — CLEAN ARCHITECTURE & REPRODUCIBILITY
- README.md mô tả đề tài, mục tiêu, cách chạy
- requirements.txt hoặc pyproject.toml
- configs/ smoke/quick/balanced/full
- scripts/ entrypoint rõ ràng
- data/README.md
- .gitignore

### PHASE 2 — DATA PIPELINE
- Input adapter
- Preprocessing: resampling, features, MinMax 0-1, sliding windows, split
- Data governance: manifest, license, provenance
- Synthetic stress benchmark: 6 scenarios

### PHASE 3 — MODEL IMPLEMENTATION
- Baseline: Moving Average, Naive
- LSTM/GRU
- TCN
- TCN-Attention-BiLSTM (chính)
- Registry

### PHASE 4 — TRAINING PIPELINE
- Smoke/quick/balanced/full configs
- Seed, logging, checkpoint, mixed precision
- History theo epoch

### PHASE 5 — EVALUATION & COMPARISON
- MAE, RMSE, R²
- Precision, Recall, F1, confusion matrix
- So sánh mô hình
- Biểu đồ

### PHASE 6 — RECOMMENDATION ENGINE
- Risk level: normal/warning/critical
- Recommended actions
- Explanation

### PHASE 7 — DEMO / PROTOTYPE
- Terminal demo
- Dashboard HTML
- --sample synthetic --model best --explain --save-report

### PHASE 8 — SCIENTIFIC REPORT MATERIALS
- Chương 1-5 outline/content
- Final summary, talking points, limitations

### PHASE 9 — TESTING & VERIFICATION
- pytest
- Chạy scripts thực tế
- Verification report

### PHASE 10 — FINAL ACCEPTANCE
- Checklist 16 mục
- FULL/PARTIAL/BLOCKED status

## Next Step

Chạy PHASE 0: Inspect & Lock Scope. Kiểm tra toàn bộ repository, tạo scope lock report.

## Restrictions

- Không train lại
- Không bịa kết quả
- Không dùng số demo giả
- Không claim measured congestion
- Không claim cross-source khi chưa có Zanbil

## Done Condition

Khi tất cả 11 phases hoàn thành, có evidence thật, pytest pass, dashboard dùng số thật.

## Blocker Condition

BLOCKER nếu: artifact metrics thiếu, payload parse lỗi, test fail không sửa được.
