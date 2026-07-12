# PHASE A — Gap Closure Plan

## Trạng thái hiện tại

### Đã có
- ✅ outputs/metrics/full_120_tcn_attention_bilstm/final_metrics.json
- ✅ outputs/web/research_defense_dashboard.html
- ✅ outputs/reports/15_final_acceptance_checklist.md
- ✅ PROJECT_FINAL_STATUS.md
- ✅ 83 tests passing

### Còn thiếu
- ❌ scripts/generate_figures.py
- ❌ scripts/run_demo.py
- ❌ data/raw/zanbil/access.log
- ❌ outputs/figures/*.png
- ❌ outputs/metrics/18_terminal_demo_output.json
- ❌ outputs/reports/18_terminal_demo_report.md

## Kế hoạch xử lý

### PHASE B — Tạo Figures PNG
- Tạo scripts/generate_figures.py
- Tạo 6 figures từ data hiện có
- Chạy script

### PHASE C — Tạo Terminal Demo
- Tạo scripts/run_demo.py
- Demo flow: load data → predict → risk score → recommendation
- Chạy demo

### PHASE D — ARIMA Baseline
- Giữ OPTIONAL/PARTIAL
- Giải thích: dùng Naive/Moving Average làm baseline chính

### PHASE E — Zanbil Raw
- Giữ BLOCKED
- Tạo hướng dẫn rõ ràng

### PHASE F — Hook Error
- Ghi nhận: đây là Claude/agent hook error, không phải code project

### PHASE G — Cập nhật Dashboard
- Thêm figures links
- Thêm demo terminal status

### PHASE H — Final Verification
- Chạy lại pytest
- Chạy generate_figures.py
- Chạy run_demo.py

### PHASE I — Cập nhật Final Status
- Cập nhật PROJECT_FINAL_STATUS.md
- Cập nhật checklist

## Commands sẽ chạy

```bash
python scripts/generate_figures.py
python scripts/run_demo.py --sample synthetic --model best --explain --save-report
python -m pytest -q
```
