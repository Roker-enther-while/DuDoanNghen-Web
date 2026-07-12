# FINAL HARD AUDIT REPORT

**Date:** 2026-05-31
**Auditor:** Claude Code Agent
**Project:** TCN-Attention-BiLSTM Web Congestion Prediction

## PHASE 1 — FILE EXISTENCE

| File | Expected | Actual | Status |
|---|---|---|---|
| PROJECT_FINAL_STATUS.md | Exist, non-empty | 3090 bytes | PASS |
| README.md | Exist, non-empty | 2889 bytes | PASS |
| outputs/web/research_defense_dashboard.html | Exist, non-empty | 23496 bytes | PASS |
| outputs/metrics/full_120_tcn_attention_bilstm/final_metrics.json | Exist, non-empty | 4494 bytes | PASS |
| outputs/reports/15_final_acceptance_checklist.md | Exist, non-empty | 2361 bytes | PASS |
| outputs/reports/22_final_gap_closure_verification.md | Exist, non-empty | 1822 bytes | PASS |
| outputs/reports/18_terminal_demo_report.md | Exist, non-empty | 635 bytes | PASS |
| outputs/metrics/18_terminal_demo_output.json | Exist, non-empty | 696 bytes | PASS |
| data/raw/zanbil/README_ZANBIL_INPUT.md | Exist, non-empty | 1063 bytes | PASS |
| outputs/figures/prediction_vs_actual.png | Exist, non-empty | 270464 bytes | PASS |
| outputs/figures/error_distribution.png | Exist, non-empty | 65987 bytes | PASS |
| outputs/figures/model_comparison_rmse.png | Exist, non-empty | 63789 bytes | PASS |
| outputs/figures/training_curves.png | Exist, non-empty | 130174 bytes | PASS |
| outputs/figures/early_warning_timeline.png | Exist, non-empty | 349201 bytes | PASS |
| outputs/figures/synthetic_stress_scenarios.png | Exist, non-empty | 73570 bytes | PASS |

**Result:** 15/15 PASS

## PHASE 2 — COMMAND AUDIT

| Command | Status | Output |
|---|---|---|
| python -m pytest -q | PASS | 83 passed in 9.36s |
| python scripts/generate_figures.py | PASS | 6 figures generated |
| python scripts/run_demo.py --sample synthetic --model best --explain --save-report | PASS | JSON + MD saved |

**Result:** 3/3 PASS

## PHASE 3 — METRICS CONSISTENCY

| Metric | JSON Value | Status |
|---|---|---|
| model | tcn_attention_bilstm | PASS |
| mae | 0.043053 | PASS |
| rmse | 0.056036 | PASS |
| r2 | 0.339994 | PASS |
| threshold | 0.183838 | PASS |
| train_time | 1092.5s | PASS |

**Result:** PASS

## PHASE 4 — DEMO OUTPUT AUDIT

| Field | Expected | Actual | Status |
|---|---|---|---|
| demo_timestamp | Exist | 20260531_161549 | PASS |
| sample_type | Exist | synthetic | PASS |
| model | Exist | tcn_attention_bilstm | PASS |
| predicted_score | Exist | 0.642102 | PASS |
| risk_level | Exist | CRITICAL | PASS |
| reason | Exist | Score significantly above threshold | PASS |
| recommended_actions | Exist | 5 actions | PASS |
| explanation | Exist | Exists | PASS |
| metrics | Exist | mae/rmse/r2 | PASS |

**Result:** PASS

## PHASE 5 — DASHBOARD AUDIT

| Content | Count | Status |
|---|---|---|
| TCN-Attention-BiLSTM mentions | 5 | PASS |
| NASA HTTP 1995 mentions | 5 | PASS |
| proxy congestion mentions | 4 | PASS |
| Zanbil mentions | 6 | PASS |
| Demo/READY mentions | 4 | PASS |

**Result:** PASS

## FINAL DECISION

```
AUDIT RESULT: READY FOR DEMO
```

## PASS

- ✅ File existence: 15/15
- ✅ Command audit: 3/3
- ✅ Metrics consistency: PASS
- ✅ Demo output: PASS
- ✅ Dashboard: PASS
- ✅ pytest: 83 passed
- ✅ Figures: 6 generated
- ✅ Demo: working with v2 metrics

## FAIL/BLOCKED

- ❌ Zanbil raw: BLOCKED (data/raw/zanbil/access.log missing)
  - Has guide: data/raw/zanbil/README_ZANBIL_INPUT.md
  - Not blocking main demo

## EVIDENCE

### Commands Run
```bash
python -m pytest -q → 83 passed
python scripts/generate_figures.py → 6 figures
python scripts/run_demo.py --sample synthetic --model best --explain --save-report → OK
```

### Metrics (from v2 run)
```json
{
  "model": "tcn_attention_bilstm",
  "mae": 0.043053,
  "rmse": 0.056036,
  "r2": 0.339994,
  "train_time_seconds": 1092.5
}
```

### Files Created
- outputs/reports/23_final_hard_audit_report.md
- outputs/metrics/23_final_hard_audit_report.json

## NEXT COMMANDS

```bash
# 1. Mở dashboard
start outputs/web/research_defense_dashboard.html

# 2. Chạy demo
python scripts/run_demo.py --sample synthetic --model best --explain --save-report

# 3. Xem metrics
cat outputs/metrics/full_120_v2_tcn_attention_bilstm/final_metrics.json

# 4. Chạy tests
python -m pytest -q

# 5. Xem figures
start outputs/figures/prediction_vs_actual.png
start outputs/figures/model_comparison_rmse.png

# 6. Xem demo report
cat outputs/reports/18_terminal_demo_report.md
```
