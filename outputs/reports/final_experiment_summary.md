# Final Experiment Summary

- Completion state: STATE B FALLBACK COMPLETE
- Reason: Zanbil raw is missing or invalid; multi-source manifest remains NASA-only
- Source governance valid sources: ['nasa_http_1995', 'zanbil_web_logs', 'synthetic_stress_public_baseline']
- Zanbil ready: False
- Multi-source sources: ['nasa_http_1995']
- Ready for cross-source claim: False

## NASA-only Real Public Proxy Result
- MAE/RMSE/R2: 0.04279174384705035 / 0.056398649724090116 / 0.3314300649004034
- Precision/Recall/F1: 0.8125 / 0.0073654390934844195 / 0.014598540145985401

## Synthetic Stress Result
- result_type: synthetic_stress_test
- synthetic_not_real_world: true
- Best synthetic precision/recall/F1: 0.577639751552795 / 0.5166666666666667 / 0.5454545454545454

## Limitations
- NASA target is a proxy congestion score, not measured congestion.
- Synthetic stress benchmark is not real-world data and is reported separately.
- No cross-source claim is allowed until Zanbil raw is imported and processed.
- Only sources with license/citation in source governance may be used.

## Next Action
- Place a valid Zanbil raw log at data/raw/zanbil/access.log, then rerun the autonomous goal.
