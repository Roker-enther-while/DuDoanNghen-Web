# Final Artifact Manifest

- State: STATE_B_FALLBACK_COMPLETE
- Exists: 15
- Missing: 0

## models

| path | exists | size_bytes | purpose | result_type |
|---|---:|---:|---|---|
| outputs/models/full_120_tcn_attention_bilstm/best_model.pt | True | 366562 | best NASA-only full 120 model | real_public_proxy |
| outputs/models/full_120_tcn_attention_bilstm/last_model.pt | True | 366818 | last NASA-only full 120 model | real_public_proxy |

## real_public_metrics

| path | exists | size_bytes | purpose | result_type |
|---|---:|---:|---|---|
| outputs/metrics/full_120_tcn_attention_bilstm/final_metrics.json | True | 4476 | NASA-only final metrics | real_public_proxy |
| outputs/predictions/full_120_tcn_attention_bilstm/test_predictions.csv | True | 1246440 | NASA-only test predictions | real_public_proxy |

## calibration

| path | exists | size_bytes | purpose | result_type |
|---|---:|---:|---|---|
| outputs/metrics/full_120_tcn_attention_bilstm/threshold_calibration.json | True | 79423 | NASA-only threshold calibration | calibration |
| outputs/reports/full_120_tcn_attention_bilstm/threshold_calibration.md | True | 1053 | NASA-only threshold calibration report | calibration |

## synthetic

| path | exists | size_bytes | purpose | result_type |
|---|---:|---:|---|---|
| outputs/synthetic_stress_eval/full_120_tcn_attention_bilstm/metrics.json | True | 31934 | synthetic stress metrics | synthetic_stress_test |
| outputs/synthetic_stress_eval/full_120_tcn_attention_bilstm/report.md | True | 689 | synthetic stress report | synthetic_stress_test |
| outputs/synthetic_stress_eval/full_120_tcn_attention_bilstm/scenario_metrics.csv | True | 849 | synthetic scenario metrics | synthetic_stress_test |
| outputs/synthetic_stress_eval/full_120_tcn_attention_bilstm/phase_metrics.csv | True | 526 | synthetic phase metrics | synthetic_stress_test |

## governance

| path | exists | size_bytes | purpose | result_type |
|---|---:|---:|---|---|
| outputs/metrics/source_license_manifest.json | True | 7270 | source license manifest | governance |
| outputs/metrics/zanbil_readiness.json | True | 1083 | Zanbil readiness status | governance |
| outputs/metrics/multi_source_manifest.json | True | 2600 | multi-source manifest | governance |

## dashboard

| path | exists | size_bytes | purpose | result_type |
|---|---:|---:|---|---|
| outputs/web/final_state_b_dashboard_payload.json | True | 4889 | final STATE B dashboard payload | dashboard_payload |
| outputs/web/full_120_tcn_attention_bilstm/model_dashboard_payload.json | True | 9327 | updated full_120 dashboard payload | dashboard_payload |
