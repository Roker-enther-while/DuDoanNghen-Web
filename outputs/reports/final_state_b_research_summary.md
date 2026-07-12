# Final Experiment Summary — STATE B

## 1. Scope
- Model: TCN-Attention-BiLSTM
- Real public data: NASA HTTP 1995
- Target: proxy congestion score
- Synthetic stress: controlled benchmark
- Cross-source: not available because Zanbil raw is missing

## 2. Data Governance
- NASA source is tracked in `outputs/metrics/source_license_manifest.json`.
- Zanbil is declared in governance but raw input is missing at `data/raw/zanbil/access.log`.
- Synthetic stress is generated from a public baseline and must remain separate from real public results.
- PII policy: hash client identifiers, strip query strings, drop user-agent by default, and do not release raw logs.

## 3. Real Public Proxy Result

| Metric | Value |
|---|---:|
| MAE | 0.042792 |
| RMSE | 0.056399 |
| R² | 0.331430 |
| Precision | 0.812500 |
| Recall | 0.007365 |
| F1 | 0.014599 |
| Threshold | 0.183838 |
| TP / FP / TN / FN | 13 / 3 / 11562 / 1752 |

Regression on the proxy target is usable but not conclusive for measured congestion. Alert recall at the original p90 validation threshold is very low, so this threshold should not be presented as a strong real-world alerting result.

## 4. Threshold Calibration
- Calibrated threshold: `0.050000`
- Calibrated F1: `0.865596`
- Calibrated recall: `0.979049`

This is a threshold calibration result. It changes alert classification and must be reported separately from the original p90-threshold result.

## 5. Synthetic Stress Benchmark
- result_type: `synthetic_stress_test`
- synthetic_not_real_world: `true`
- Positive ratio: `0.300000`
- Scenarios: 6
- Checkpoint-threshold F1: `0.459344`
- Best synthetic-threshold F1: `0.545455`
- Best scenario: `periodic_spike` with F1 `0.757576`
- Worst scenario: `error_surge` with F1 `0.142857`

Synthetic stress is a controlled benchmark and not a real-world performance claim.

## 6. Why Multi-source Was Not Trained
- Missing file: `data/raw/zanbil/access.log`
- Multi-source currently contains only `nasa_http_1995`.
- `ready_for_cross_source_claim=false`.
- Training a multi-source model in this state would be misleading.

## 7. Honest Conclusion
The pipeline, governance, training, and evaluation flow are complete for NASA-only STATE B. The model learns the NASA proxy regression signal, but alerting requires threshold calibration and more diverse real data. Synthetic stress shows stronger response to periodic spikes and weaker behavior on error surge. There is no cross-source or measured-congestion conclusion yet.

## 8. Next Work
- Place a valid Zanbil raw log at `data/raw/zanbil/access.log`.
- Prepare Zanbil.
- Build NASA+Zanbil multi-source data.
- Train `multisource_full_120_tcn_attention_bilstm`.
- Compare NASA-only vs multi-source with the same governance and threshold policy.
