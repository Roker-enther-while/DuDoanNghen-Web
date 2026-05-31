# Dataset Adequacy Report

- Data: C:\Users\dhp01\OneDrive\Máy tính\TCN-Attention-BiLSTM\data\processed\multi_source_web_logs\windows\windows_fp16.npz
- Train samples: 62425
- Val samples: 13330
- Test samples: 13330
- Source count: 1
- Scenario count: 0
- Test positive count @ threshold 0.183838: 1762
- Test positive count @ threshold 0.70: 0
- Test positive rate: 0.132183
- Test volatility mean abs delta: 0.046407
- Test spike count delta>0.05: 4738
- Test quiet: False
- Target type: proxy
- Ready for training: True
- Ready for real-world claim: False
- Ready for stress benchmark: False
- Recommended next action: Proceed with transparent training/evaluation using fixed source and threshold policy.

## Warnings
- threshold_0_70_has_no_positive_cases_in_test
- target_is_proxy_not_measured_congestion
