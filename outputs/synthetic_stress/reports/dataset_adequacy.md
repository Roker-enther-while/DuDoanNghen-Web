# Dataset Adequacy Report

- Data: C:\Users\dhp01\OneDrive\Máy tính\TCN-Attention-BiLSTM\data\processed\synthetic_stress\windows\windows_fp16.npz
- Train samples: 62425
- Val samples: 13330
- Test samples: 1800
- Source count: 1
- Scenario count: 6
- Test positive count @ threshold 0.183838: 756
- Test positive count @ threshold 0.70: 388
- Test positive rate: 0.420000
- Test volatility mean abs delta: 0.057829
- Test spike count delta>0.05: 661
- Test quiet: False
- Target type: synthetic_label
- Ready for training: True
- Ready for real-world claim: False
- Ready for stress benchmark: True
- Recommended next action: Proceed with transparent training/evaluation using fixed source and threshold policy.

## Synthetic Labels
- Positive count: 540
- Negative count: 1260
- Positive ratio: 0.300000
- Phase distribution: `{"background": 900, "incident": 450, "pre_incident": 270, "recovery": 180}`
- Scenario distribution: `{"flash_crowd": 300, "burst_traffic": 300, "error_surge": 300, "slow_ramp": 300, "periodic_spike": 300, "mixed_incident": 300}`
- Severity distribution: `{"min": 0.0, "max": 0.9798981436194152, "mean": 0.26013568774909246, "std": 0.3372159171526539}`

## Warnings
- synthetic_data_must_not_be_reported_as_real_world
