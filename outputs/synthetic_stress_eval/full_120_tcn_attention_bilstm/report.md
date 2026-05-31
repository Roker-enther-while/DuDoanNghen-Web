# Synthetic Stress Evaluation

- result_type: synthetic_stress_test
- synthetic_not_real_world: true
- not_mixed_with_real_public_result: true
- Model: outputs/models/full_120_tcn_attention_bilstm/best_model.pt
- Samples: 1800
- Positive ratio: 0.300000
- Checkpoint threshold: 0.183838
- Checkpoint precision/recall/F1: 1.000000 / 0.298148 / 0.459344
- Best synthetic F1 threshold: 0.150000
- Best synthetic precision/recall/F1: 0.577640 / 0.516667 / 0.545455
- Best scenario by F1: periodic_spike (0.757576)
- Worst scenario by F1: error_surge (0.142857)

Synthetic stress results are controlled benchmark results only and must not be reported as real-world performance.
