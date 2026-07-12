# Model Comparison

Target is a proxy congestion score for NASA HTTP logs, not a measured congestion label.

| model | category | status | threshold | test_strategy | true_pos | pred_pos | MAE | RMSE | R2 | alert_f1 | warning |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| naive_last_value | baseline | success | quantile=0.183838 | full | 1765 | 1514 | 0.054805 | 0.073650 | -0.140120 | 0.284843 |  |
| moving_average | baseline | success | quantile=0.183838 | full | 1765 | 1006 | 0.046843 | 0.062242 | 0.185716 | 0.307470 |  |
| lstm | rnn | success | quantile=0.183838 | full | 1765 | 769 | 0.043268 | 0.056562 | 0.327548 | 0.304657 |  |
| gru | rnn | success | quantile=0.183838 | full | 1765 | 681 | 0.042702 | 0.055843 | 0.344529 | 0.285364 |  |
| tcn | convolutional | success | quantile=0.183838 | full | 1765 | 372 | 0.041912 | 0.056602 | 0.326604 | 0.206832 |  |
| transformer | attention | success | quantile=0.183838 | full | 1765 | 0 | 0.042760 | 0.057879 | 0.295869 | 0.000000 | no_positive_predictions_for_threshold |
| tcn_lstm | hybrid | success | quantile=0.183838 | full | 1765 | 158 | 0.042780 | 0.056155 | 0.337192 | 0.106084 |  |
| tcn_attention_bilstm | proposed | success | quantile=0.183838 | full | 1765 | 0 | 0.043916 | 0.058894 | 0.270956 | 0.000000 | no_positive_predictions_for_threshold |

## Best Models
- Best by RMSE: gru (0.055843)
- Best by alert F1: moving_average (0.307470)
- Best deep model by RMSE: gru (0.055843)
- Best baseline by RMSE: moving_average (0.062242)

## Automatic Checks
- Proposed beats tcn_lstm by RMSE: no
- Proposed beats transformer by RMSE: no
- Proposed beats tcn by RMSE: no
- Proposed beats best baseline by RMSE: yes

## Notes
- Quick/smoke training is intentionally small and not tuned.
- Baselines are simple controls and should not be treated as optimized forecasting models.
- If deep models underperform naive baselines, likely causes include limited data, short training, proxy target dynamics, and untuned hyperparameters.
- Proxy target note: alert threshold should be calibrated by quantile or target formula should be adjusted
