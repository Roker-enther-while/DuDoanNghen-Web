# Threshold Calibration

- Old threshold: 0.183838
- Old precision/recall/F1: 0.812500 / 0.007365 / 0.014599
- Best threshold by validation F1: 0.050000
- Calibrated test precision/recall/F1: 0.775706 / 0.979049 / 0.865596
- Old confusion TP/FP/TN/FN: 13 / 3 / 11562 / 1752
- New confusion TP/FP/TN/FN: 9860 / 2851 / 408 / 211
- Model alert behavior after calibration: more balanced

## Alternative Thresholds
- Recall >= 0.5 candidate: {'threshold': 0.05, 'precision': 0.7257178722736545, 'recall': 0.9760472723435687, 'f1': 0.8324708635197768, 'accuracy': 0.7207051762940735, 'alert_threshold': 0.05, 'alert_positive_count_true': 9477, 'alert_positive_count_pred': 12746, 'tp': 9250, 'fp': 3496, 'tn': 357, 'fn': 227, 'warning': None}
- Balanced precision/recall candidate: {'threshold': 0.5, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 1.0, 'alert_threshold': 0.5, 'alert_positive_count_true': 0, 'alert_positive_count_pred': 0, 'tp': 0, 'fp': 0, 'tn': 13330, 'fn': 0, 'warning': 'no_positive_cases_in_y_true_for_threshold'}
