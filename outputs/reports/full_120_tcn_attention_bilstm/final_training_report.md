# Full 120 Training Report

## Run Setup
- Data path: data\processed\nasa_http_3m\windows\windows_fp16.npz
- X_train shape/dtype: [62425, 60, 19] / float16
- X_val shape/dtype: [13330, 60, 19] / float16
- X_test shape/dtype: [13330, 60, 19] / float16
- Backend: PyTorch CUDA (NVIDIA GeForce RTX 4060 Laptop GPU)
- Epochs requested/completed: 120 / 120
- Batch size: 128
- Mixed precision: True
- Threshold: quantile 0.9 from val = 0.183838

## Training Summary
- Best epoch: 30
- Best val RMSE: 0.055146
- Final train loss: 0.004036
- Final val loss: 0.003174
- Total train time seconds: 536.518
- Average epoch time seconds: 4.462

## Test Result
- MAE: 0.042792
- RMSE: 0.056399
- R2: 0.331430
- Precision: 0.812500
- Recall: 0.007365
- F1: 0.014599
- Accuracy: 0.868342
- TP/FP/TN/FN: 13 / 3 / 11562 / 1752
- Positive true/pred: 1765 / 16

## Diagnostic Comparison
- Diagnostic tcn_attention_bilstm: RMSE 0.056104, F1 0.284050
- Full 120 tcn_attention_bilstm: RMSE 0.056399, F1 0.014599

## Limitations
- NASA target is a proxy congestion score, not a measured congestion label.
- NASA HTTP logs do not include CPU, memory, or response-time telemetry.
- This run uses one seed: 42.

## Next Step
- Run stability seeds if needed, or tune threshold/model if recall is too low.
