# TCN-Attention-BiLSTM Small Tuning

| option | status | best_val_rmse | test_rmse | test_mae | model_path |
|---|---|---:|---:|---:|---|
| option_a | success | 0.056784 | 0.056087 | 0.043213 | outputs\tuning\tcn_attention_bilstm\option_a\models\tcn_attention_bilstm\model.pt |
| option_b | success | 0.057012 | 0.056164 | 0.042750 | outputs\tuning\tcn_attention_bilstm\option_b\models\tcn_attention_bilstm\model.pt |
| option_c | success | 0.056708 | 0.056177 | 0.043677 | outputs\tuning\tcn_attention_bilstm\option_c\models\tcn_attention_bilstm\model.pt |

- Best by validation RMSE: option_c
- Diagnostic tuning only, not full hyperparameter search.
