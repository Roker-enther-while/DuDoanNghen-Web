| variant | model | RMSE_mean | F1_mean |
|---|---|---|---|
| TCN only | tcn32 | 10.505005573693827 | 0.6287559610434625 |
| TCN + BiLSTM | tcn_bilstm32_no_attn | 13.857112466573994 | 0.5416394408410609 |
| TCN + BiLSTM + Temporal Attention | tcn_bilstm32_temporal_attention | 10.776463188551206 | 0.6406800450593056 |
| TCN + Feature Attention + BiLSTM + Temporal Attention | tcn_feature_attention_bilstm_temporal_attention | 11.672862647392265 | 0.6210969147348373 |
