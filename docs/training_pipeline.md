# Training Pipeline

This phase reads the already prepared artifact:

```text
data/processed/nasa_http/windows/windows_fp16.npz
```

The saved artifact remains `float16`. Training code converts batches to `float32` in memory because Keras models generally train more reliably with float32. No normalization is recomputed.

## Smoke Commands

```powershell
python -m pytest -q
python scripts/train_model.py --data data/processed/nasa_http/windows/windows_fp16.npz --model lstm --config configs/training/smoke.yaml
python scripts/run_training_smoke.py --data data/processed/nasa_http/windows/windows_fp16.npz --config configs/training/smoke.yaml
```

## Models

Implemented now:

- `naive_last_value`
- `moving_average`
- `lstm`
- `gru`
- `tcn`

Registered skeletons for later:

- `transformer`
- `tcn_lstm`
- `tcn_attention_bilstm`

## Outputs

- Models: `outputs/models/{model_name}/`
- Metrics: `outputs/metrics/{model_name}_metrics.json`
- Histories: `outputs/metrics/{model_name}_history.json`
- Predictions: `outputs/predictions/{model_name}_test_predictions.csv`
- Comparison: `outputs/metrics/model_comparison.json`, `outputs/reports/model_comparison.md`

NASA target values are proxy congestion scores, not measured congestion labels. Smoke training is intended to verify the framework and first comparison path, not to produce tuned final results.
