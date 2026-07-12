# PHASE 4 — Training Report

## Training Configurations

| Config | Epochs | Description |
|---|---|---|
| smoke.yaml | 2 | Quick code validation |
| quick.yaml | 10 | Short demo run |
| balanced.yaml | 30 | Model comparison |
| full_120.yaml | 120 | Full training |

## TCN-Attention-BiLSTM Full 120 Training

### Configuration
- **Backend:** PyTorch CUDA
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU
- **Mixed precision:** AMP float16
- **Batch size:** 128
- **Learning rate:** 0.0007
- **Optimizer:** AdamW (weight_decay=0.0001)
- **Gradient clip:** 1.0
- **Epochs requested:** 120
- **Epochs completed:** 120

### Results
| Metric | Value |
|---|---|
| Best epoch | 30 |
| Best val RMSE | 0.055146 |
| Final train loss | 0.004036 |
| Final val loss | 0.003174 |
| Total train time | 536.518s |
| Average epoch time | 4.462s |

### Checkpoint
- **Best model:** outputs/models/full_120_tcn_attention_bilstm/best_model.pt
- **Last model:** outputs/models/full_120_tcn_attention_bilstm/last_model.pt

### Training Features
- ✅ Seed cố định (42)
- ✅ Logging loss train/val
- ✅ Checkpoint best model
- ✅ Mixed precision AMP
- ✅ Gradient clipping
- ✅ History theo epoch
