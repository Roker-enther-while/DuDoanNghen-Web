# PHASE 3 — Model Architecture Report

## Models Implemented

### 1. Baseline Models

#### Naive Last Value
- **Type:** Statistical baseline
- **Logic:** Predict last value in window
- **Use case:** Lower bound comparison

#### Moving Average
- **Type:** Statistical baseline
- **Logic:** Predict moving average of window
- **Use case:** Lower bound comparison

### 2. Deep Learning Models

#### LSTM (Long Short-Term Memory)
- **Type:** Recurrent Neural Network
- **Layers:** LSTM + Dense
- **Use case:** Sequence modeling baseline

#### GRU (Gated Recurrent Unit)
- **Type:** Recurrent Neural Network
- **Layers:** GRU + Dense
- **Use case:** Sequence modeling baseline

#### TCN (Temporal Convolutional Network)
- **Type:** Convolutional
- **Layers:** Causal convolutions with dilations
- **Use case:** Temporal feature extraction

#### Transformer
- **Type:** Attention-based
- **Layers:** Self-attention + Feed-forward
- **Use case:** Attention baseline

#### TCN-LSTM
- **Type:** Hybrid
- **Layers:** TCN + LSTM
- **Use case:** Hybrid baseline

### 3. Proposed Model: TCN-Attention-BiLSTM

#### Architecture
```
Input [batch, 60, 19]
  → TCN (dilated causal convolutions)
    → Conv1d(kernel=3, dilation=1) + ReLU + Dropout
    → Conv1d(kernel=3, dilation=2) + ReLU + Dropout
    → Conv1d(kernel=3, dilation=4) + ReLU + Dropout
    → Conv1d(kernel=3, dilation=8) + ReLU + Dropout
  → Multi-Head Self-Attention
    → 2 heads, key_dim=16
  → Bidirectional LSTM
    → 32 units forward + 32 units backward
  → Dense(64) + ReLU + Dropout
  → Dense(1) → prediction
```

#### Hyperparameters
| Parameter | Value |
|---|---|
| TCN filters | 64 |
| TCN kernel size | 3 |
| TCN dilations | [1, 2, 4, 8] |
| Attention heads | 2 |
| Attention key dim | 16 |
| LSTM units | 32 |
| Dense units | 64 |
| Dropout | 0.15 |
| Learning rate | 0.0007 |
| Optimizer | AdamW |
| Weight decay | 0.0001 |
| Batch size | 128 |
| Epochs | 120 |

#### Design Rationale
1. **TCN:** Trích xuất đặc trưng thời gian bằng convolution/dilation
2. **Attention:** Học trọng số thời điểm/đặc trưng quan trọng
3. **BiLSTM:** Học phụ thuộc chuỗi hai chiều trong cửa sổ dữ liệu

## Framework

- **Backend:** PyTorch CUDA
- **Mixed precision:** AMP float16
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU
