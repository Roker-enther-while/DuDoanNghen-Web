# Chương 3: Mô Hình Đề Xuất

## 3.1 Bài toán dự đoán nghẽn

### Định nghĩa bài toán
Cho chuỗi thời gian X = {x₁, x₂, ..., xₜ} với mỗi xᵢ là vector đặc trưng tại thời điểm i, dự đoán giá trị yₜ₊₁ (congestion score) tại thời điểm t+1.

### Input
- Window size: 60 time steps
- Features: 19 đặc trưng
- Shape: [batch_size, 60, 19]

### Output
- Prediction: scalar value [0, 1]
- Congestion score proxy

## 3.2 Pipeline telemetry

```
Raw HTTP Logs
    ↓
Parse (regex)
    ↓
Aggregate (1-min window)
    ↓
Feature Engineering
    ↓
Normalize (MinMax 0-1)
    ↓
Sliding Windows (lookback=60)
    ↓
Train/Val/Test Split
```

## 3.3 Kiến trúc TCN-Attention-BiLSTM

### Tổng quan
```
Input [batch, 60, 19]
    ↓
TCN Block (4 layers)
    ↓
Multi-Head Self-Attention
    ↓
Bidirectional LSTM
    ↓
Dense Layer
    ↓
Output [batch, 1]
```

### TCN Block
- 4 convolutional layers
- Kernel size: 3
- Dilations: [1, 2, 4, 8]
- Filters: 64
- Activation: ReLU
- Dropout: 0.15

### Multi-Head Self-Attention
- Heads: 2
- Key dimension: 16
- Học trọng số cho từng time step

### Bidirectional LSTM
- Units: 32 (forward) + 32 (backward)
- Học phụ thuộc hai chiều

### Dense Layer
- Units: 64
- Activation: ReLU
- Dropout: 0.15
- Output: Linear (1 unit)

## 3.4 Recommendation Engine

### Logic
```
Input: predicted_score, threshold, phase
    ↓
Risk Level Classification
    ↓
Action Recommendation
    ↓
Output: risk_level, actions, explanation
```

### Risk Levels
- Normal: score < threshold
- Watch: score gần threshold
- Warning: score ≥ threshold
- Critical: score cao + persistent

## 3.5 Sơ đồ luồng xử lý

```
[Data Collection] → [Preprocessing] → [Model Training] → [Evaluation]
                                                        ↓
[Raw Logs] → [Parse] → [Features] → [Normalize] → [Windows]
                                                        ↓
[TCN-Attention-BiLSTM] → [Prediction] → [Risk Score] → [Alert]
                                                        ↓
[Recommendation Engine] → [Actions] → [Dashboard]
```
