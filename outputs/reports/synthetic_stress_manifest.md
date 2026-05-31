# Synthetic Stress Benchmark Manifest

Synthetic stress benchmark is generated from public real-data baseline.
It is used only for controlled stress evaluation, not real-world performance claims.

- Base data: C:\Users\dhp01\OneDrive\Máy tính\TCN-Attention-BiLSTM\data\processed\nasa_http_3m\windows\windows_fp16.npz
- Output: C:\Users\dhp01\OneDrive\Máy tính\TCN-Attention-BiLSTM\data\processed\synthetic_stress\windows\windows_fp16.npz
- Scenarios: flash_crowd, burst_traffic, error_surge, slow_ramp, periodic_spike, mixed_incident
- Synthetic test samples: 1800
- Positive labels: 540
- Negative labels: 1260
- Positive ratio: 0.3000
- Phase counts: `{"background": 900, "incident": 450, "pre_incident": 270, "recovery": 180}`
- Seed: 42
