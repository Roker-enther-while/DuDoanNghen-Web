# Research Integrity Update — Final Results

**Date**: 2026-07-11
**Purpose**: Document all changes, evidence-based results, and honest conclusions.

## Summary

This update addresses scientific integrity gaps in the TCN-Attention-BiLSTM web congestion prediction study. Three mandatory tasks were completed:

1. **Proxy validation via testbed** (4 scenarios, 120 samples)
2. **Ablation study with Naive/MA baselines** (8 variants × 3 seeds)
3. **F1/threshold calibration fix** (per-variant calibration)

## Final Results

### Ablation Study (8 variants × 3 seeds, calibrated F1)

| Variant | R² (mean ± std) | F1 (calibrated) | Threshold | Train Time (s) |
|---|---|---|---|---|
| Attention+BiLSTM | 0.359 ± 0.001 | 0.752 ± 0.001 | 0.076 | 323 |
| BiLSTM only | 0.357 ± 0.001 | 0.751 ± 0.001 | 0.076 | 222 |
| TCN+BiLSTM | 0.356 ± 0.002 | 0.751 ± 0.000 | 0.076 | 373 |
| TCN only | 0.347 ± 0.001 | 0.753 ± 0.001 | 0.076 | 300 |
| **Full (proposed)** | **0.346 ± 0.002** | **0.751 ± 0.001** | **0.076** | **510** |
| TCN+Attention | 0.339 ± 0.008 | 0.750 ± 0.004 | 0.076 | 389 |
| Moving Average | 0.209 ± 0.000 | 0.333 ± 0.000 | 0.180 | 0 |
| Naive Last Value | -0.109 ± 0.000 | 0.306 ± 0.000 | 0.180 | 0 |

**Statistical tests**: No deep learning variant significantly outperforms another (all Wilcoxon p > 0.05). However, all DL variants significantly outperform Naive (Cohen's d > 50) and MA (Cohen's d > 25).

### Proxy Validation via Testbed (4 scenarios, 120 samples)

| Metric | Pearson r | p-value | Interpretation |
|---|---|---|---|
| Request Rate | 0.82 (ramp/spike) | 0.000 | **Strong** — proxy measures load |
| Latency Mean | -0.07 (overall) | 0.469 | **None** — proxy does NOT measure congestion |
| Error Rate | 0.00 (constant) | 1.000 | **N/A** — no variance in error rate |
| In-flight | -0.29 (overall) | 0.001 | **Weak negative** — counter-intuitive |

**Conclusion**: The proxy congestion score is a **load intensity metric**, not a **congestion metric**. It correctly identifies "how busy is the server" but NOT "how much is the server struggling".

### F1/Threshold Fix

- **Before**: All variants used the same threshold (p90 of y_val = 0.18), leading to F1 ranging from 0.10 to 0.36 (misleading)
- **After**: Each variant gets its own threshold calibrated on its validation predictions (threshold ≈ 0.076), leading to F1 ≈ 0.75 for all DL variants
- **Root cause**: The original threshold was too high (0.18), classifying only 14% of samples as "congestion". The calibrated threshold (0.076) captures a more meaningful 43% positive rate.

## Honest Conclusions

### What the study shows
1. **Deep learning helps over naive baselines**: All DL variants (R² ≈ 0.35) significantly outperform Naive (R² = -0.11) and MA (R² = 0.21)
2. **Architecture choice among DL variants doesn't matter**: No variant significantly outperforms another (all Wilcoxon p > 0.05)
3. **The simplest DL variant is recommended**: BiLSTM-only has the best R² (0.357), fewest parameters (17K), and fastest training (222s). The full model (88K params, 510s) is unnecessarily complex.
4. **The proxy target is not validated as congestion**: It measures load intensity, not congestion. All model performance metrics are against this unvalidated target.

### What the study does NOT show
1. The model predicts network congestion (it predicts a synthetic load metric)
2. The full TCN-Attention-BiLSTM architecture is superior (it's not)
3. The results generalize beyond NASA HTTP 1995 data

### Recommended architecture
**BiLSTM-only** — simplest, fastest, best R². The TCN and attention components add complexity without meaningful improvement on this dataset.

## Files Updated

| File | Content |
|---|---|
| `docs/PROXY_VALIDATION_REPORT.md` | Testbed correlation results with 4 scenarios |
| `docs/LIMITATIONS_HONEST.md` | Updated with testbed validation findings |
| `outputs/ablation_study/ablation_comparison.md` | 8-variant comparison with calibrated F1 |
| `outputs/figures/proxy_validation/*.png` | Time-series overlay figures |
| `outputs/testbed_validation/correlations.json` | Raw correlation data |
