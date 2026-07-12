# Rolling-Origin Cross-Validation Results

**Folds**: 5
**Models**: naive_last_value, moving_average, bilstm, tcn_attention_bilstm
**Target**: proxy_congestion_score (synthetic composite, NOT measured congestion).

## Per-Model Results (mean +/- std across folds)

| Model | MAE | RMSE | R² | F1 |
|---|---|---|---|---|
| bilstm | 0.051030 +/- 0.018406 | 0.063723 +/- 0.024533 | -0.012233 +/- 0.041725 | 0.000000 +/- 0.000000 |
| moving_average | 0.052506 +/- 0.019358 | 0.069492 +/- 0.024541 | -0.226944 +/- 0.109895 | 0.000000 +/- 0.000000 |
| tcn_attention_bilstm | 0.068114 +/- 0.037641 | 0.079737 +/- 0.039779 | -0.529693 +/- 0.512278 | 0.000000 +/- 0.000000 |
| naive_last_value | 0.061433 +/- 0.023849 | 0.081864 +/- 0.029006 | -0.698925 +/- 0.120724 | 0.000000 +/- 0.000000 |

## Statistical Tests (Proposed vs. Baselines)

| Baseline | R² Diff | Wilcoxon p | Cohen's d | Significant? |
|---|---|---|---|---|
| naive_last_value | +0.169232 | 0.625000 | 0.2571 | No |
| moving_average | -0.302748 | 0.625000 | -0.4545 | No |
| bilstm | -0.517459 | 0.312500 | -0.8673 | No |

## Notes
- Rolling-origin evaluation ensures temporal validity.
- Statistical tests use Wilcoxon signed-rank (non-parametric).
- Negative R² means the model is worse than predicting the mean.
- All results use chronological splits (no future data leakage).
