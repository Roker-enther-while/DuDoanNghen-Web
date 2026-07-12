# Ablation Study Results

Target: proxy_congestion_score (synthetic composite, NOT measured congestion).
All results on chronological test split of NASA HTTP 1995 data.

## Metrics (mean +/- std across seeds)

| Variant | MAE | RMSE | R² | F1 (shared) | F1 (calibrated) | Threshold | Precision | Recall | Train Time (s) |
|---|---|---|---|---|---|---|---|---|---|
| attention_bilstm | 0.043840 +/- 0.000126 | 0.055793 +/- 0.000058 | 0.359344 +/- 0.001337 | 0.295835 +/- 0.017876 | 0.752244 +/- 0.001360 | 0.0764 | 0.702877 | 0.809184 | 349.4 |
| bilstm_only | 0.044594 +/- 0.000125 | 0.055890 +/- 0.000027 | 0.357113 +/- 0.000626 | 0.358975 +/- 0.008879 | 0.750860 +/- 0.000811 | 0.0764 | 0.667949 | 0.857337 | 221.9 |
| tcn_bilstm | 0.043681 +/- 0.000158 | 0.055950 +/- 0.000095 | 0.355719 +/- 0.002195 | 0.156510 +/- 0.055729 | 0.750907 +/- 0.000141 | 0.0764 | 0.670258 | 0.853696 | 375.2 |
| tcn_only | 0.043130 +/- 0.000119 | 0.056335 +/- 0.000034 | 0.346840 +/- 0.000786 | 0.085996 +/- 0.056672 | 0.752859 +/- 0.001000 | 0.0764 | 0.716299 | 0.793408 | 300.4 |
| full ** | 0.043690 +/- 0.000150 | 0.056370 +/- 0.000091 | 0.346011 +/- 0.002111 | 0.100820 +/- 0.011365 | 0.751235 +/- 0.001036 | 0.0764 | 0.687100 | 0.828795 | 509.8 |
| tcn_attention | 0.043383 +/- 0.000170 | 0.056675 +/- 0.000333 | 0.338909 +/- 0.007770 | 0.142323 +/- 0.099885 | 0.749897 +/- 0.003865 | 0.0760 | 0.703850 | 0.802502 | 422.9 |
| moving_average | 0.046750 +/- 0.000000 | 0.061983 +/- 0.000000 | 0.209297 +/- 0.000000 | 0.332842 +/- 0.000000 | 0.332842 +/- 0.000000 | 0.1804 | 0.445320 | 0.265726 | 0.0 |
| naive_last_value | 0.054687 +/- 0.000000 | 0.073419 +/- 0.000000 | -0.109380 +/- 0.000000 | 0.305704 +/- 0.000000 | 0.305704 +/- 0.000000 | 0.1804 | 0.329484 | 0.285126 | 0.0 |

## Statistical Tests (Full vs. Each Variant)

| Variant | R² Difference | Wilcoxon p-value | Cohen's d | Significant (p<0.05)? |
|---|---|---|---|---|
| naive_last_value | +0.455391 | 0.250000 | 176.1001 | No |
| moving_average | +0.136714 | 0.250000 | 52.8673 | No |
| tcn_only | -0.000829 | 1.000000 | -0.2418 | No |
| bilstm_only | -0.011102 | 0.250000 | -4.9274 | No |
| attention_bilstm | -0.013332 | 0.250000 | -5.7104 | No |
| tcn_bilstm | -0.009708 | 0.250000 | -5.0772 | No |
| tcn_attention | +0.007102 | 0.250000 | 0.7152 | No |

## Notes
- **Bold** = proposed full model.
- R² values can be negative (worse than predicting the mean).
- Statistical tests use Wilcoxon signed-rank (non-parametric, suitable for small n).
- Cohen's d > 0.8 = large effect, > 0.5 = medium, > 0.2 = small.
- If the full model does not significantly outperform simpler variants,
  the simpler variant is the recommended architecture (parsimony principle).
