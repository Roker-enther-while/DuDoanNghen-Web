# README, Paper Summary, and Long-Run Testbed Plan

Created: 2026-05-17.

## 1. README Claims That Need Correction

The current `README.md` contains claims that are too strong or not supported by the current artifacts:

- Line 1 uses a branded primary name: `WebTAB`.
- Line 4 says the repo is an official implementation of a branded framework and a `state-of-the-art (SOTA)` solution.
- Line 9 introduces the model under the branded name instead of a neutral architecture description.
- Line 11 claims:
  - `Evaluation against SOTA benchmarks`.
  - `accuracy of 96.30%`.
  - `R2 Score of 0.971`.
  - `25.3% reduction in energy consumption`.
- Line 18 says the model provides a robust low-latency engine for next-generation cloud-native platforms.
- Lines 25, 31, 45, 54, 62, 69, 73, 80, 85, 96, and 108 keep using the branded name or claims such as `SOTA`, `superior foundation`, and ROI/energy improvement.
- The baseline table in the current README does not match the current generated artifacts in `paper_artifacts/model_selection_seed_*/model_selection_metrics.csv`.
- The current artifacts show that simple baselines are competitive:
  - Best RMSE across seeds is `moving_average` with RMSE `9.589731`.
  - Best F1 is `lstm32` for seed `42`, but `persistence` for seeds `123` and `2026`.
  - The full `TCN + Feature Attention + BiLSTM + Temporal Attention` variant has RMSE mean `11.672863` and F1 mean `0.621097` in `paper_artifacts/ablation_architecture.csv`, so the README must not claim it wins absolutely.

## 2. Existing Testbed Results

Existing short testbed output:

- `Data/testbed/prometheus_metrics.csv`: 109 rows.
- `Data/testbed/testbed_labeled.csv`: 109 rows.
- Label distribution from `Data/testbed/testbed_labeled.csv`:
  - label `0`: 60 rows.
  - label `1`: 49 rows.
- `Data/testbed/testbed_harmonized.csv`: 109 rows.
- SQLite table `testbed_pool`: 109 rows, according to `docs/testbed_run_summary_20260517.md`.

Generated artifact groups:

- `paper_artifacts/tables/threshold_search.md`.
- `paper_artifacts/tables/imputation_report.md`.
- `paper_artifacts/tables/arima_behavior_analysis.md`.
- `paper_artifacts/tables/recommendation_engine_audit.md`.
- `paper_artifacts/tables/stability_test.md`.
- `paper_artifacts/tables/ablation_architecture.md`.
- `paper_artifacts/figures/*.png`.
- `paper_artifacts/model_selection_seed_42/`.
- `paper_artifacts/model_selection_seed_123/`.
- `paper_artifacts/model_selection_seed_2026/`.
- Locust short-run CSVs for `normal`, `gradual`, `spike`, `stress`, and `recovery`.

Artifact scan for `not_run`, `failed`, `missing`, and `error`:

- `not_run` remains only in optional old holdout tables:
  - `paper_artifacts/model_selection_seed_42/tables/table_14_holdout_old20_results.md`.
  - `paper_artifacts/model_selection_seed_123/tables/table_14_holdout_old20_results.md`.
  - `paper_artifacts/model_selection_seed_2026/tables/table_14_holdout_old20_results.md`.
- `missing` appears as expected column text in `imputation_report`.
- `error` appears in expected contexts such as `error_rate` and recommendation text.
- No required testbed artifact is currently marked `failed`.

## 3. README Rewrite Plan

The README will be rewritten in a neutral research style:

- Title: `Web Congestion Prediction with Multivariate Time-Series AI`.
- Describe the repo as a research prototype for student research, not as a production-ready or official framework.
- Replace branded model naming with:
  - `TCN + Feature Attention + BiLSTM + Temporal Attention`.
  - `hybrid deep learning model for multivariate time series`.
- Remove unsupported claims:
  - no `SOTA` claim.
  - no `96.30% accuracy`.
  - no `R2 0.971`.
  - no `25.3% energy reduction`.
  - no absolute `best model` claim unless tied to a specific artifact and metric.
- State clearly that the Docker testbed is production-like laboratory data, not real production traffic.
- State that on the short testbed artifacts, persistence and moving average are strong baselines.
- Emphasize reproducible pipeline, monitoring, labeling, multi-seed stability, ablation, and honest artifact generation over headline metrics.

Required README sections:

- Overview.
- Key Features.
- Repository Structure.
- Testbed Components.
- How to Run Testbed.
- How to Run Load Profiles.
- How to Collect Prometheus Metrics.
- How to Label Congestion.
- How to Run Paper Experiments.
- Current Evidence and Limitations.
- Reproducibility Notes.
- Citation / Paper Artifacts.
- License / Academic Use.

## 4. Long-Run Plan

The long-run experiment will:

- Run five Locust profiles:
  - `normal`.
  - `gradual`.
  - `spike`.
  - `stress`.
  - `recovery`.
- Run at least 30 minutes per profile. If the machine remains stable, this can be extended to 60 minutes per profile in a later run.
- Use a timestamped output directory:
  - `Data/testbed/longrun_<timestamp>/`.
  - `paper_artifacts/longrun_<timestamp>/`.
  - `docs/testbed_longrun_summary_<timestamp>.md`.
- Avoid overwriting the short-run data in:
  - `Data/testbed/prometheus_metrics.csv`.
  - `Data/testbed/testbed_labeled.csv`.
  - `Data/testbed/testbed_harmonized.csv`.
  - top-level `paper_artifacts/`.
- Keep long-run Locust CSV files under the timestamped `paper_artifacts/longrun_<timestamp>/` directory.
- Collect Prometheus metrics using explicit start/end timestamps covering the full long-run window.
- Label and harmonize the long-run CSV.
- Write the harmonized long-run dataset into a timestamp-specific SQLite table, such as `testbed_longrun_<timestamp>`.
- Run paper experiments with the long-run CSV inputs and write output under `paper_artifacts/longrun_<timestamp>/`.
- Generate a new summary file with:
  - Docker status.
  - Locust profile summaries.
  - Prometheus row count.
  - label distribution.
  - label thresholds.
  - SQLite table and row count.
  - generated tables/figures.
  - any remaining `not_run`, `failed`, `missing`, or `error` entries with context.

No long-run result will be written into README or `paper_artifacts/paper_summary.md` until the long-run files exist and have been read back from disk.
