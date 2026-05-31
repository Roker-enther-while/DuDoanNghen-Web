# Final STATE B Runbook

## Check Tests

```bash
python -m pytest -q
```

## View Final Reports

- `outputs/reports/final_state_b_research_summary.md`
- `outputs/reports/final_experiment_summary.md`
- `outputs/reports/final_artifact_manifest.md`

## Re-run Synthetic Stress Evaluation

```bash
python scripts/evaluate_synthetic_stress.py   --data data/processed/synthetic_stress/windows/windows_fp16.npz   --labels data/processed/synthetic_stress/labels/synthetic_stress_labels.csv   --model-path outputs/models/full_120_tcn_attention_bilstm/best_model.pt   --output-dir outputs/synthetic_stress_eval/full_120_tcn_attention_bilstm
```

## Add Zanbil Raw

Place the valid raw Zanbil access log at:

```text
data/raw/zanbil/access.log
```

or import a downloaded file:

```bash
python scripts/import_zanbil_raw.py --input <path_to_downloaded_zanbil_file>
```

Then run:

```bash
python scripts/check_zanbil_readiness.py
python scripts/prepare_zanbil_logs.py --input data/raw/zanbil/access.log --config configs/data/zanbil_logs.yaml
python scripts/build_multi_source_dataset.py --config configs/data/multi_source_web_logs.yaml
```

Do not train multi-source while `ready_for_cross_source_claim=false` or while Zanbil is missing.
