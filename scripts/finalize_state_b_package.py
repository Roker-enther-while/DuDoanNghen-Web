"""Finalize STATE B package with verified NASA-only and synthetic stress results."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"missing": True, "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(obj):
    if isinstance(obj, dict):
        return {key: _finite(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_finite(value) for value in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _artifact(path: str, purpose: str, result_type: str) -> dict[str, Any]:
    p = ROOT / path
    return {
        "path": path,
        "exists": p.exists(),
        "size_bytes": int(p.stat().st_size) if p.exists() else 0,
        "purpose": purpose,
        "result_type": result_type,
    }


def _best_worst_scenario(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    df = pd.read_csv(path)
    best = df.sort_values(["f1", "recall"], ascending=False).iloc[0].to_dict()
    worst = df.sort_values(["f1", "recall"], ascending=True).iloc[0].to_dict()
    return best, worst


def _phase_summary(path: str | Path) -> dict[str, dict[str, Any]]:
    df = pd.read_csv(path)
    return {str(row["phase"]): {key: row[key] for key in df.columns if key != "phase"} for _, row in df.iterrows()}


def build_state_b_payload() -> dict[str, Any]:
    metrics = _load_json(ROOT / "outputs" / "metrics" / "full_120_tcn_attention_bilstm" / "final_metrics.json")
    calibration = _load_json(ROOT / "outputs" / "metrics" / "full_120_tcn_attention_bilstm" / "threshold_calibration.json")
    synthetic = _load_json(ROOT / "outputs" / "synthetic_stress_eval" / "full_120_tcn_attention_bilstm" / "metrics.json")
    readiness = _load_json(ROOT / "outputs" / "metrics" / "zanbil_readiness.json")
    multi = _load_json(ROOT / "outputs" / "metrics" / "multi_source_manifest.json")
    source_manifest = _load_json(ROOT / "outputs" / "metrics" / "source_license_manifest.json")
    scenario_path = ROOT / "outputs" / "synthetic_stress_eval" / "full_120_tcn_attention_bilstm" / "scenario_metrics.csv"
    phase_path = ROOT / "outputs" / "synthetic_stress_eval" / "full_120_tcn_attention_bilstm" / "phase_metrics.csv"
    best_scenario, worst_scenario = _best_worst_scenario(scenario_path)
    phases = _phase_summary(phase_path)

    reg = metrics.get("metrics", {})
    alert = metrics.get("alert_metrics", {})
    calibrated = calibration.get("calibrated_test_metrics", {})
    best_synth = synthetic.get("best_synthetic_f1_threshold", {})
    checkpoint_synth = synthetic.get("checkpoint_threshold_metrics", {})
    sources = multi.get("output", {}).get("sources", [])

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_state": "STATE_B_FALLBACK_COMPLETE",
        "model_name": "tcn_attention_bilstm",
        "model_path": "outputs/models/full_120_tcn_attention_bilstm/best_model.pt",
        "data_context": {
            "real_public_dataset": "nasa_http_1995",
            "target_type": "proxy_congestion_score",
            "train_samples": 62425,
            "val_samples": 13330,
            "test_samples": 13330,
            "multi_source_available": False,
            "cross_source_claim": False,
            "zanbil_raw_missing": not bool(readiness.get("raw_exists")),
            "missing_zanbil_path": "data/raw/zanbil/access.log",
            "multi_source_sources": sources,
            "ready_for_cross_source_claim": bool(multi.get("ready_for_cross_source_claim", False)),
        },
        "real_public_proxy_result": {
            "mae": reg.get("mae"),
            "rmse": reg.get("rmse"),
            "r2": reg.get("r2"),
            "precision": alert.get("precision"),
            "recall": alert.get("recall"),
            "f1": alert.get("f1"),
            "threshold": metrics.get("alert_threshold", alert.get("alert_threshold")),
            "true_positive_count": alert.get("alert_positive_count_true"),
            "pred_positive_count": alert.get("alert_positive_count_pred"),
            "tp": alert.get("tp"),
            "fp": alert.get("fp"),
            "tn": alert.get("tn"),
            "fn": alert.get("fn"),
        },
        "threshold_calibration": {
            "calibrated_threshold": calibration.get("calibrated_threshold"),
            "calibrated_f1": calibrated.get("f1"),
            "calibrated_recall": calibrated.get("recall"),
            "calibrated_precision": calibrated.get("precision"),
            "warning": "calibrated threshold must be reported separately from original p90 threshold",
            "path": "outputs/metrics/full_120_tcn_attention_bilstm/threshold_calibration.json",
        },
        "synthetic_stress_result": {
            "result_type": "synthetic_stress_test",
            "synthetic_not_real_world": True,
            "positive_ratio": synthetic.get("positive_ratio"),
            "checkpoint_threshold": synthetic.get("checkpoint_threshold"),
            "checkpoint_threshold_precision": checkpoint_synth.get("precision"),
            "checkpoint_threshold_recall": checkpoint_synth.get("recall"),
            "checkpoint_threshold_f1": checkpoint_synth.get("f1"),
            "best_synthetic_threshold": best_synth.get("threshold"),
            "best_synthetic_precision": best_synth.get("precision"),
            "best_synthetic_recall": best_synth.get("recall"),
            "best_synthetic_f1": best_synth.get("f1"),
            "best_scenario": best_scenario.get("scenario_name"),
            "best_scenario_f1": best_scenario.get("f1"),
            "worst_scenario": worst_scenario.get("scenario_name"),
            "worst_scenario_f1": worst_scenario.get("f1"),
            "phase_metrics": phases,
        },
        "governance": {
            "source_license_manifest": "outputs/metrics/source_license_manifest.json",
            "valid_sources": [source.get("source_id") for source in source_manifest.get("sources", [])],
            "synthetic_separate_from_real": True,
            "no_cross_source_claim": True,
            "no_measured_congestion_claim": True,
        },
        "warnings": [
            "NASA target is proxy congestion score, not measured congestion.",
            "Zanbil raw is missing, so multi-source training was not performed.",
            "Synthetic stress benchmark is controlled simulation, not real-world result.",
            "Original p90 threshold has very low recall; calibrated threshold must be explained separately.",
        ],
        "artifact_paths": {
            "final_metrics": "outputs/metrics/full_120_tcn_attention_bilstm/final_metrics.json",
            "threshold_calibration": "outputs/metrics/full_120_tcn_attention_bilstm/threshold_calibration.json",
            "synthetic_metrics": "outputs/synthetic_stress_eval/full_120_tcn_attention_bilstm/metrics.json",
            "final_report": "outputs/reports/final_state_b_research_summary.md",
            "artifact_manifest": "outputs/metrics/final_artifact_manifest.json",
        },
    }
    return _finite(payload)


def build_artifact_manifest(dashboard_html_path: str | None = None) -> dict[str, Any]:
    groups = {
        "models": [
            _artifact("outputs/models/full_120_tcn_attention_bilstm/best_model.pt", "best NASA-only full 120 model", "real_public_proxy"),
            _artifact("outputs/models/full_120_tcn_attention_bilstm/last_model.pt", "last NASA-only full 120 model", "real_public_proxy"),
        ],
        "real_public_metrics": [
            _artifact("outputs/metrics/full_120_tcn_attention_bilstm/final_metrics.json", "NASA-only final metrics", "real_public_proxy"),
            _artifact("outputs/predictions/full_120_tcn_attention_bilstm/test_predictions.csv", "NASA-only test predictions", "real_public_proxy"),
        ],
        "calibration": [
            _artifact("outputs/metrics/full_120_tcn_attention_bilstm/threshold_calibration.json", "NASA-only threshold calibration", "calibration"),
            _artifact("outputs/reports/full_120_tcn_attention_bilstm/threshold_calibration.md", "NASA-only threshold calibration report", "calibration"),
        ],
        "synthetic": [
            _artifact("outputs/synthetic_stress_eval/full_120_tcn_attention_bilstm/metrics.json", "synthetic stress metrics", "synthetic_stress_test"),
            _artifact("outputs/synthetic_stress_eval/full_120_tcn_attention_bilstm/report.md", "synthetic stress report", "synthetic_stress_test"),
            _artifact("outputs/synthetic_stress_eval/full_120_tcn_attention_bilstm/scenario_metrics.csv", "synthetic scenario metrics", "synthetic_stress_test"),
            _artifact("outputs/synthetic_stress_eval/full_120_tcn_attention_bilstm/phase_metrics.csv", "synthetic phase metrics", "synthetic_stress_test"),
        ],
        "governance": [
            _artifact("outputs/metrics/source_license_manifest.json", "source license manifest", "governance"),
            _artifact("outputs/metrics/zanbil_readiness.json", "Zanbil readiness status", "governance"),
            _artifact("outputs/metrics/multi_source_manifest.json", "multi-source manifest", "governance"),
        ],
        "dashboard": [
            _artifact("outputs/web/final_state_b_dashboard_payload.json", "final STATE B dashboard payload", "dashboard_payload"),
            _artifact("outputs/web/full_120_tcn_attention_bilstm/model_dashboard_payload.json", "updated full_120 dashboard payload", "dashboard_payload"),
        ],
    }
    if dashboard_html_path:
        groups["dashboard"].append(_artifact(dashboard_html_path, "dashboard HTML", "dashboard"))
    flat = [item for items in groups.values() for item in items]
    return _finite(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "state": "STATE_B_FALLBACK_COMPLETE",
            "groups": groups,
            "summary": {
                "artifact_count": len(flat),
                "exists_count": sum(1 for item in flat if item["exists"]),
                "missing_count": sum(1 for item in flat if not item["exists"]),
            },
        }
    )


def write_payloads(payload: dict[str, Any]) -> None:
    final_payload = ROOT / "outputs" / "web" / "final_state_b_dashboard_payload.json"
    existing_payload = ROOT / "outputs" / "web" / "full_120_tcn_attention_bilstm" / "model_dashboard_payload.json"
    final_payload.parent.mkdir(parents=True, exist_ok=True)
    existing_payload.parent.mkdir(parents=True, exist_ok=True)
    final_payload.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    old = _load_json(existing_payload)
    old.update(payload)
    existing_payload.write_text(json.dumps(_finite(old), indent=2, ensure_ascii=False), encoding="utf-8")


def write_research_summary(payload: dict[str, Any]) -> None:
    result = payload["real_public_proxy_result"]
    calibration = payload["threshold_calibration"]
    synthetic = payload["synthetic_stress_result"]
    path = ROOT / "outputs" / "reports" / "final_state_b_research_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Final Experiment Summary — STATE B

## 1. Scope
- Model: TCN-Attention-BiLSTM
- Real public data: NASA HTTP 1995
- Target: proxy congestion score
- Synthetic stress: controlled benchmark
- Cross-source: not available because Zanbil raw is missing

## 2. Data Governance
- NASA source is tracked in `outputs/metrics/source_license_manifest.json`.
- Zanbil is declared in governance but raw input is missing at `data/raw/zanbil/access.log`.
- Synthetic stress is generated from a public baseline and must remain separate from real public results.
- PII policy: hash client identifiers, strip query strings, drop user-agent by default, and do not release raw logs.

## 3. Real Public Proxy Result

| Metric | Value |
|---|---:|
| MAE | {result['mae']:.6f} |
| RMSE | {result['rmse']:.6f} |
| R² | {result['r2']:.6f} |
| Precision | {result['precision']:.6f} |
| Recall | {result['recall']:.6f} |
| F1 | {result['f1']:.6f} |
| Threshold | {result['threshold']:.6f} |
| TP / FP / TN / FN | {result['tp']} / {result['fp']} / {result['tn']} / {result['fn']} |

Regression on the proxy target is usable but not conclusive for measured congestion. Alert recall at the original p90 validation threshold is very low, so this threshold should not be presented as a strong real-world alerting result.

## 4. Threshold Calibration
- Calibrated threshold: `{calibration['calibrated_threshold']:.6f}`
- Calibrated F1: `{calibration['calibrated_f1']:.6f}`
- Calibrated recall: `{calibration['calibrated_recall']:.6f}`

This is a threshold calibration result. It changes alert classification and must be reported separately from the original p90-threshold result.

## 5. Synthetic Stress Benchmark
- result_type: `synthetic_stress_test`
- synthetic_not_real_world: `true`
- Positive ratio: `{synthetic['positive_ratio']:.6f}`
- Scenarios: 6
- Checkpoint-threshold F1: `{synthetic['checkpoint_threshold_f1']:.6f}`
- Best synthetic-threshold F1: `{synthetic['best_synthetic_f1']:.6f}`
- Best scenario: `{synthetic['best_scenario']}` with F1 `{synthetic['best_scenario_f1']:.6f}`
- Worst scenario: `{synthetic['worst_scenario']}` with F1 `{synthetic['worst_scenario_f1']:.6f}`

Synthetic stress is a controlled benchmark and not a real-world performance claim.

## 6. Why Multi-source Was Not Trained
- Missing file: `data/raw/zanbil/access.log`
- Multi-source currently contains only `nasa_http_1995`.
- `ready_for_cross_source_claim=false`.
- Training a multi-source model in this state would be misleading.

## 7. Honest Conclusion
The pipeline, governance, training, and evaluation flow are complete for NASA-only STATE B. The model learns the NASA proxy regression signal, but alerting requires threshold calibration and more diverse real data. Synthetic stress shows stronger response to periodic spikes and weaker behavior on error surge. There is no cross-source or measured-congestion conclusion yet.

## 8. Next Work
- Place a valid Zanbil raw log at `data/raw/zanbil/access.log`.
- Prepare Zanbil.
- Build NASA+Zanbil multi-source data.
- Train `multisource_full_120_tcn_attention_bilstm`.
- Compare NASA-only vs multi-source with the same governance and threshold policy.
"""
    path.write_text(text, encoding="utf-8")


def write_artifact_manifest(manifest: dict[str, Any]) -> None:
    json_path = ROOT / "outputs" / "metrics" / "final_artifact_manifest.json"
    md_path = ROOT / "outputs" / "reports" / "final_artifact_manifest.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Final Artifact Manifest",
        "",
        f"- State: {manifest['state']}",
        f"- Exists: {manifest['summary']['exists_count']}",
        f"- Missing: {manifest['summary']['missing_count']}",
        "",
    ]
    for group, items in manifest["groups"].items():
        lines.extend([f"## {group}", "", "| path | exists | size_bytes | purpose | result_type |", "|---|---:|---:|---|---|"])
        for item in items:
            lines.append(f"| {item['path']} | {item['exists']} | {item['size_bytes']} | {item['purpose']} | {item['result_type']} |")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def write_runbook() -> None:
    path = ROOT / "docs" / "final_state_b_runbook.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# Final STATE B Runbook

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
python scripts/evaluate_synthetic_stress.py \
  --data data/processed/synthetic_stress/windows/windows_fp16.npz \
  --labels data/processed/synthetic_stress/labels/synthetic_stress_labels.csv \
  --model-path outputs/models/full_120_tcn_attention_bilstm/best_model.pt \
  --output-dir outputs/synthetic_stress_eval/full_120_tcn_attention_bilstm
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
""",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard-html", default=None)
    args = parser.parse_args(argv)
    payload = build_state_b_payload()
    write_payloads(payload)
    write_research_summary(payload)
    manifest = build_artifact_manifest(args.dashboard_html)
    write_artifact_manifest(manifest)
    write_runbook()
    print(
        json.dumps(
            {
                "status": "success",
                "payload": "outputs/web/final_state_b_dashboard_payload.json",
                "research_summary": "outputs/reports/final_state_b_research_summary.md",
                "artifact_manifest": "outputs/metrics/final_artifact_manifest.json",
                "runbook": "docs/final_state_b_runbook.md",
                "exists_count": manifest["summary"]["exists_count"],
                "missing_count": manifest["summary"]["missing_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
