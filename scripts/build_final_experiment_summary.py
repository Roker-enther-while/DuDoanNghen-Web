"""Build a final experiment package summary for the current autonomous run."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"missing": True, "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_payload(obj):
    if isinstance(obj, dict):
        return {key: _finite_payload(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_finite_payload(item) for item in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def build_summary(state: str = "auto") -> dict:
    source_manifest = _load_json(ROOT / "outputs" / "metrics" / "source_license_manifest.json")
    candidates = _load_json(ROOT / "outputs" / "metrics" / "zanbil_raw_candidates.json")
    readiness = _load_json(ROOT / "outputs" / "metrics" / "zanbil_readiness.json")
    multi_manifest = _load_json(ROOT / "outputs" / "metrics" / "multi_source_manifest.json")
    nasa_metrics = _load_json(ROOT / "outputs" / "metrics" / "full_120_tcn_attention_bilstm" / "final_metrics.json")
    synthetic_eval = _load_json(ROOT / "outputs" / "synthetic_stress_eval" / "full_120_tcn_attention_bilstm" / "metrics.json")
    calibration = _load_json(ROOT / "outputs" / "metrics" / "full_120_tcn_attention_bilstm" / "threshold_calibration.json")
    inventory = _load_json(ROOT / "outputs" / "metrics" / "current_artifact_inventory.json")

    multi_sources = multi_manifest.get("output", {}).get("sources", [])
    has_zanbil = "zanbil_web_logs" in multi_sources
    complete_state = "STATE A MULTI-SOURCE COMPLETE" if has_zanbil else "STATE B FALLBACK COMPLETE"
    if state != "auto":
        complete_state = state
    reason = (
        "multi-source manifest contains NASA and Zanbil"
        if has_zanbil
        else "Zanbil raw is missing or invalid; multi-source manifest remains NASA-only"
    )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "completion_state": complete_state,
        "state_reason": reason,
        "source_governance": {
            "valid_source_count": source_manifest.get("valid_source_count"),
            "rejected_source_count": source_manifest.get("rejected_source_count"),
            "sources": [source.get("source_id") for source in source_manifest.get("sources", [])],
            "path": "outputs/metrics/source_license_manifest.json",
        },
        "zanbil_status": {
            "raw_found_or_imported": bool(readiness.get("raw_exists")),
            "parser_ready": bool(readiness.get("parser_ready")),
            "ready_for_prepare": bool(readiness.get("ready_for_prepare")),
            "candidate_count": candidates.get("candidate_count"),
            "parseable_candidate_count": candidates.get("parseable_candidate_count"),
            "required_path": "data/raw/zanbil/access.log",
            "readiness_path": "outputs/metrics/zanbil_readiness.json",
            "candidate_path": "outputs/metrics/zanbil_raw_candidates.json",
        },
        "multi_source_status": {
            "sources": multi_sources,
            "ready_for_cross_source_claim": bool(multi_manifest.get("ready_for_cross_source_claim", False)),
            "warnings": multi_manifest.get("warnings", []),
            "manifest_path": "outputs/metrics/multi_source_manifest.json",
        },
        "nasa_only_real_public_proxy_result": {
            "metrics": nasa_metrics.get("metrics", {}),
            "alert_metrics": nasa_metrics.get("alert_metrics", {}),
            "target_notice": nasa_metrics.get("target_notice", "NASA target is a proxy congestion score."),
            "final_metrics_path": "outputs/metrics/full_120_tcn_attention_bilstm/final_metrics.json",
        },
        "threshold_calibration": {
            "available": not calibration.get("missing", False),
            "path": "outputs/metrics/full_120_tcn_attention_bilstm/threshold_calibration.json",
            "old_threshold": calibration.get("old_threshold"),
            "calibrated_threshold": calibration.get("calibrated_threshold"),
            "calibrated_test_metrics": calibration.get("calibrated_test_metrics"),
        },
        "synthetic_stress_result_separate": {
            "available": not synthetic_eval.get("missing", False),
            "result_type": synthetic_eval.get("result_type"),
            "synthetic_not_real_world": synthetic_eval.get("synthetic_not_real_world"),
            "checkpoint_threshold_metrics": synthetic_eval.get("checkpoint_threshold_metrics"),
            "best_synthetic_f1_threshold": synthetic_eval.get("best_synthetic_f1_threshold"),
            "path": "outputs/synthetic_stress_eval/full_120_tcn_attention_bilstm/metrics.json",
        },
        "inventory_path": "outputs/metrics/current_artifact_inventory.json",
        "artifact_count": inventory.get("summary", {}).get("npz_count"),
        "limitations": [
            "NASA target is a proxy congestion score, not measured congestion.",
            "Synthetic stress benchmark is not real-world data and is reported separately.",
            "No cross-source claim is allowed until Zanbil raw is imported and processed.",
            "Only sources with license/citation in source governance may be used.",
        ],
        "next_action": (
            "Place a valid Zanbil raw log at data/raw/zanbil/access.log, then rerun the autonomous goal."
            if not has_zanbil
            else "Update dashboard using the final payload and review multi-source metrics."
        ),
        "output_paths": {
            "source_license_manifest": "outputs/metrics/source_license_manifest.json",
            "raw_candidates": "outputs/metrics/zanbil_raw_candidates.json",
            "zanbil_readiness": "outputs/metrics/zanbil_readiness.json",
            "synthetic_eval": "outputs/synthetic_stress_eval/full_120_tcn_attention_bilstm/metrics.json",
            "final_summary_json": "outputs/metrics/final_experiment_summary.json",
            "final_summary_md": "outputs/reports/final_experiment_summary.md",
        },
    }
    return _finite_payload(summary)


def write_summary(summary: dict) -> dict[str, str]:
    metrics_path = ROOT / "outputs" / "metrics" / "final_experiment_summary.json"
    report_path = ROOT / "outputs" / "reports" / "final_experiment_summary.md"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    synthetic = summary["synthetic_stress_result_separate"].get("best_synthetic_f1_threshold") or {}
    nasa_metrics = summary["nasa_only_real_public_proxy_result"].get("metrics") or {}
    nasa_alert = summary["nasa_only_real_public_proxy_result"].get("alert_metrics") or {}
    lines = [
        "# Final Experiment Summary",
        "",
        f"- Completion state: {summary['completion_state']}",
        f"- Reason: {summary['state_reason']}",
        f"- Source governance valid sources: {summary['source_governance']['sources']}",
        f"- Zanbil ready: {summary['zanbil_status']['ready_for_prepare']}",
        f"- Multi-source sources: {summary['multi_source_status']['sources']}",
        f"- Ready for cross-source claim: {summary['multi_source_status']['ready_for_cross_source_claim']}",
        "",
        "## NASA-only Real Public Proxy Result",
        f"- MAE/RMSE/R2: {nasa_metrics.get('mae')} / {nasa_metrics.get('rmse')} / {nasa_metrics.get('r2')}",
        f"- Precision/Recall/F1: {nasa_alert.get('precision')} / {nasa_alert.get('recall')} / {nasa_alert.get('f1')}",
        "",
        "## Synthetic Stress Result",
        "- result_type: synthetic_stress_test",
        "- synthetic_not_real_world: true",
        f"- Best synthetic precision/recall/F1: {synthetic.get('precision')} / {synthetic.get('recall')} / {synthetic.get('f1')}",
        "",
        "## Limitations",
    ]
    lines.extend([f"- {item}" for item in summary["limitations"]])
    lines.extend(["", f"## Next Action\n- {summary['next_action']}"])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(metrics_path), "markdown": str(report_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default="auto")
    args = parser.parse_args(argv)
    summary = build_summary(args.state)
    paths = write_summary(summary)
    print(json.dumps({"status": "success", "state": summary["completion_state"], "outputs": paths}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
