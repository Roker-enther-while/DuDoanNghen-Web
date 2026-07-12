"""Quantitative checks for dataset size, event coverage, and transparency."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe_float(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(value)


def summarize_vector(values: np.ndarray, threshold: float = 0.70) -> dict[str, Any]:
    """Summarize target distribution and event coverage."""
    y = np.asarray(values, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return {"count": 0, "positive_count": 0, "positive_rate": 0.0}
    delta = np.abs(np.diff(y)) if y.size > 1 else np.array([], dtype=np.float32)
    autocorr = {}
    for lag in [1, 5, 15, 30]:
        if y.size > lag and float(np.std(y[:-lag])) > 0 and float(np.std(y[lag:])) > 0:
            corr = float(np.corrcoef(y[:-lag], y[lag:])[0, 1])
        else:
            corr = 0.0
        autocorr[f"lag_{lag}"] = _safe_float(corr)
    quantiles = {
        "p90": _safe_float(np.quantile(y, 0.90)),
        "p95": _safe_float(np.quantile(y, 0.95)),
        "p99": _safe_float(np.quantile(y, 0.99)),
    }
    threshold_counts = {
        "0.50": int(np.sum(y >= 0.50)),
        "0.60": int(np.sum(y >= 0.60)),
        "0.70": int(np.sum(y >= 0.70)),
        "0.80": int(np.sum(y >= 0.80)),
        "p90": int(np.sum(y >= quantiles["p90"])),
        "p95": int(np.sum(y >= quantiles["p95"])),
        "p99": int(np.sum(y >= quantiles["p99"])),
    }
    return {
        "count": int(y.size),
        "min": _safe_float(np.min(y)),
        "max": _safe_float(np.max(y)),
        "mean": _safe_float(np.mean(y)),
        "std": _safe_float(np.std(y)),
        **quantiles,
        "threshold_counts": threshold_counts,
        "positive_count": int(np.sum(y >= threshold)),
        "positive_rate": _safe_float(np.mean(y >= threshold)),
        "volatility_mean_abs_delta": _safe_float(np.mean(delta)) if delta.size else 0.0,
        "near_constant_rate_delta_lt_0_005": _safe_float(np.mean(delta < 0.005)) if delta.size else 0.0,
        "near_constant_rate_delta_lt_0_01": _safe_float(np.mean(delta < 0.01)) if delta.size else 0.0,
        "spike_count_delta_gt_0_05": int(np.sum(delta > 0.05)) if delta.size else 0,
        "autocorrelation": autocorr,
    }


def _labels_summary(labels_path: str | Path | None) -> dict[str, Any] | None:
    if not labels_path:
        return None
    path = Path(labels_path)
    if not path.exists():
        return None
    labels = pd.read_csv(path)
    total = int(len(labels))
    positives = int(labels["true_alert_label"].astype(bool).sum()) if "true_alert_label" in labels.columns else 0
    negatives = total - positives
    phase_distribution = labels["phase"].value_counts().to_dict() if "phase" in labels.columns else {}
    scenario_distribution = labels["scenario_name"].value_counts().to_dict() if "scenario_name" in labels.columns else {}
    severity = labels["severity"].astype(float) if "severity" in labels.columns else pd.Series(dtype=float)
    return {
        "labels_path": str(path),
        "total": total,
        "positive_count": positives,
        "negative_count": negatives,
        "positive_ratio": float(positives / total) if total else 0.0,
        "phase_distribution": {str(k): int(v) for k, v in phase_distribution.items()},
        "scenario_distribution": {str(k): int(v) for k, v in scenario_distribution.items()},
        "severity_distribution": {
            "min": _safe_float(float(severity.min())) if len(severity) else 0.0,
            "max": _safe_float(float(severity.max())) if len(severity) else 0.0,
            "mean": _safe_float(float(severity.mean())) if len(severity) else 0.0,
            "std": _safe_float(float(severity.std(ddof=0))) if len(severity) else 0.0,
        },
        "background_count": int(phase_distribution.get("background", 0)),
        "incident_count": int(phase_distribution.get("incident", 0)),
    }


def analyze_windows_artifact(
    path: str | Path,
    threshold: float = 0.70,
    source_manifest: dict | None = None,
    labels_path: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze a model-ready windows artifact without training."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        y_train = data["y_train"]
        y_val = data["y_val"]
        y_test = data["y_test"]
        scenario_names = data["scenario_name"] if "scenario_name" in data.files else np.array([], dtype=object)
        synthetic_flags = data["is_synthetic"] if "is_synthetic" in data.files else np.array([], dtype=bool)
        true_alert_label = data["true_alert_label"] if "true_alert_label" in data.files else None
        target_column = str(data["target_column"].tolist()) if "target_column" in data.files else ""
        source_keys = [key for key in data.files if key.startswith("source_id_")]
        sources = sorted({str(item) for key in source_keys for item in data[key].tolist()})
        if not sources:
            sources = ["unknown_or_single_source"]
        shapes = {
            split: {
                "X": list(data[f"X_{split}"].shape),
                "y": list(data[f"y_{split}"].shape),
                "dtype": str(data[f"X_{split}"].dtype),
            }
            for split in ["train", "val", "test"]
        }

    summaries = {
        "train": summarize_vector(y_train, threshold),
        "val": summarize_vector(y_val, threshold),
        "test": summarize_vector(y_test, threshold),
    }
    if true_alert_label is not None:
        summaries["test"]["synthetic_true_alert_count"] = int(np.sum(true_alert_label))
    scenario_count = int(len(set(str(item) for item in scenario_names.tolist()))) if scenario_names.size else 0
    synthetic_count = int(np.sum(synthetic_flags)) if synthetic_flags.size else 0
    label_summary = _labels_summary(labels_path)
    if label_summary is not None:
        scenario_count = len(label_summary["scenario_distribution"])
    test_positive = int(summaries["test"].get("positive_count", 0))
    test_quiet = test_positive < max(10, int(0.01 * summaries["test"]["count"]))
    fixed_070_quiet = int(summaries["test"]["threshold_counts"]["0.70"]) == 0
    target_is_proxy = "congestion_score" in target_column or "nasa_http" in str(path)
    enough_train = summaries["train"]["count"] >= 1000
    provenance_ok = bool(source_manifest and source_manifest.get("valid_source_count", 0) > 0)
    ready_for_stress = scenario_count >= 6 and synthetic_count > 0

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(path),
        "threshold": threshold,
        "shapes": shapes,
        "sources": sources,
        "source_count": len(sources),
        "scenario_count": scenario_count,
        "synthetic_count": synthetic_count,
        "label_summary": label_summary,
        "splits": summaries,
        "test_quiet": test_quiet,
        "fixed_0_70_has_no_positive_test_cases": fixed_070_quiet,
        "target_type": "synthetic_label" if label_summary is not None else ("proxy" if target_is_proxy else "unknown_or_measured"),
        "provenance_ok": provenance_ok,
        "transparency": {
            "source_provenance_full": provenance_ok,
            "license_full": provenance_ok,
            "citation_full": provenance_ok,
            "synthetic_flagged": bool(synthetic_count == 0 or synthetic_flags.size == synthetic_count),
            "raw_pii_handling_declared": provenance_ok,
        },
        "ready_for_training": bool(enough_train and summaries["val"]["count"] > 0 and summaries["test"]["count"] > 0),
        "ready_for_real_world_claim": bool(provenance_ok and not test_quiet and synthetic_count == 0 and not target_is_proxy),
        "ready_for_stress_benchmark": bool(
            ready_for_stress
            and (
                label_summary is None
                or (label_summary["positive_count"] > 0 and label_summary["negative_count"] > 0)
            )
        ),
        "recommended_next_action": (
            "Use synthetic stress benchmark for alert evaluation and add more real public sources."
            if test_quiet
            else "Proceed with transparent training/evaluation using fixed source and threshold policy."
        ),
        "warnings": [
            warning
            for warning, active in [
                ("test_split_has_too_few_positive_events", test_quiet),
                ("threshold_0_70_has_no_positive_cases_in_test", fixed_070_quiet),
                ("target_is_proxy_not_measured_congestion", target_is_proxy and label_summary is None),
                ("source_license_manifest_missing_or_not_loaded", not provenance_ok),
                ("synthetic_data_must_not_be_reported_as_real_world", synthetic_count > 0),
                ("synthetic_positive_ratio_is_100_percent", bool(label_summary and label_summary["positive_ratio"] == 1.0)),
                ("synthetic_negative_count_is_zero", bool(label_summary and label_summary["negative_count"] == 0)),
            ]
            if active
        ],
    }


def write_adequacy_report(result: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write adequacy JSON and Markdown report."""
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    reports_dir = output_dir / "reports"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = metrics_dir / "dataset_adequacy.json"
    md_path = reports_dir / "dataset_adequacy.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    test = result["splits"]["test"]
    lines = [
        "# Dataset Adequacy Report",
        "",
        f"- Data: {result['data_path']}",
        f"- Train samples: {result['splits']['train']['count']}",
        f"- Val samples: {result['splits']['val']['count']}",
        f"- Test samples: {test['count']}",
        f"- Source count: {result['source_count']}",
        f"- Scenario count: {result['scenario_count']}",
        f"- Test positive count @ threshold {result['threshold']}: {test['positive_count']}",
        f"- Test positive count @ threshold 0.70: {test['threshold_counts']['0.70']}",
        f"- Test positive rate: {test['positive_rate']:.6f}",
        f"- Test volatility mean abs delta: {test['volatility_mean_abs_delta']:.6f}",
        f"- Test spike count delta>0.05: {test['spike_count_delta_gt_0_05']}",
        f"- Test quiet: {result['test_quiet']}",
        f"- Target type: {result['target_type']}",
        f"- Ready for training: {result['ready_for_training']}",
        f"- Ready for real-world claim: {result['ready_for_real_world_claim']}",
        f"- Ready for stress benchmark: {result['ready_for_stress_benchmark']}",
        f"- Recommended next action: {result['recommended_next_action']}",
    ]
    if result.get("label_summary"):
        labels = result["label_summary"]
        lines.extend(
            [
                "",
                "## Synthetic Labels",
                f"- Positive count: {labels['positive_count']}",
                f"- Negative count: {labels['negative_count']}",
                f"- Positive ratio: {labels['positive_ratio']:.6f}",
                f"- Phase distribution: `{json.dumps(labels['phase_distribution'])}`",
                f"- Scenario distribution: `{json.dumps(labels['scenario_distribution'])}`",
                f"- Severity distribution: `{json.dumps(labels['severity_distribution'])}`",
            ]
        )
    lines.extend(["", "## Warnings"])
    lines.extend([f"- {warning}" for warning in result["warnings"]] or ["- None"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}
