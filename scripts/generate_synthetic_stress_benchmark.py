"""Generate a transparent synthetic stress benchmark from a public baseline artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.synthetic_stress import generate_synthetic_stress_windows


def write_synthetic_dashboard_payload(manifest: dict, output_root: Path) -> str:
    """Write machine-readable context for future dashboard rendering."""
    web_dir = output_root / "web" / "synthetic_stress"
    web_dir.mkdir(parents=True, exist_ok=True)
    payload_path = web_dir / "model_dashboard_payload.json"
    payload = {
        "run_type": "synthetic_stress_benchmark",
        "data_artifact_path": manifest["path"],
        "labels_path": manifest["labels_path"],
        "evaluation_context": {
            "result_type": "synthetic_stress_test",
            "target_type": "synthetic_label",
            "positive_ratio": manifest["positive_ratio"],
            "negative_count": manifest["negative_cases"],
            "positive_count": manifest["positive_cases"],
            "phase_distribution": manifest["phase_counts"],
            "scenario_distribution": manifest["scenario_counts"],
        },
        "data_governance": {
            "synthetic_policy": "Generated from public baseline; do not mix with real public test claims.",
            "source_id": "synthetic_stress_public_baseline",
        },
        "warnings": [
            "synthetic_not_real_world",
            "generated_from_public_baseline",
            "must_not_mix_with_real_public_test",
        ],
    }
    payload_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(payload_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-data", default=None)
    parser.add_argument("--config", default="configs/data/synthetic_stress_benchmark.yaml")
    args = parser.parse_args(argv)

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8")) or {}
    base_data = Path(args.base_data or config.get("base_data"))
    if not base_data.is_absolute():
        base_data = ROOT / base_data
    output_path = ROOT / config.get("output_path", "data/processed/synthetic_stress/windows/windows_fp16.npz")
    result = generate_synthetic_stress_windows(
        base_npz_path=base_data,
        output_path=output_path,
        samples_per_scenario=int(config.get("samples_per_scenario", 200)),
        seed=int(config.get("seed", 42)),
        phase_ratios={
            "background_ratio": float(config.get("background_ratio", 0.50)),
            "pre_incident_ratio": float(config.get("pre_incident_ratio", 0.15)),
            "incident_ratio": float(config.get("incident_ratio", 0.25)),
            "recovery_ratio": float(config.get("recovery_ratio", 0.10)),
        },
        target_positive_ratio_min=float(config.get("target_positive_ratio_min", 0.20)),
        target_positive_ratio_max=float(config.get("target_positive_ratio_max", 0.40)),
    )

    metrics_path = ROOT / "outputs" / "metrics" / "synthetic_stress_manifest.json"
    reports_path = ROOT / "outputs" / "reports" / "synthetic_stress_manifest.md"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    reports_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        **result,
        "policy": config.get("policy", {}),
        "scenario_labels": [
            "is_synthetic",
            "source_id",
            "scenario_name",
            "phase",
            "incident_start",
            "incident_peak",
            "incident_end",
            "true_alert_label",
            "severity",
            "generation_config_id",
        ],
    }
    manifest["dashboard_payload"] = write_synthetic_dashboard_payload(manifest, ROOT / "outputs")
    metrics_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    reports_path.write_text(
        "\n".join(
            [
                "# Synthetic Stress Benchmark Manifest",
                "",
                "Synthetic stress benchmark is generated from public real-data baseline.",
                "It is used only for controlled stress evaluation, not real-world performance claims.",
                "",
                f"- Base data: {base_data}",
                f"- Output: {result['path']}",
                f"- Scenarios: {', '.join(result['scenarios'])}",
                f"- Synthetic test samples: {result['synthetic_test_samples']}",
                f"- Positive labels: {result['positive_cases']}",
                f"- Negative labels: {result['negative_cases']}",
                f"- Positive ratio: {result['positive_ratio']:.4f}",
                f"- Phase counts: `{json.dumps(result['phase_counts'])}`",
                f"- Seed: {result['generation_config']['seed']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "success", "output": result["path"], "positive_cases": result["positive_cases"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
