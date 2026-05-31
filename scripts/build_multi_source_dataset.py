"""Build a multi-source web-log window artifact without losing source identity."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.multi_source_merge import merge_readiness, merge_window_artifacts
from src.data.source_governance import build_license_manifest, load_public_sources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/multi_source_web_logs.yaml")
    args = parser.parse_args(argv)
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8")) or {}

    source_manifest = build_license_manifest(load_public_sources(ROOT / config.get("license_manifest", "configs/data/public_sources.yaml")))
    valid_source_ids = {source["source_id"] for source in source_manifest["sources"]}
    inputs = []
    skipped = []
    for item in config.get("inputs", []):
        path = ROOT / item["path"]
        if item["source_id"] not in valid_source_ids:
            skipped.append({"source_id": item["source_id"], "reason": "not license-approved"})
            continue
        if not path.exists():
            skipped.append({"source_id": item["source_id"], "path": str(path), "reason": "missing optional input"})
            if not item.get("optional", False):
                raise FileNotFoundError(path)
            continue
        inputs.append({"source_id": item["source_id"], "path": path})
    if not inputs:
        raise SystemExit("No approved source window artifacts are available")

    output_path = ROOT / config.get("output_path", "data/processed/multi_source_web_logs/windows/windows_fp16.npz")
    merge_meta = merge_window_artifacts(inputs, output_path)
    readiness = merge_readiness(merge_meta["sources"])
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "output": merge_meta,
        "skipped_inputs": skipped,
        "warnings": readiness["warnings"],
        "ready_for_training": readiness["ready_for_training"],
        "ready_for_cross_source_claim": readiness["ready_for_cross_source_claim"],
        "ready_for_real_world_claim": readiness["ready_for_real_world_claim"],
        "license_summary": {
            source["source_id"]: source["license_name"] for source in source_manifest["sources"]
        },
        "pii_handling_summary": {
            source["source_id"]: source.get("pii_handling", []) for source in source_manifest["sources"]
        },
        "leakage_policy": "Window artifacts are concatenated only within their existing train/val/test partitions.",
    }
    metrics_path = ROOT / "outputs" / "metrics" / "multi_source_manifest.json"
    report_path = ROOT / "outputs" / "reports" / "multi_source_manifest.md"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(
        "\n".join(
            [
                "# Multi-Source Web Log Manifest",
                "",
                f"- Output: {merge_meta['path']}",
                f"- Sources: {', '.join(merge_meta['sources'])}",
                f"- Skipped inputs: {len(skipped)}",
                f"- Warnings: {', '.join(readiness['warnings']) if readiness['warnings'] else 'None'}",
                f"- Ready for training: {manifest['ready_for_training']}",
                f"- Ready for cross-source claim: {manifest['ready_for_cross_source_claim']}",
                f"- Ready for real-world claim: {manifest['ready_for_real_world_claim']}",
                f"- Source distribution: `{json.dumps(merge_meta['source_distribution'])}`",
                "- Source identity is preserved in source_id arrays per split.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "success", "output": merge_meta["path"], "skipped": skipped}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
