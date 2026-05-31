"""Build source license/provenance manifest for data expansion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.source_governance import build_license_manifest, load_public_sources, write_license_outputs


def _update_dashboard_schema(manifest: dict, output_root: Path) -> None:
    """Add governance fields to existing full-run payloads without changing model results."""
    payload_path = output_root / "web" / "full_120_tcn_attention_bilstm" / "model_dashboard_payload.json"
    if not payload_path.exists():
        return
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload.setdefault("data_governance", {})
    payload["data_governance"].update(
        {
            "sources": [source["source_id"] for source in manifest["sources"]],
            "license_summary": {
                source["source_id"]: source["license_name"] for source in manifest["sources"]
            },
            "citation_summary": {
                source["source_id"]: source["citation"] for source in manifest["sources"]
            },
            "pii_handling": {
                source["source_id"]: source.get("pii_handling", []) for source in manifest["sources"]
            },
            "synthetic_policy": "Synthetic stress data must be flagged and reported separately from real public test results.",
        }
    )
    payload.setdefault("evaluation_context", {})
    payload["evaluation_context"].update(
        {
            "result_type": "real_public_test",
            "target_type": "proxy",
            "threshold_policy": payload.get("threshold_info", {}),
            "dataset_adequacy_status": "pending_dataset_adequacy_report",
        }
    )
    warnings = payload.setdefault("warnings", [])
    for warning in [
        "NASA target is a proxy congestion score, not measured congestion.",
        "Synthetic stress results must not be reported as real-world performance.",
    ]:
        if warning not in warnings:
            warnings.append(warning)
    payload_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/public_sources.yaml")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args(argv)

    sources = load_public_sources(ROOT / args.config)
    manifest = build_license_manifest(sources)
    paths = write_license_outputs(manifest, ROOT / args.output_dir)
    _update_dashboard_schema(manifest, ROOT / args.output_dir)
    print(json.dumps({"status": "success", "outputs": paths, "valid_sources": manifest["valid_source_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
