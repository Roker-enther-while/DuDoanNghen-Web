"""Check whether a local Zanbil raw log is ready for privacy-safe preparation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.source_governance import build_license_manifest, load_public_sources
from src.data.zanbil_logs import parse_zanbil_log_line, sample_zanbil_lines


GUIDANCE = [
    "Download the dataset from the source declared in configs/data/public_sources.yaml.",
    "Place the authorized raw log at data/raw/zanbil/access.log.",
    "Do not commit the raw log unless project policy explicitly allows it.",
    "Run: python scripts/prepare_zanbil_logs.py --input data/raw/zanbil/access.log --config configs/data/zanbil_logs.yaml",
]


def check_zanbil_readiness(input_path: str | Path, source_config: str | Path) -> dict:
    """Return readiness metadata; missing raw file is not an error."""
    input_path = Path(input_path)
    manifest = build_license_manifest(load_public_sources(source_config))
    source = next((item for item in manifest["sources"] if item["source_id"] == "zanbil_web_logs"), None)
    governance_ok = source is not None
    pii_policy = source.get("pii_handling", []) if source else []
    sample_lines: list[str] = []
    parsed_samples = 0
    parser_ready = False
    if input_path.exists():
        sample_lines = sample_zanbil_lines(input_path, 5)
        parsed_samples = sum(1 for line in sample_lines if parse_zanbil_log_line(line, salt="readiness-check") is not None)
        parser_ready = parsed_samples > 0

    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "raw_path": str(input_path),
        "raw_exists": input_path.exists(),
        "file_size_bytes": int(input_path.stat().st_size) if input_path.exists() else 0,
        "sample_line_count": len(sample_lines),
        "parsed_sample_count": parsed_samples,
        "parser_ready": parser_ready,
        "source_governance_ready": governance_ok,
        "license_name": source.get("license_name") if source else None,
        "citation": source.get("citation") if source else None,
        "pii_policy": pii_policy,
        "pii_policy_ready": all(item in pii_policy for item in ["hash_ip", "drop_query_string", "drop_user_agent"]),
        "ready_for_prepare": bool(input_path.exists() and parser_ready and governance_ok),
        "guidance": [] if input_path.exists() else GUIDANCE,
    }


def write_readiness_outputs(result: dict, output_root: str | Path = "outputs") -> dict[str, str]:
    output_root = Path(output_root)
    metrics_path = output_root / "metrics" / "zanbil_readiness.json"
    report_path = output_root / "reports" / "zanbil_readiness.md"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Zanbil Readiness",
        "",
        f"- Raw path: {result['raw_path']}",
        f"- Raw exists: {result['raw_exists']}",
        f"- File size bytes: {result['file_size_bytes']}",
        f"- Parser ready: {result['parser_ready']}",
        f"- Parsed sample count: {result['parsed_sample_count']}",
        f"- Source governance ready: {result['source_governance_ready']}",
        f"- PII policy ready: {result['pii_policy_ready']}",
        f"- Ready for prepare: {result['ready_for_prepare']}",
    ]
    if result["guidance"]:
        lines.extend(["", "## Next Steps"])
        lines.extend([f"{idx}. {step}" for idx, step in enumerate(result["guidance"], start=1)])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(metrics_path), "markdown": str(report_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/zanbil/access.log")
    parser.add_argument("--source-config", default="configs/data/public_sources.yaml")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args(argv)
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    source_config = Path(args.source_config)
    if not source_config.is_absolute():
        source_config = ROOT / source_config
    result = check_zanbil_readiness(input_path, source_config)
    paths = write_readiness_outputs(result, ROOT / args.output_dir)
    print(json.dumps({"status": "success", "raw_exists": result["raw_exists"], "parser_ready": result["parser_ready"], "outputs": paths}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
