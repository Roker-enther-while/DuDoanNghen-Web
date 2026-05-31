"""Prepare a local Zanbil access.log into privacy-safe time-series windows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.source_governance import build_license_manifest, load_public_sources
from src.data.zanbil_logs import prepare_zanbil_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--config", default="configs/data/zanbil_logs.yaml")
    parser.add_argument("--limit-lines", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    public_sources = load_public_sources(ROOT / config.get("license_manifest", "configs/data/public_sources.yaml"))
    manifest = build_license_manifest(public_sources)
    source_ids = {source["source_id"] for source in manifest["sources"]}
    if "zanbil_web_logs" not in source_ids:
        raise SystemExit("zanbil_web_logs is not license-approved in source manifest")

    input_path = Path(args.input or config.get("input_path", "data/raw/zanbil/access.log"))
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    if not input_path.exists():
        guidance = {
            "status": "missing_raw_input",
            "input_path": str(input_path),
            "message": "Place an authorized local copy of access.log at data/raw/zanbil/access.log. No Kaggle/Dataverse raw download is performed automatically.",
            "citation": "Zaker, Farzin, 2019, Online Shopping Store - Web Server Logs, Harvard Dataverse, V1, https://doi.org/10.7910/DVN/3QBYB5",
        }
        print(json.dumps(guidance, indent=2))
        return 0
    if input_path.suffix.lower() == ".zip":
        raise SystemExit("zip input must be imported first: python scripts/import_zanbil_raw.py --input <file.zip>")

    if args.dry_run:
        print(json.dumps({"status": "dry_run", "input_path": str(input_path)}, indent=2))
        return 0

    privacy = config.get("privacy", {}) or {}
    salt = os.environ.get(privacy.get("salt_env_var", "ZANBIL_HASH_SALT"), "local-zanbil-salt")
    output_root = ROOT / config.get("output_root", "data/processed/zanbil")
    result = prepare_zanbil_dataset(
        input_path=input_path,
        output_root=output_root,
        window_minutes=int(config.get("window_minutes", 1)),
        lookback_steps=int(config.get("lookback_steps", 60)),
        horizon_steps=int(config.get("horizon_steps", 15)),
        limit_lines=args.limit_lines if args.limit_lines is not None else config.get("limit_lines"),
        salt=salt,
        min_parse_rate=float(config.get("parser_min_parse_rate", 0.20)),
    )

    reports_dir = ROOT / "outputs" / "reports"
    metrics_dir = ROOT / "outputs" / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "zanbil_data_quality.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (reports_dir / "zanbil_data_quality.md").write_text(
        "\n".join(
            [
                "# Zanbil Data Quality",
                "",
                f"- Source: zanbil_web_logs",
                f"- Parsed lines: {result['parse_stats']['parsed_lines']}",
                f"- Skipped lines: {result['parse_stats']['skipped_lines']}",
                f"- Windows: {result['windows']['path']}",
                "- PII handling: client identifiers hashed; query strings stripped; raw user-agent dropped by default.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "success", "windows": result["windows"]["path"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
