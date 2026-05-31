"""Run the end-to-end data preparation pipeline from a YAML config."""

from __future__ import annotations

import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fetch_public_data import download_source, write_manifest
from scripts.prepare_nasa_http import prepare_nasa_http
from src.data.sources import get_source


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _sample_nasa_lines(minutes: int = 220) -> list[str]:
    lines: list[str] = []
    for minute in range(minutes):
        day = 1 + minute // (24 * 60)
        hour = (minute // 60) % 24
        mm = minute % 60
        base_count = 2 + (minute % 7)
        if minute % 45 == 0:
            base_count += 12
        for i in range(base_count):
            second = i % 60
            method = "GET" if i % 11 else "POST"
            if i % 17 == 0:
                method = "HEAD"
            status = 200
            if i % 19 == 0:
                status = 404
            if minute % 50 == 0 and i % 5 == 0:
                status = 500
            size = 1200 + (minute % 13) * 100 + i * 3
            lines.append(
                f"host{i % 23}.example.com - - "
                f"[{day:02d}/Jul/1995:{hour:02d}:{mm:02d}:{second:02d} -0400] "
                f"\"{method} /resource/{minute % 9}.html HTTP/1.0\" {status} {size}\n"
            )
    lines.append("bad line for parser smoke\n")
    return lines


def ensure_smoke_raw_data(path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = _sample_nasa_lines()
    with gzip.open(path, "wt", encoding="latin-1") as handle:
        handle.writelines(lines)
    return {
        "source_name": "nasa_smoke_sample",
        "url": None,
        "local_path": str(path),
        "file_size": int(path.stat().st_size),
        "sha256": None,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "reused_existing": False,
        "offline_sample": True,
    }


def _npz_shapes(path: str | Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {
            key: {"shape": list(data[key].shape), "dtype": str(data[key].dtype)}
            for key in data.files
            if key.startswith("X_") or key.startswith("y_")
        }


def run_pipeline(config_path: str | Path) -> dict:
    started_at = datetime.now(timezone.utc)
    config_path = Path(config_path)
    config = load_config(config_path)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ")
    warnings: list[str] = []
    status = "success"

    sources = config.get("sources", ["nasa_jul95"])
    raw_entries = []
    input_paths: list[Path] = []
    if config.get("use_sample_data", False):
        sample_path = ROOT / "data" / "raw" / "nasa_http" / "nasa_smoke_sample.log.gz"
        raw_entries.append(ensure_smoke_raw_data(sample_path))
        input_paths.append(sample_path)
    else:
        for source_name in sources:
            source = get_source(source_name)
            entry = download_source(source_name, skip_existing=True, force=False, dry_run=False)
            raw_entries.append(entry)
            if source.local_path:
                input_paths.append(ROOT / source.local_path)

    raw_manifest_path = ROOT / "outputs" / "metrics" / "raw_data_manifest.json"
    write_manifest(raw_entries, raw_manifest_path)

    output_root = Path(config.get("output_root", ROOT / "data" / "processed" / "nasa_http"))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    metadata = prepare_nasa_http(
        input_paths,
        output_root,
        window_minutes=int(config.get("window_minutes", 1)),
        limit_lines=config.get("limit_lines"),
        lookback_steps=int(config.get("lookback_steps", 60)),
        horizon_steps=int(config.get("horizon_steps", 15)),
    )

    windows_path = metadata["windows"]["path"]
    shapes = _npz_shapes(windows_path)
    finished_at = datetime.now(timezone.utc)
    manifest = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "status": status,
        "config_path": str(config_path),
        "config": config,
        "source_list": sources,
        "raw_manifest": str(raw_manifest_path),
        "output_paths": {
            "timeseries_csv": metadata["timeseries_csv"],
            "timeseries_parquet": metadata["timeseries_parquet"],
            "splits": str(output_root / "splits"),
            "normalized": str(output_root / "normalized"),
            "windows": windows_path,
            "scaler": metadata["scaler"],
            "quality_report": metadata["quality_report"],
        },
        "row_counts": {
            "raw_parsed": metadata["parse_stats"]["parsed_lines"],
            "raw_skipped": metadata["parse_stats"]["skipped_lines"],
        },
        "split_counts": metadata["split"]["counts"],
        "feature_columns": metadata["feature_columns"],
        "target_column": metadata["target_column"],
        "dtype": {"normalized_fp32": "float32", "normalized_fp16": "float16", "windows": "float16"},
        "X_y_shapes": shapes,
        "quality_summary": metadata["quality_report"],
        "warnings": warnings,
    }
    manifest_path = ROOT / "outputs" / "metrics" / "data_pipeline_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_json = json.dumps(manifest, indent=2)
    manifest_path.write_text(manifest_json, encoding="utf-8")
    (ROOT / "outputs" / "metrics" / "data_preparation_manifest.json").write_text(manifest_json, encoding="utf-8")

    report_path = ROOT / "outputs" / "reports" / "data_pipeline_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = "\n".join(
        [
            "# Data Pipeline Report",
            "",
            f"- Run ID: {run_id}",
            f"- Status: {status}",
            f"- Config: {config_path}",
            f"- Sources: {', '.join(sources)}",
            f"- Raw parsed rows: {metadata['parse_stats']['parsed_lines']}",
            f"- Raw skipped rows: {metadata['parse_stats']['skipped_lines']}",
            f"- Windows artifact: {windows_path}",
            f"- X/y shapes: {json.dumps(shapes)}",
            "",
            "No model training is performed by this data pipeline.",
        ]
    )
    report_path.write_text(report_text + "\n", encoding="utf-8")
    (ROOT / "outputs" / "reports" / "data_preparation_report.md").write_text(report_text + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    try:
        manifest = run_pipeline(args.config)
    except Exception as exc:
        manifest_path = ROOT / "outputs" / "metrics" / "data_pipeline_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "status": "failed",
                    "config_path": args.config,
                    "error": str(exc),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise
    print(json.dumps({"status": manifest["status"], "run_id": manifest["run_id"], "windows": manifest["output_paths"]["windows"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
