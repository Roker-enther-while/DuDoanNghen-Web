"""Import an already-downloaded Zanbil raw log into the expected raw path."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.find_zanbil_raw_candidates import sha256_file
from src.data.source_governance import build_license_manifest, load_public_sources
from src.data.zanbil_logs import parse_zanbil_log_line, sample_zanbil_lines


def _approved_zanbil_source(config_path: Path) -> dict:
    manifest = build_license_manifest(load_public_sources(config_path))
    source = next((item for item in manifest["sources"] if item["source_id"] == "zanbil_web_logs"), None)
    if source is None:
        raise ValueError("zanbil_web_logs is not approved in source governance manifest")
    return source


def _copy_stream(src, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("wb") as out:
        shutil.copyfileobj(src, out)


def _choose_zip_member(path: Path) -> zipfile.ZipInfo:
    with zipfile.ZipFile(path) as archive:
        members = [
            item for item in archive.infolist()
            if not item.is_dir() and Path(item.filename).suffix.lower() in {".log", ".txt", ".csv", ""}
        ]
        if not members:
            raise ValueError("zip has no candidate access log member")
        members.sort(key=lambda item: (
            not any(hint in item.filename.lower() for hint in ("access", "zanbil", "web", "server", "log")),
            -item.file_size,
        ))
        return members[0]


def import_zanbil_raw(
    input_path: str | Path,
    destination: str | Path = ROOT / "data" / "raw" / "zanbil" / "access.log",
    source_config: str | Path = ROOT / "configs" / "data" / "public_sources.yaml",
) -> dict:
    """Import .log/.txt/.gz/.zip to data/raw/zanbil/access.log with provenance."""
    input_path = Path(input_path)
    destination = Path(destination)
    source = _approved_zanbil_source(Path(source_config))
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    original_sha = sha256_file(input_path)
    backup_path = None
    if destination.exists():
        existing_sha = sha256_file(destination)
        if existing_sha != original_sha:
            backup_path = destination.with_name(f"{destination.name}.bak.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(destination), str(backup_path))

    suffix = input_path.suffix.lower()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if suffix in {".log", ".txt", ".csv"}:
        shutil.copy2(input_path, destination)
        imported_member = None
    elif suffix == ".gz":
        with gzip.open(input_path, "rb") as src:
            _copy_stream(src, destination)
        imported_member = None
    elif suffix == ".zip":
        member = _choose_zip_member(input_path)
        with zipfile.ZipFile(input_path) as archive, archive.open(member) as src:
            _copy_stream(src, destination)
        imported_member = member.filename
    else:
        raise ValueError(f"Unsupported Zanbil raw input extension: {suffix}")

    imported_sha = sha256_file(destination)
    sample_lines = sample_zanbil_lines(destination, 5)
    parsed = sum(1 for line in sample_lines if parse_zanbil_log_line(line, salt="import-check") is not None)
    return {
        "original_input_path": str(input_path),
        "imported_path": str(destination),
        "backup_path": str(backup_path) if backup_path else None,
        "archive_member": imported_member,
        "original_sha256": original_sha,
        "imported_sha256": imported_sha,
        "size_bytes": int(destination.stat().st_size),
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "source_id": "zanbil_web_logs",
        "license_name": source["license_name"],
        "citation": source["citation"],
        "pii_policy": source.get("pii_handling", []),
        "parser_sample_result": {
            "sample_line_count": len(sample_lines),
            "parsed_sample_count": parsed,
            "parser_can_parse": parsed > 0,
        },
    }


def write_import_outputs(manifest: dict, output_root: str | Path = "outputs") -> dict[str, str]:
    output_root = Path(output_root)
    metrics_path = output_root / "metrics" / "zanbil_raw_import_manifest.json"
    report_path = output_root / "reports" / "zanbil_raw_import_manifest.md"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(
        "\n".join(
            [
                "# Zanbil Raw Import Manifest",
                "",
                f"- Source ID: {manifest['source_id']}",
                f"- Original input: {manifest['original_input_path']}",
                f"- Imported path: {manifest['imported_path']}",
                f"- Size bytes: {manifest['size_bytes']}",
                f"- Original sha256: {manifest['original_sha256']}",
                f"- Imported sha256: {manifest['imported_sha256']}",
                f"- Backup path: {manifest['backup_path']}",
                f"- License: {manifest['license_name']}",
                f"- Parser can parse sample: {manifest['parser_sample_result']['parser_can_parse']}",
                "- Raw IP/client values are not printed in this report.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"json": str(metrics_path), "markdown": str(report_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--destination", default="data/raw/zanbil/access.log")
    parser.add_argument("--source-config", default="configs/data/public_sources.yaml")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args(argv)
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    destination = Path(args.destination)
    if not destination.is_absolute():
        destination = ROOT / destination
    source_config = Path(args.source_config)
    if not source_config.is_absolute():
        source_config = ROOT / source_config
    manifest = import_zanbil_raw(input_path, destination, source_config)
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    paths = write_import_outputs(manifest, output_root)
    print(json.dumps({"status": "success", "imported_path": manifest["imported_path"], "outputs": paths}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
