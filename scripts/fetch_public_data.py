"""Download public raw datasets for the data pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.sources import get_source, list_sources


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(source_name: str, skip_existing: bool = True, force: bool = False, dry_run: bool = False) -> dict:
    source = get_source(source_name)
    if source.url is None or source.local_path is None:
        raise ValueError(f"Source '{source_name}' is not directly downloadable by this script")
    local_path = ROOT / source.local_path
    reused = False
    if local_path.exists() and local_path.stat().st_size > 0 and skip_existing and not force:
        reused = True
    elif not dry_run:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        last_error = None
        for attempt in range(1, 4):
            try:
                print(f"Downloading {source.name} attempt {attempt}: {source.url}")
                with urllib.request.urlopen(source.url, timeout=30) as response, local_path.open("wb") as out:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        print(f"  {local_path.name}: {out.tell()} bytes", end="\r")
                print()
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                time.sleep(attempt)
        if last_error is not None:
            raise RuntimeError(f"Failed to download {source.name}: {last_error}") from last_error

    exists = local_path.exists()
    return {
        "source_name": source.name,
        "url": source.url,
        "local_path": str(local_path),
        "file_size": int(local_path.stat().st_size) if exists else 0,
        "sha256": sha256_file(local_path) if exists and not dry_run else None,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "reused_existing": reused,
        "dry_run": dry_run,
    }


def write_manifest(entries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sources": entries}, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=["nasa_jul95"])
    parser.add_argument("--list-sources", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", default=str(ROOT / "outputs" / "metrics" / "raw_data_manifest.json"))
    args = parser.parse_args(argv)

    if args.list_sources:
        print(json.dumps(list_sources(), indent=2))
        return 0

    entries = [download_source(name, args.skip_existing, args.force, args.dry_run) for name in args.sources]
    write_manifest(entries, Path(args.manifest))
    print(f"Wrote raw data manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
