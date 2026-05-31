"""Analyze whether a windows artifact is adequate for training and alert evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_adequacy import analyze_windows_artifact, write_adequacy_report


def _load_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--threshold", type=float, default=0.183838)
    parser.add_argument("--source-manifest", default="outputs/metrics/source_license_manifest.json")
    parser.add_argument("--labels", default=None)
    args = parser.parse_args(argv)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    manifest_path = Path(args.source_manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    source_manifest = _load_manifest(manifest_path)
    labels_path = Path(args.labels) if args.labels else None
    if labels_path is not None and not labels_path.is_absolute():
        labels_path = ROOT / labels_path
    result = analyze_windows_artifact(
        data_path,
        threshold=args.threshold,
        source_manifest=source_manifest,
        labels_path=labels_path,
    )
    paths = write_adequacy_report(result, ROOT / args.output_dir)
    print(json.dumps({"status": "success", "outputs": paths, "ready_for_training": result["ready_for_training"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
