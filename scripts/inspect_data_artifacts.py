"""Inspect processed data artifacts and write an inventory report."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _safe_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def inspect_npz(path: Path) -> dict:
    info = {
        "path": str(path),
        "file_size_bytes": int(path.stat().st_size),
        "kind": "unknown_npz",
        "error": None,
    }
    try:
        with np.load(path, allow_pickle=True) as data:
            keys = list(data.files)
            info["keys"] = keys
            if all(key in keys for key in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]):
                info["kind"] = "windows"
                for split in ["train", "val", "test"]:
                    X = data[f"X_{split}"]
                    y = data[f"y_{split}"]
                    source_key = f"source_id_{split}"
                    source_distribution = {}
                    if source_key in keys:
                        values, counts = np.unique(data[source_key].astype(str), return_counts=True)
                        source_distribution = {str(value): int(count) for value, count in zip(values, counts)}
                    info[split] = {
                        "X_shape": list(X.shape),
                        "y_shape": list(y.shape),
                        "X_dtype": str(X.dtype),
                        "y_dtype": str(y.dtype),
                        "min": _safe_float(min(np.nanmin(X), np.nanmin(y))) if X.size and y.size else None,
                        "max": _safe_float(max(np.nanmax(X), np.nanmax(y))) if X.size and y.size else None,
                        "nan_count": int(np.isnan(X).sum() + np.isnan(y).sum()),
                        "inf_count": int(np.isinf(X).sum() + np.isinf(y).sum()),
                        "source_distribution": source_distribution,
                    }
                train_n = int(data["X_train"].shape[0])
                info["artifact_scale"] = "smoke" if train_n <= 1000 else ("medium" if train_n < 50000 else "large")
                info["is_train_ready"] = True
            elif all(key in keys for key in ["features", "target"]):
                info["kind"] = "normalized_split"
                info["features_shape"] = list(data["features"].shape)
                info["target_shape"] = list(data["target"].shape)
                info["features_dtype"] = str(data["features"].dtype)
                info["target_dtype"] = str(data["target"].dtype)
    except Exception as exc:
        info["error"] = str(exc)
    return info


def write_inventory(
    root: Path,
    metrics_path: Path | None = None,
    report_path: Path | None = None,
) -> dict:
    artifacts = [inspect_npz(path) for path in sorted(root.rglob("*.npz"))]
    manifest_paths = [ROOT / "outputs" / "metrics" / "data_pipeline_manifest.json"]
    manifests = []
    for path in manifest_paths:
        if path.exists():
            try:
                manifests.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception as exc:
                manifests.append({"path": str(path), "error": str(exc)})
    inventory = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "artifacts": artifacts,
        "manifests": manifests,
        "summary": {
            "npz_count": len(artifacts),
            "window_artifact_count": sum(1 for item in artifacts if item.get("kind") == "windows"),
            "largest_window_artifact": max(
                [item for item in artifacts if item.get("kind") == "windows"],
                key=lambda item: item.get("train", {}).get("X_shape", [0])[0],
                default=None,
            ),
        },
    }
    metrics_path = metrics_path or ROOT / "outputs" / "metrics" / "data_artifact_inventory.json"
    report_path = report_path or ROOT / "outputs" / "reports" / "data_artifact_inventory.md"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    lines = ["# Data Artifact Inventory", "", f"- Generated at: {inventory['generated_at']}", f"- Root: {root}", ""]
    lines.extend(["| path | kind | scale | file_size_bytes | X_train | X_val | X_test | dtype | source_distribution_train | issues |", "|---|---|---:|---:|---:|---:|---:|---|---|---|"])
    for item in artifacts:
        train = item.get("train", {})
        val = item.get("val", {})
        test = item.get("test", {})
        issues = item.get("error") or ""
        for split in [train, val, test]:
            if split and (split.get("nan_count", 0) or split.get("inf_count", 0)):
                issues = "contains NaN/Inf"
        lines.append(
            f"| {item['path']} | {item.get('kind')} | {item.get('artifact_scale', '')} | {item['file_size_bytes']} | "
            f"{train.get('X_shape', '')} | {val.get('X_shape', '')} | {test.get('X_shape', '')} | "
            f"{train.get('X_dtype', item.get('features_dtype', ''))} | {train.get('source_distribution', {})} | {issues} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return inventory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--report-output", default=None)
    args = parser.parse_args(argv)
    metrics_output = Path(args.metrics_output) if args.metrics_output else None
    report_output = Path(args.report_output) if args.report_output else None
    if metrics_output is not None and not metrics_output.is_absolute():
        metrics_output = ROOT / metrics_output
    if report_output is not None and not report_output.is_absolute():
        report_output = ROOT / report_output
    inventory = write_inventory(Path(args.root), metrics_output, report_output)
    print(json.dumps({"artifacts": len(inventory["artifacts"]), "window_artifacts": inventory["summary"]["window_artifact_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
