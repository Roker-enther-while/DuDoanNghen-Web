"""Safely clean small smoke/diagnostic artifacts without touching source or large data."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


MODEL_NAMES = [
    "naive_last_value",
    "moving_average",
    "lstm",
    "gru",
    "tcn",
    "transformer",
    "tcn_lstm",
    "tcn_attention_bilstm",
]

PROTECTED_TOP_LEVEL = {"src", "scripts", "tests", "configs", "docs"}
PROTECTED_OUTPUT_MARKERS = ("full_120", "balanced", "final", "production")
KEEP_METRIC_FILES = {
    "target_distribution.json",
    "proxy_target_quality.json",
    "data_artifact_inventory.json",
    "data_pipeline_manifest.json",
    "data_preparation_manifest.json",
    "data_quality_report.json",
    "raw_data_manifest.json",
    "tcn_attention_bilstm_tuning.json",
}
KEEP_REPORT_FILES = {
    "target_distribution.md",
    "proxy_target_quality.md",
    "data_artifact_inventory.md",
    "data_pipeline_report.md",
    "data_preparation_report.md",
    "data_quality_report.md",
    "tcn_attention_bilstm_tuning.md",
}


@dataclass
class CleanupItem:
    path: str
    kind: str
    bytes: int
    reason: str

    def to_dict(self) -> dict:
        return {"path": self.path, "kind": self.kind, "bytes": self.bytes, "reason": self.reason}


def _resolve(root: Path, rel_or_abs: str | Path) -> Path:
    path = Path(rel_or_abs)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    return int(sum(p.stat().st_size for p in path.rglob("*") if p.is_file()))


def _load_balanced_references(root: Path) -> set[Path]:
    refs: set[Path] = set()
    for manifest in (root / "outputs" / "metrics").glob("*balanced*model_comparison.json"):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in payload.get("models", []):
            for key in ["model_path", "prediction_path", "history_path", "metrics_path"]:
                if item.get(key):
                    refs.add(_resolve(root, item[key]))
            model = item.get("model")
            if model:
                for suffix in ["metrics.json", "history.json", "training_log.csv"]:
                    refs.add(_resolve(root, Path("outputs") / "metrics" / f"{model}_{suffix}"))
    return refs


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def is_protected(root: Path, path: Path, balanced_refs: set[Path]) -> tuple[bool, str | None]:
    resolved = path.resolve()
    try:
        rel_parts = resolved.relative_to(root.resolve()).parts
    except ValueError:
        return True, "outside_repo_root"
    if not rel_parts:
        return True, "repo_root"
    if rel_parts[0] in PROTECTED_TOP_LEVEL:
        return True, f"protected_top_level_{rel_parts[0]}"
    if len(rel_parts) >= 2 and rel_parts[0] == "data" and rel_parts[1] in {"raw", "processed"}:
        return True, "protected_data"
    rel_text = str(Path(*rel_parts)).replace("\\", "/").lower()
    if any(marker in rel_text for marker in PROTECTED_OUTPUT_MARKERS):
        return True, "protected_full_balanced_final_marker"
    for ref in balanced_refs:
        if resolved == ref.resolve() or _is_under(ref, resolved):
            return True, "referenced_by_balanced_manifest"
    return False, None


def _add_candidate(root: Path, items: list[CleanupItem], skipped: list[dict], path: Path, reason: str, balanced_refs: set[Path]) -> None:
    if not path.exists():
        skipped.append({"path": str(path), "reason": "skipped_missing"})
        return
    protected, why = is_protected(root, path, balanced_refs)
    if protected:
        skipped.append({"path": str(path), "reason": f"skipped_protected:{why}"})
        return
    kind = "dir" if path.is_dir() else "file"
    items.append(CleanupItem(path=str(path), kind=kind, bytes=_path_size(path), reason=reason))


def build_cleanup_plan(root: str | Path = ROOT) -> dict:
    root = Path(root).resolve()
    balanced_refs = _load_balanced_references(root)
    items: list[CleanupItem] = []
    skipped: list[dict] = []

    # Old Keras smoke/checkpoint files inside root model folders. Keep .pt or baseline files referenced by balanced.
    for model in MODEL_NAMES:
        model_dir = root / "outputs" / "models" / model
        for filename in ["model.keras", "best_model.keras"]:
            _add_candidate(root, items, skipped, model_dir / filename, "old keras smoke model not used by torch balanced run", balanced_refs)

    # Untagged quick/diagnostic comparison files superseded by balanced outputs.
    if (root / "outputs" / "metrics" / "balanced_model_comparison.json").exists():
        _add_candidate(root, items, skipped, root / "outputs" / "metrics" / "model_comparison.json", "untagged quick/diagnostic comparison superseded by balanced", balanced_refs)
        _add_candidate(root, items, skipped, root / "outputs" / "reports" / "model_comparison.md", "untagged quick/diagnostic report superseded by balanced", balanced_refs)
        _add_candidate(root, items, skipped, root / "outputs" / "web" / "model_dashboard_payload.json", "untagged quick/diagnostic dashboard payload superseded by balanced", balanced_refs)
        _add_candidate(root, items, skipped, root / "outputs" / "web" / "model_comparison_table.csv", "untagged quick/diagnostic dashboard table superseded by balanced", balanced_refs)

    # Root metrics files that are not protected by balanced refs or keep-list.
    metrics_dir = root / "outputs" / "metrics"
    if metrics_dir.exists():
        for path in metrics_dir.glob("*"):
            if path.name in KEEP_METRIC_FILES:
                continue
            if any(marker in path.name.lower() for marker in PROTECTED_OUTPUT_MARKERS):
                continue
            if path.name.endswith("_training_log.csv") or path.name.endswith("_metrics.json") or path.name.endswith("_history.json"):
                _add_candidate(root, items, skipped, path, "root smoke/diagnostic per-model metric not protected", balanced_refs)

    # Root reports with smoke/quick/diagnostic names, excluding protected keep-list.
    reports_dir = root / "outputs" / "reports"
    if reports_dir.exists():
        for path in reports_dir.glob("*.md"):
            if path.name in KEEP_REPORT_FILES:
                continue
            lowered = path.name.lower()
            if any(marker in lowered for marker in PROTECTED_OUTPUT_MARKERS):
                continue
            if any(token in lowered for token in ["smoke", "quick", "diagnostic"]):
                _add_candidate(root, items, skipped, path, "smoke/quick/diagnostic report", balanced_refs)

    dedup: dict[str, CleanupItem] = {item.path: item for item in items}
    items = list(dedup.values())
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "delete_items": [item.to_dict() for item in sorted(items, key=lambda x: x.path)],
        "skipped": skipped,
        "total_bytes": int(sum(item.bytes for item in items)),
        "total_mb": float(sum(item.bytes for item in items) / (1024 * 1024)),
    }


def write_plan(plan: dict, root: str | Path = ROOT) -> tuple[str, str]:
    root = Path(root)
    metrics = root / "outputs" / "metrics" / "cleanup_small_test_artifacts_plan.json"
    report = root / "outputs" / "reports" / "cleanup_small_test_artifacts_plan.md"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Cleanup Small Test Artifacts Plan",
        "",
        f"- Generated at: {plan['generated_at']}",
        f"- Items to delete: {len(plan['delete_items'])}",
        f"- Total bytes: {plan['total_bytes']}",
        f"- Total MB: {plan['total_mb']:.3f}",
        "",
        "| kind | bytes | reason | path |",
        "|---|---:|---|---|",
    ]
    for item in plan["delete_items"]:
        lines.append(f"| {item['kind']} | {item['bytes']} | {item['reason']} | {item['path']} |")
    lines.extend(["", "## Skipped"])
    for item in plan["skipped"]:
        lines.append(f"- {item['reason']}: {item['path']}")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(metrics), str(report)


def execute_delete(plan: dict, root: str | Path = ROOT) -> dict:
    root = Path(root).resolve()
    deleted_files: list[str] = []
    deleted_dirs: list[str] = []
    errors: list[dict] = []
    skipped_protected: list[dict] = []
    balanced_refs = _load_balanced_references(root)
    total_bytes = 0
    for item in plan["delete_items"]:
        path = Path(item["path"]).resolve()
        protected, why = is_protected(root, path, balanced_refs)
        if protected:
            skipped_protected.append({"path": str(path), "reason": why})
            continue
        try:
            total_bytes += _path_size(path)
            if path.is_dir():
                shutil.rmtree(path)
                deleted_dirs.append(str(path))
            elif path.exists():
                path.unlink()
                deleted_files.append(str(path))
        except Exception as exc:
            errors.append({"path": str(path), "error": str(exc)})
    manifest = {
        "deleted_at": datetime.now(timezone.utc).isoformat(),
        "deleted_files": deleted_files,
        "deleted_dirs": deleted_dirs,
        "skipped_protected_paths": skipped_protected,
        "total_deleted_bytes": int(total_bytes),
        "total_deleted_mb": float(total_bytes / (1024 * 1024)),
        "errors": errors,
    }
    metrics = root / "outputs" / "metrics" / "cleanup_small_test_artifacts_deleted.json"
    report = root / "outputs" / "reports" / "cleanup_small_test_artifacts_deleted.md"
    metrics.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Cleanup Small Test Artifacts Deleted",
        "",
        f"- Deleted at: {manifest['deleted_at']}",
        f"- Deleted files: {len(deleted_files)}",
        f"- Deleted dirs: {len(deleted_dirs)}",
        f"- Total bytes: {manifest['total_deleted_bytes']}",
        f"- Total MB: {manifest['total_deleted_mb']:.3f}",
        "",
        "## Deleted Files",
    ]
    lines.extend([f"- {path}" for path in deleted_files] or ["- None"])
    lines.extend(["", "## Deleted Directories"])
    lines.extend([f"- {path}" for path in deleted_dirs] or ["- None"])
    lines.extend(["", "## Protected Skips"])
    lines.extend([f"- {item['reason']}: {item['path']}" for item in skipped_protected] or ["- None"])
    lines.extend(["", "## Errors"])
    lines.extend([f"- {item['path']}: {item['error']}" for item in errors] or ["- None"])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-delete", action="store_true")
    parser.add_argument("--repo-root", default=str(ROOT))
    args = parser.parse_args(argv)
    if args.dry_run and args.confirm_delete:
        raise SystemExit("Use only one of --dry-run or --confirm-delete")
    if not args.dry_run and not args.confirm_delete:
        args.dry_run = True
    root = Path(args.repo_root).resolve()
    plan = build_cleanup_plan(root)
    plan_json, plan_md = write_plan(plan, root)
    print(json.dumps({"plan_json": plan_json, "plan_md": plan_md, "items": len(plan["delete_items"]), "total_mb": plan["total_mb"]}, indent=2, ensure_ascii=False))
    if args.confirm_delete:
        manifest = execute_delete(plan, root)
        print(json.dumps({"deleted_files": len(manifest["deleted_files"]), "deleted_dirs": len(manifest["deleted_dirs"]), "total_deleted_mb": manifest["total_deleted_mb"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
