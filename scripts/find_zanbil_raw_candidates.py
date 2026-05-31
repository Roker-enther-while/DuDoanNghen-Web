"""Find local Zanbil raw log candidates without downloading external data."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.zanbil_logs import parse_zanbil_log_line, sample_zanbil_lines


PROTECTED_DIRS = {
    ".git",
    "outputs",
    "outputs/models",
    "outputs/predictions",
    "data/processed",
    "venv",
    ".venv",
    "node_modules",
    "__pycache__",
}
SEARCH_EXTENSIONS = {".log", ".txt", ".gz", ".zip", ".csv"}
NAME_HINTS = ("access", "zanbil", "web", "server", "log", "access_log")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_protected(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = path.as_posix()
    parts = set(rel.split("/"))
    return any(protected in rel for protected in PROTECTED_DIRS) or bool(parts & PROTECTED_DIRS)


def _name_matches(path: Path) -> bool:
    text = path.name.lower()
    return any(hint in text for hint in NAME_HINTS)


def _sample_zip(path: Path, max_lines: int = 5) -> tuple[list[str], str | None]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = [
                member for member in archive.infolist()
                if not member.is_dir() and Path(member.filename).suffix.lower() in {".log", ".txt", ".csv", ""}
            ]
            if not members:
                return [], None
            members.sort(key=lambda item: (
                not any(hint in item.filename.lower() for hint in NAME_HINTS),
                -item.file_size,
            ))
            chosen = members[0]
            lines = []
            with archive.open(chosen) as handle:
                for _, raw in zip(range(max_lines), handle):
                    lines.append(raw.decode("utf-8", errors="replace").rstrip("\n"))
            return lines, chosen.filename
    except Exception:
        return [], None


def sample_candidate(path: Path, max_lines: int = 5) -> tuple[list[str], str | None]:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        return _sample_zip(path, max_lines)
    if suffix in {".log", ".txt", ".gz", ".csv"}:
        try:
            if suffix == ".csv":
                with path.open("rt", encoding="utf-8", errors="replace") as handle:
                    return [line.rstrip("\n") for _, line in zip(range(max_lines), handle)], None
            return sample_zanbil_lines(path, max_lines), None
        except (OSError, gzip.BadGzipFile, UnicodeError):
            return [], None
    return [], None


def inspect_candidate(path: Path) -> dict:
    lines, archive_member = sample_candidate(path)
    parsed = sum(1 for line in lines if parse_zanbil_log_line(line, salt="candidate-check") is not None)
    lower_path = path.as_posix().lower()
    wrong_known_source = "nasa" in lower_path or "nasa_http" in lower_path
    parser_can_parse = parsed > 0 and not wrong_known_source
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "extension": path.suffix.lower(),
        "sha256": sha256_file(path),
        "sample_first_lines_masked": [line[:240] for line in lines],
        "parser_can_parse": parser_can_parse,
        "parsed_sample_count": parsed,
        "estimated_log_format": "nginx_or_apache_common_combined" if parser_can_parse else "unknown",
        "archive_member": archive_member,
        "reason_if_rejected": (
            ""
            if parser_can_parse
            else (
                "known_nasa_source_not_zanbil"
                if wrong_known_source
                else "sample_lines_not_parseable_as_common_or_combined_access_log"
            )
        ),
    }


def find_candidates(root: str | Path = ROOT, search_roots: list[str] | None = None) -> list[dict]:
    root = Path(root)
    search_roots = search_roots or ["data/raw/zanbil", "data/raw", "datasets", "downloads", "."]
    candidates: list[dict] = []
    seen: set[Path] = set()
    for relative in search_roots:
        base = root / relative
        if not base.exists() or _is_protected(base, root):
            continue
        for path in base.rglob("*"):
            if path in seen or not path.is_file() or _is_protected(path, root):
                continue
            if path.suffix.lower() not in SEARCH_EXTENSIONS or not _name_matches(path):
                continue
            seen.add(path)
            candidates.append(inspect_candidate(path))
    candidates.sort(key=lambda item: (not item["parser_can_parse"], -item["size_bytes"], item["path"]))
    return candidates


def write_candidate_outputs(candidates: list[dict], output_root: str | Path = "outputs") -> dict[str, str]:
    output_root = Path(output_root)
    metrics_path = output_root / "metrics" / "zanbil_raw_candidates.json"
    report_path = output_root / "reports" / "zanbil_raw_candidates.md"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(candidates),
        "parseable_candidate_count": sum(1 for item in candidates if item["parser_can_parse"]),
        "candidates": candidates,
        "guidance": "If no parseable candidate is found, place the authorized raw log at data/raw/zanbil/access.log.",
    }
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Zanbil Raw Candidates",
        "",
        f"- Candidate count: {payload['candidate_count']}",
        f"- Parseable candidate count: {payload['parseable_candidate_count']}",
        "",
        "| path | size_bytes | extension | parser_can_parse | parsed_sample_count | reason |",
        "|---|---:|---|---:|---:|---|",
    ]
    for item in candidates:
        lines.append(
            f"| {item['path']} | {item['size_bytes']} | {item['extension']} | "
            f"{item['parser_can_parse']} | {item['parsed_sample_count']} | {item['reason_if_rejected']} |"
        )
    if not candidates:
        lines.extend(["", "No local Zanbil raw candidate found. Place the authorized raw log at `data/raw/zanbil/access.log`."])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(metrics_path), "markdown": str(report_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args(argv)
    root = Path(args.root)
    candidates = find_candidates(root)
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    paths = write_candidate_outputs(candidates, output_root)
    print(json.dumps({"status": "success", "candidate_count": len(candidates), "outputs": paths}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
