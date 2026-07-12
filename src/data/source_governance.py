"""Source license, citation, and provenance checks for public datasets."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REQUIRED_FIELDS = {
    "source_id",
    "source_name",
    "source_type",
    "original_url",
    "citation",
    "license_name",
    "license_url",
    "allowed_use",
    "requires_attribution",
    "contains_personal_data_risk",
    "pii_handling",
    "raw_download_default",
    "notes",
}


def load_public_sources(path: str | Path) -> list[dict[str, Any]]:
    """Load public source registry YAML."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    sources = payload.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("public_sources.yaml must contain a list under 'sources'")
    return sources


def source_hash(source: dict[str, Any]) -> str:
    """Create a stable hash for source provenance/version tracking."""
    normalized = json.dumps(source, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_required_citation(source: dict[str, Any]) -> None:
    """Fail if a source has no citation or URL usable in a report."""
    citation = str(source.get("citation", "")).strip()
    if not citation:
        raise ValueError(f"{source.get('source_id', '<unknown>')} is missing citation")


def validate_source_license(source: dict[str, Any]) -> None:
    """Fail if license metadata is incomplete."""
    missing = sorted(field for field in REQUIRED_FIELDS if field not in source or source[field] in (None, ""))
    if missing:
        raise ValueError(f"{source.get('source_id', '<unknown>')} missing required fields: {', '.join(missing)}")
    license_name = str(source.get("license_name", "")).strip().lower()
    license_url = str(source.get("license_url", "")).strip()
    if license_name in {"unknown", "unclear", "n/a"} or not license_url:
        raise ValueError(f"{source.get('source_id')} has unclear license metadata")


def validate_allowed_for_research(source: dict[str, Any]) -> None:
    """Fail if research use is not explicitly allowed."""
    allowed = source.get("allowed_use", {}) or {}
    if not bool(allowed.get("research", False)):
        raise ValueError(f"{source.get('source_id')} is not marked as allowed for research")


def validate_pii_policy(source: dict[str, Any]) -> None:
    """Ensure medium/high PII-risk sources define concrete handling rules."""
    risk = str(source.get("contains_personal_data_risk", "")).lower()
    handling = source.get("pii_handling", []) or []
    if risk in {"medium", "high"} and not handling:
        raise ValueError(f"{source.get('source_id')} has {risk} PII risk without pii_handling")


def validate_source(source: dict[str, Any]) -> dict[str, Any]:
    """Validate one source and return normalized provenance metadata."""
    validate_source_license(source)
    validate_allowed_for_research(source)
    validate_required_citation(source)
    validate_pii_policy(source)
    normalized = dict(source)
    normalized["source_hash"] = source_hash(source)
    normalized["validated_at"] = datetime.now(timezone.utc).isoformat()
    normalized["validation_status"] = "valid"
    return normalized


def build_license_manifest(sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate sources and separate usable sources from rejected candidates."""
    valid_sources: list[dict[str, Any]] = []
    rejected_sources: list[dict[str, Any]] = []
    for source in sources:
        candidate = dict(source)
        try:
            validated = validate_source(candidate)
            if candidate.get("enabled", True):
                valid_sources.append(validated)
            else:
                validated["validation_status"] = "valid_disabled"
                rejected_sources.append(validated)
        except Exception as exc:
            candidate["validation_status"] = "rejected"
            candidate["rejection_reason"] = str(exc)
            candidate["source_hash"] = source_hash(candidate)
            rejected_sources.append(candidate)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "valid_source_count": len(valid_sources),
        "rejected_source_count": len(rejected_sources),
        "sources": valid_sources,
        "rejected_sources": rejected_sources,
        "policy": {
            "no_unclear_license_in_pipeline": True,
            "raw_pii_logs_are_not_released": True,
            "synthetic_must_be_flagged": True,
        },
    }


def write_license_outputs(manifest: dict[str, Any], output_root: str | Path = "outputs") -> dict[str, str]:
    """Write JSON and Markdown license manifest."""
    output_root = Path(output_root)
    metrics_path = output_root / "metrics" / "source_license_manifest.json"
    reports_path = output_root / "reports" / "source_license_manifest.md"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    reports_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Source License Manifest",
        "",
        "Only sources with explicit license and citation metadata are allowed into the pipeline.",
        "",
        "## Valid Sources",
        "",
        "| source_id | type | license | research | derived_features | attribution | PII risk | raw default |",
        "|---|---|---|---:|---:|---:|---|---:|",
    ]
    for source in manifest["sources"]:
        allowed = source.get("allowed_use", {})
        lines.append(
            "| {source_id} | {source_type} | {license_name} | {research} | {derived} | {attrib} | {risk} | {raw} |".format(
                source_id=source["source_id"],
                source_type=source["source_type"],
                license_name=source["license_name"],
                research=bool(allowed.get("research")),
                derived=bool(allowed.get("derived_features")),
                attrib=bool(source.get("requires_attribution")),
                risk=source.get("contains_personal_data_risk"),
                raw=bool(source.get("raw_download_default")),
            )
        )
    lines.extend(["", "## Rejected Or Disabled Sources", ""])
    if manifest["rejected_sources"]:
        for source in manifest["rejected_sources"]:
            lines.append(
                f"- `{source.get('source_id')}`: {source.get('validation_status')} - "
                f"{source.get('rejection_reason', source.get('notes', 'disabled by policy'))}"
            )
    else:
        lines.append("- None")
    reports_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(metrics_path), "markdown": str(reports_path)}
