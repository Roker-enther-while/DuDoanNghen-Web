"""Registry for public datasets used by the data pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class PublicDataSource:
    name: str
    url: str | None
    filename: str | None
    expected_type: str
    default_enabled: bool
    priority: str
    local_subdir: str
    notes: str

    @property
    def local_path(self) -> Path | None:
        if not self.filename:
            return None
        return Path("data") / "raw" / self.local_subdir / self.filename

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["local_path"] = str(self.local_path) if self.local_path else None
        return payload


SOURCE_REGISTRY: Dict[str, PublicDataSource] = {
    "nasa_jul95": PublicDataSource(
        name="nasa_jul95",
        url="https://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz",
        filename="NASA_access_log_Jul95.gz",
        expected_type="apache_common_log_like",
        default_enabled=True,
        priority="high",
        local_subdir="nasa_http",
        notes="NASA Kennedy Space Center HTTP access log, July 1995; about 20 MB compressed.",
    ),
    "nasa_aug95": PublicDataSource(
        name="nasa_aug95",
        url="https://ita.ee.lbl.gov/traces/NASA_access_log_Aug95.gz",
        filename="NASA_access_log_Aug95.gz",
        expected_type="apache_common_log_like",
        default_enabled=False,
        priority="high",
        local_subdir="nasa_http",
        notes="NASA Kennedy Space Center HTTP access log, August 1995; about 22 MB compressed.",
    ),
    "google_cluster_2011": PublicDataSource(
        name="google_cluster_2011",
        url=None,
        filename=None,
        expected_type="cluster_workload_trace",
        default_enabled=False,
        priority="medium",
        local_subdir="google_cluster",
        notes="Adapter skeleton only; use a prepared local sample/export before running.",
    ),
    "google_cluster_2019": PublicDataSource(
        name="google_cluster_2019",
        url=None,
        filename=None,
        expected_type="borg_workload_trace_bigquery",
        default_enabled=False,
        priority="future",
        local_subdir="google_cluster",
        notes="Do not download by default. Use BigQuery or a small export/sample; full trace is very large.",
    ),
}


def list_sources() -> List[dict]:
    """Return source metadata sorted by source name."""
    return [SOURCE_REGISTRY[name].to_dict() for name in sorted(SOURCE_REGISTRY)]


def get_source(name: str) -> PublicDataSource:
    """Return a configured source or raise a clear validation error."""
    try:
        return SOURCE_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(SOURCE_REGISTRY))
        raise ValueError(f"Unknown data source '{name}'. Available sources: {available}") from exc


def validate_source_name(name: str) -> bool:
    """Validate that a source exists."""
    get_source(name)
    return True


def default_source_names() -> List[str]:
    """Return source names enabled by default."""
    return [source.name for source in SOURCE_REGISTRY.values() if source.default_enabled]
