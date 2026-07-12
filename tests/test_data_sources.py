from pathlib import Path

import pytest

from scripts.fetch_public_data import download_source
from src.data.sources import get_source, list_sources, validate_source_name


def test_list_sources_contains_nasa_sources():
    names = {source["name"] for source in list_sources()}
    assert {"nasa_jul95", "nasa_aug95", "google_cluster_2011", "google_cluster_2019"} <= names


def test_validate_source_name_accepts_and_rejects():
    assert validate_source_name("nasa_jul95") is True
    with pytest.raises(ValueError):
        validate_source_name("missing_source")


def test_fetch_dry_run_manifest_entry_does_not_download():
    entry = download_source("nasa_jul95", dry_run=True)
    assert entry["source_name"] == "nasa_jul95"
    assert entry["dry_run"] is True
    assert "local_path" in entry
    assert "sha256" in entry
