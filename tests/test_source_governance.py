import pytest

from src.data.source_governance import build_license_manifest, validate_source


def _valid_source():
    return {
        "source_id": "s",
        "source_name": "Source",
        "source_type": "real_web_log",
        "original_url": "https://example.com",
        "citation": "Example citation",
        "license_name": "CC0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "allowed_use": {"research": True, "derived_features": True},
        "restrictions": [],
        "requires_attribution": False,
        "contains_personal_data_risk": "medium",
        "pii_handling": ["hash_ip", "drop_query_string"],
        "raw_download_default": False,
        "notes": "ok",
    }


def test_source_missing_license_fails():
    source = _valid_source()
    source["license_name"] = ""
    with pytest.raises(ValueError, match="license"):
        validate_source(source)


def test_source_missing_citation_fails():
    source = _valid_source()
    source["citation"] = ""
    with pytest.raises(ValueError, match="citation"):
        validate_source(source)


def test_valid_source_passes_with_pii_policy():
    validated = validate_source(_valid_source())
    assert validated["validation_status"] == "valid"
    assert validated["source_hash"]


def test_manifest_separates_rejected_sources():
    valid = _valid_source()
    invalid = _valid_source()
    invalid["source_id"] = "bad"
    invalid["allowed_use"] = {"research": False}
    manifest = build_license_manifest([valid, invalid])
    assert manifest["valid_source_count"] == 1
    assert manifest["rejected_source_count"] == 1
