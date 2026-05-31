from pathlib import Path

from scripts.check_zanbil_readiness import check_zanbil_readiness, write_readiness_outputs


def _source_config(path: Path):
    path.write_text(
        """
sources:
  - source_id: zanbil_web_logs
    source_name: Zanbil
    source_type: real_web_log
    enabled: true
    original_url: https://doi.org/10.7910/DVN/3QBYB5
    mirror_url: ""
    citation: Citation
    license_name: CC0
    license_url: https://creativecommons.org/publicdomain/zero/1.0/
    allowed_use:
      research: true
      derived_features: true
    restrictions: []
    requires_attribution: false
    contains_personal_data_risk: medium
    pii_handling:
      - hash_ip
      - drop_query_string
      - drop_user_agent
    raw_download_default: false
    notes: ok
""",
        encoding="utf-8",
    )


def test_missing_zanbil_raw_does_not_crash_and_writes_guidance(tmp_path):
    config = tmp_path / "sources.yaml"
    _source_config(config)
    result = check_zanbil_readiness(tmp_path / "missing.log", config)
    assert result["raw_exists"] is False
    assert result["ready_for_prepare"] is False
    assert result["guidance"]
    paths = write_readiness_outputs(result, tmp_path / "outputs")
    report = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "Place the authorized raw log" in report


def test_zanbil_sample_file_parses(tmp_path):
    config = tmp_path / "sources.yaml"
    _source_config(config)
    raw = tmp_path / "access.log"
    raw.write_text(
        '1.2.3.4 - - [01/Jan/2020:00:00:01 +0000] "GET /a?x=1 HTTP/1.1" 200 123 "-" "UA"\n',
        encoding="utf-8",
    )
    result = check_zanbil_readiness(raw, config)
    assert result["raw_exists"] is True
    assert result["parser_ready"] is True
    assert result["pii_policy_ready"] is True
