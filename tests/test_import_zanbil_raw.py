import gzip
import zipfile

from scripts.import_zanbil_raw import import_zanbil_raw


SAMPLE = '1.2.3.4 - - [01/Jan/2020:00:00:01 +0000] "GET /a?x=1 HTTP/1.1" 200 123 "-" "UA"\n'


def _source_config(path):
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


def test_import_log(tmp_path):
    config = tmp_path / "sources.yaml"
    _source_config(config)
    src = tmp_path / "access.log"
    dst = tmp_path / "raw" / "access.log"
    src.write_text(SAMPLE, encoding="utf-8")
    manifest = import_zanbil_raw(src, dst, config)
    assert dst.exists()
    assert manifest["parser_sample_result"]["parser_can_parse"] is True
    assert manifest["license_name"] == "CC0"
    assert manifest["imported_sha256"]


def test_import_gz(tmp_path):
    config = tmp_path / "sources.yaml"
    _source_config(config)
    src = tmp_path / "access.log.gz"
    dst = tmp_path / "raw" / "access.log"
    with gzip.open(src, "wt", encoding="utf-8") as handle:
        handle.write(SAMPLE)
    manifest = import_zanbil_raw(src, dst, config)
    assert dst.read_text(encoding="utf-8") == SAMPLE
    assert manifest["parser_sample_result"]["parsed_sample_count"] == 1


def test_import_zip_with_access_log(tmp_path):
    config = tmp_path / "sources.yaml"
    _source_config(config)
    src = tmp_path / "zanbil.zip"
    dst = tmp_path / "raw" / "access.log"
    with zipfile.ZipFile(src, "w") as archive:
        archive.writestr("nested/access.log", SAMPLE)
        archive.writestr("readme.txt", "not a log")
    manifest = import_zanbil_raw(src, dst, config)
    assert dst.exists()
    assert manifest["archive_member"] == "nested/access.log"
    assert manifest["parser_sample_result"]["parser_can_parse"] is True


def test_import_existing_destination_gets_backup(tmp_path):
    config = tmp_path / "sources.yaml"
    _source_config(config)
    src = tmp_path / "access.log"
    dst = tmp_path / "raw" / "access.log"
    dst.parent.mkdir()
    dst.write_text("old\n", encoding="utf-8")
    src.write_text(SAMPLE, encoding="utf-8")
    manifest = import_zanbil_raw(src, dst, config)
    assert manifest["backup_path"]
    assert dst.read_text(encoding="utf-8") == SAMPLE
