from scripts.find_zanbil_raw_candidates import find_candidates


SAMPLE = '1.2.3.4 - - [01/Jan/2020:00:00:01 +0000] "GET /a HTTP/1.1" 200 123 "-" "UA"\n'


def test_find_candidate_in_temp_root(tmp_path):
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    candidate = downloads / "zanbil_access.log"
    candidate.write_text(SAMPLE, encoding="utf-8")
    candidates = find_candidates(tmp_path, ["downloads"])
    assert len(candidates) == 1
    assert candidates[0]["parser_can_parse"] is True
    assert candidates[0]["estimated_log_format"] == "nginx_or_apache_common_combined"


def test_find_candidates_skips_protected_dirs(tmp_path):
    protected = tmp_path / "outputs" / "models"
    protected.mkdir(parents=True)
    (protected / "zanbil_access.log").write_text(SAMPLE, encoding="utf-8")
    candidates = find_candidates(tmp_path, ["."])
    assert candidates == []
