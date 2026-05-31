import gzip

from src.data.zanbil_logs import aggregate_zanbil_requests, parse_zanbil_log_line, read_zanbil_records


def test_parse_zanbil_hashes_ip_and_strips_query():
    line = '1.2.3.4 - - [01/Jan/2020:00:00:01 +0000] "GET /item?id=123 HTTP/1.1" 200 512 "-" "UA"'
    record = parse_zanbil_log_line(line, salt="test", drop_user_agent=True, drop_query_string=True)
    assert record is not None
    assert record["client_hash"] != "1.2.3.4"
    assert record["path"] == "/item"
    assert "user_agent" not in record
    assert record["source_id"] == "zanbil_web_logs"


def test_parse_apache_common_log():
    line = '1.2.3.4 - - [01/Jan/2020:00:00:01 +0000] "HEAD /plain HTTP/1.1" 304 -'
    record = parse_zanbil_log_line(line, salt="test")
    assert record is not None
    assert record["method"] == "HEAD"
    assert record["status"] == 304
    assert record["bytes"] == 0


def test_parse_combined_can_keep_user_agent_when_requested():
    line = '1.2.3.4 - - [01/Jan/2020:00:00:01 +0000] "GET /a HTTP/1.1" 200 1 "-" "Mozilla"'
    record = parse_zanbil_log_line(line, salt="test", drop_user_agent=False)
    assert record is not None
    assert record["user_agent"] == "Mozilla"


def test_read_zanbil_gz(tmp_path):
    path = tmp_path / "access.log.gz"
    line = '1.2.3.4 - - [01/Jan/2020:00:00:01 +0000] "GET /a HTTP/1.1" 200 1 "-" "UA"\n'
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(line)
    df, stats = read_zanbil_records(path)
    assert len(df) == 1
    assert stats.to_dict()["parse_rate"] == 1.0


def test_aggregate_zanbil_common_schema():
    lines = [
        '1.2.3.4 - - [01/Jan/2020:00:00:01 +0000] "GET /a HTTP/1.1" 200 100 "-" "UA"',
        '1.2.3.5 - - [01/Jan/2020:00:00:30 +0000] "POST /b HTTP/1.1" 500 300 "-" "UA"',
    ]
    records = [parse_zanbil_log_line(line, salt="test") for line in lines]
    df = aggregate_zanbil_requests(__import__("pandas").DataFrame(records), window_minutes=1)
    assert df.loc[0, "request_count"] == 2
    assert df.loc[0, "unique_clients"] == 2
    assert df.loc[0, "error_rate"] == 0.5
    assert "source_id" in df.columns
