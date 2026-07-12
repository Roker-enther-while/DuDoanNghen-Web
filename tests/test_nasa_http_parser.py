import pandas as pd

from src.data.nasa_http import parse_nasa_log_line, records_from_lines


def test_parse_valid_get_line():
    line = '199.72.81.55 - - [01/Jul/1995:00:00:01 -0400] "GET /history/apollo/ HTTP/1.0" 200 6245'
    record = parse_nasa_log_line(line)
    assert record["host"] == "199.72.81.55"
    assert record["method"] == "GET"
    assert record["path"] == "/history/apollo/"
    assert record["protocol"] == "HTTP/1.0"
    assert record["status"] == 200
    assert record["bytes"] == 6245
    assert record["timestamp"] == pd.Timestamp("1995-07-01T04:00:01Z")


def test_parse_methods_status_and_dash_bytes():
    lines = [
        'h1 - - [01/Jul/1995:00:00:01 -0400] "POST /x HTTP/1.0" 302 -',
        'h2 - - [01/Jul/1995:00:00:02 -0400] "HEAD /x HTTP/1.0" 404 10',
        'h3 - - [01/Jul/1995:00:00:03 -0400] "PUT /x HTTP/1.0" 500 20',
    ]
    df, stats = records_from_lines(lines)
    assert stats.parsed_lines == 3
    assert list(df["method"]) == ["POST", "HEAD", "PUT"]
    assert list(df["status"]) == [302, 404, 500]
    assert list(df["bytes"]) == [0, 10, 20]


def test_bad_line_is_skipped():
    df, stats = records_from_lines(["bad line", 'h - - [01/Jul/1995:00:00:01 -0400] "GET / HTTP/1.0" 200 1'])
    assert stats.total_lines == 2
    assert stats.skipped_lines == 1
    assert len(df) == 1
