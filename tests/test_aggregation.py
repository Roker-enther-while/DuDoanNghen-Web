import pandas as pd

from src.data.nasa_http import aggregate_requests, records_from_lines


def test_aggregate_same_minute_and_fill_missing_minute():
    lines = [
        'h1 - - [01/Jul/1995:00:00:01 -0400] "GET /a HTTP/1.0" 200 100',
        'h2 - - [01/Jul/1995:00:00:02 -0400] "POST /b HTTP/1.0" 404 300',
        'h1 - - [01/Jul/1995:00:02:00 -0400] "HEAD /c HTTP/1.0" 500 600',
    ]
    records, _ = records_from_lines(lines)
    ts = aggregate_requests(records, window_minutes=1)
    assert len(ts) == 3
    first = ts.iloc[0]
    assert first["request_count"] == 2
    assert first["bytes_sum"] == 400
    assert first["bytes_mean"] == 200
    assert first["unique_hosts"] == 2
    assert first["status_2xx"] == 1
    assert first["status_4xx"] == 1
    assert first["error_count"] == 1
    assert first["error_rate"] == 0.5
    assert first["method_get"] == 1
    assert first["method_post"] == 1
    missing = ts.iloc[1]
    assert missing["request_count"] == 0
    assert missing["bytes_sum"] == 0
    assert missing["error_rate"] == 0
    assert ts.drop(columns=["timestamp"]).isna().sum().sum() == 0
