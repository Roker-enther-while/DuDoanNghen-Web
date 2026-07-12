import numpy as np
import pandas as pd

from src.data.nasa_http import add_congestion_proxy_target


def test_congestion_proxy_range_shift_and_no_nan():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("1995-07-01", periods=20, freq="min", tz="UTC"),
            "request_count": np.arange(20),
            "bytes_sum": np.arange(20) * 100,
            "bytes_mean": np.arange(20) * 10,
            "bytes_max": np.arange(20) * 20,
            "bytes_p95": np.arange(20) * 15,
            "unique_hosts": np.arange(20) % 5 + 1,
            "status_2xx": np.ones(20),
            "status_3xx": np.zeros(20),
            "status_4xx": np.zeros(20),
            "status_5xx": np.zeros(20),
            "error_count": np.zeros(20),
            "error_rate": np.zeros(20),
            "method_get": np.ones(20),
            "method_post": np.zeros(20),
            "method_head": np.zeros(20),
            "method_other": np.zeros(20),
            "throughput_bytes_per_min": np.arange(20) * 100,
        }
    )
    out, meta = add_congestion_proxy_target(df, horizon_steps=3)
    assert out["congestion_score_proxy"].between(0, 1).all()
    assert out["target_next_congestion_score"].between(0, 1).all()
    assert not out.isna().any().any()
    assert np.isclose(out.loc[0, "target_next_congestion_score"], out.loc[3, "congestion_score_proxy"])
    assert meta["is_proxy_target"] is True
