import numpy as np
import pandas as pd

from src.data.multi_source_merge import (
    chronological_within_source_split,
    leave_one_source_out_split,
    merge_readiness,
    merge_window_artifacts,
)


def _write_windows(path, value):
    X = np.full((3, 4, 2), value, dtype=np.float16)
    y = np.full((3,), value, dtype=np.float16)
    np.savez_compressed(
        path,
        X_train=X,
        y_train=y,
        ts_train=np.arange(3),
        X_val=X,
        y_val=y,
        ts_val=np.arange(3),
        X_test=X,
        y_test=y,
        ts_test=np.arange(3),
        feature_columns=np.array(["request_count", "error_rate"], dtype=object),
        target_column=np.array("target_next_congestion_score", dtype=object),
    )


def test_merge_window_artifacts_preserves_source_id(tmp_path):
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    _write_windows(a, 0.1)
    _write_windows(b, 0.2)
    meta = merge_window_artifacts(
        [{"source_id": "a", "path": a}, {"source_id": "b", "path": b}],
        tmp_path / "merged.npz",
    )
    with np.load(meta["path"], allow_pickle=True) as data:
        assert data["X_train"].shape[0] == 6
        assert set(data["source_id_train"].tolist()) == {"a", "b"}


def test_chronological_split_no_source_leakage():
    df = pd.DataFrame(
        {
            "source_id": ["a"] * 10 + ["b"] * 10,
            "timestamp": list(pd.date_range("2020-01-01", periods=10, freq="min", tz="UTC")) * 2,
            "value": range(20),
        }
    )
    train, val, test = chronological_within_source_split(df)
    assert set(train["source_id"]) == {"a", "b"}
    assert len(train) + len(val) + len(test) == len(df)


def test_leave_one_source_out():
    df = pd.DataFrame(
        {
            "source_id": ["a"] * 5 + ["b"] * 5,
            "timestamp": list(pd.date_range("2020-01-01", periods=5, freq="min", tz="UTC")) * 2,
        }
    )
    train, val, test = leave_one_source_out_split(df, "b")
    assert set(test["source_id"]) == {"b"}
    assert set(train["source_id"]) == {"a"}
    assert set(val["source_id"]) == {"a"}


def test_nasa_only_readiness_warns_single_source():
    status = merge_readiness(["nasa_http_1995"])
    assert status["ready_for_training"] is True
    assert status["ready_for_cross_source_claim"] is False
    assert "multi_source_contains_single_source_only" in status["warnings"]


def test_two_source_readiness_allows_cross_source_claim():
    status = merge_readiness(["nasa_http_1995", "zanbil_web_logs"])
    assert status["ready_for_cross_source_claim"] is True
    assert status["warnings"] == []
