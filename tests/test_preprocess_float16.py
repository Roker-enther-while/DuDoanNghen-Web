import numpy as np
import pandas as pd

from src.data.preprocess import chronological_split, fit_minmax_scaler, transform_split


def _frame():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="min", tz="UTC"),
            "a": np.arange(10) * 10,
            "b": np.arange(10),
            "target": np.linspace(0, 1, 10),
        }
    )


def test_chronological_split_no_overlap_and_ratios():
    train, val, test, meta = chronological_split(_frame(), 0.6, 0.2, 0.2)
    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 2
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()
    assert meta["chronological"] is True


def test_normalize_train_only_dtype_range_and_fp16_size():
    df = _frame()
    train, val, test, _ = chronological_split(df, 0.6, 0.2, 0.2)
    scaler = fit_minmax_scaler(train, ["a", "b"], "target")
    assert scaler["features"]["a"]["max"] == 50.0
    val_fp32 = transform_split(val, ["a", "b"], "target", scaler, np.float32)
    val_fp16 = transform_split(val, ["a", "b"], "target", scaler, np.float16)
    assert val_fp32["features"].dtype == np.float32
    assert val_fp32["target"].dtype == np.float32
    assert val_fp16["features"].dtype == np.float16
    assert val_fp16["target"].dtype == np.float16
    assert val_fp32["features"].min() >= 0
    assert val_fp32["features"].max() <= 1
    assert val_fp16["features"].nbytes < val_fp32["features"].nbytes
