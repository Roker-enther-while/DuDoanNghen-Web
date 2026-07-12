import numpy as np

from src.data.windowing import create_sliding_windows


def test_sliding_window_shape_dtype_and_first_window():
    features = np.arange(30, dtype=np.float16).reshape(10, 3)
    target = np.arange(10, dtype=np.float16)
    timestamps = np.arange(10)
    X, y, ts = create_sliding_windows(features, target, timestamps, lookback_steps=4)
    assert X.shape == (6, 4, 3)
    assert y.shape == (6,)
    assert X.dtype == np.float16
    assert y.dtype == np.float16
    np.testing.assert_array_equal(X[0], features[:4])
    assert y[0] == target[4]
    assert ts[0] == timestamps[4]


def test_no_split_overlap_by_building_each_split_independently():
    train_X, _, train_ts = create_sliding_windows(np.ones((6, 2), dtype=np.float16), np.ones(6, dtype=np.float16), np.arange(6), 3)
    val_X, _, val_ts = create_sliding_windows(np.ones((6, 2), dtype=np.float16), np.ones(6, dtype=np.float16), np.arange(10, 16), 3)
    assert train_X.shape[0] == 3
    assert val_X.shape[0] == 3
    assert set(train_ts).isdisjoint(set(val_ts))
