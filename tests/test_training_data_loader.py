import numpy as np
import pytest

from src.training.data_loader import convert_to_train_dtype, get_data_summary, load_window_data, validate_window_data


def _write_npz(path):
    payload = {
        "X_train": np.ones((4, 3, 2), dtype=np.float16),
        "y_train": np.ones((4,), dtype=np.float16),
        "X_val": np.ones((2, 3, 2), dtype=np.float16),
        "y_val": np.ones((2,), dtype=np.float16),
        "X_test": np.ones((2, 3, 2), dtype=np.float16),
        "y_test": np.ones((2,), dtype=np.float16),
        "feature_columns": np.array(["a", "congestion_score_proxy"], dtype=object),
        "target_column": np.array("target", dtype=object),
    }
    np.savez_compressed(path, **payload)


def test_load_validate_summary_and_convert(tmp_path):
    path = tmp_path / "sample.npz"
    _write_npz(path)
    data = load_window_data(path)
    assert validate_window_data(data) is True
    summary = get_data_summary(data)
    assert summary["train"]["X_shape"] == [4, 3, 2]
    converted = convert_to_train_dtype(data, "float32")
    assert converted["X_train"].dtype == np.float32
    assert converted["y_train"].dtype == np.float32
    assert data["X_train"].dtype == np.float16


def test_validate_missing_key_fails(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, X_train=np.ones((1, 2, 1), dtype=np.float16))
    data = load_window_data(path)
    with pytest.raises(ValueError, match="missing required keys"):
        validate_window_data(data)
