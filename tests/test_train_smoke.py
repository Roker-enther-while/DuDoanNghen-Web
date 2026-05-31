import numpy as np

from src.training.trainer import train_and_evaluate
import pytest
import torch


def _write_synthetic(path):
    rng = np.random.default_rng(10)
    payload = {
        "X_train": rng.random((20, 6, 4), dtype=np.float32).astype(np.float16),
        "y_train": rng.random(20, dtype=np.float32).astype(np.float16),
        "X_val": rng.random((8, 6, 4), dtype=np.float32).astype(np.float16),
        "y_val": rng.random(8, dtype=np.float32).astype(np.float16),
        "X_test": rng.random((8, 6, 4), dtype=np.float32).astype(np.float16),
        "y_test": rng.random(8, dtype=np.float32).astype(np.float16),
        "ts_test": np.arange(8),
        "feature_columns": np.array(["a", "b", "c", "congestion_score_proxy"], dtype=object),
        "target_column": np.array("target", dtype=object),
    }
    np.savez_compressed(path, **payload)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="PyTorch CUDA GPU is not available")
def test_train_one_deep_model_on_synthetic_data_cuda(tmp_path):
    data_path = tmp_path / "synthetic.npz"
    _write_synthetic(data_path)
    result = train_and_evaluate(
        "lstm",
        data_path,
        {
            "epochs": 1,
            "batch_size": 4,
            "max_train_samples": 12,
            "max_val_samples": 4,
            "max_test_samples": 4,
            "output_dir": str(tmp_path / "outputs"),
            "verbose": 0,
            "early_stopping": False,
            "backend": "torch",
            "require_gpu": True,
            "mixed_precision": True,
        },
    )
    assert result["status"] == "success"
    assert result["gpu_memory_plan"]["device"] == "cuda:0"
    assert result["metrics"]["rmse"] >= 0
    assert result["prediction_path"]
