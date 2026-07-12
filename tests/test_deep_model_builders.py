import numpy as np
import pytest
import torch

from src.training.torch_models import build_torch_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="PyTorch CUDA GPU is not available")
def test_deep_models_build_predict_finite_on_cuda():
    X = torch.as_tensor(np.random.default_rng(5).random((4, 6, 3), dtype=np.float32), device="cuda")
    for model_name in ["lstm", "gru", "tcn", "transformer", "tcn_lstm", "tcn_attention_bilstm"]:
        model = build_torch_model(
            model_name,
            input_shape=(6, 3),
            config={"filters": 8, "d_model": 8, "num_heads": 2, "ff_dim": 16, "lstm_units": 4, "dense_units": 8, "attention_heads": 2},
        ).to("cuda")
        pred = model(X)
        assert pred.shape == (4,)
        assert pred.is_cuda
        assert torch.isfinite(pred).all()
