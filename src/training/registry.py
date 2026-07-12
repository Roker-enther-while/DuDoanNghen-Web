"""Model registry for baselines, smoke deep models, and future models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Callable


@dataclass(frozen=True)
class ModelMetadata:
    name: str
    category: str
    implemented: bool
    module: str
    description: str
    recommended_config: str

    def to_dict(self) -> dict:
        return asdict(self)


MODEL_REGISTRY = {
    "naive_last_value": ModelMetadata(
        "naive_last_value",
        "baseline",
        True,
        "src.models.naive",
        "Predicts with the last observed congestion proxy feature in the input window.",
        "configs/training/compare_baselines.yaml",
    ),
    "moving_average": ModelMetadata(
        "moving_average",
        "baseline",
        True,
        "src.models.moving_average",
        "Predicts with a moving average over the last proxy pressure feature values.",
        "configs/training/compare_baselines.yaml",
    ),
    "lstm": ModelMetadata("lstm", "rnn", True, "src.models.lstm", "Small Keras LSTM smoke model.", "configs/training/lstm.yaml"),
    "gru": ModelMetadata("gru", "rnn", True, "src.models.gru", "Small Keras GRU smoke model.", "configs/training/gru.yaml"),
    "tcn": ModelMetadata("tcn", "convolutional", True, "src.models.tcn", "Small causal Conv1D TCN-style smoke model.", "configs/training/tcn.yaml"),
    "transformer": ModelMetadata(
        "transformer", "attention", True, "src.models.transformer", "Attention-based Transformer encoder baseline.", "configs/training/transformer.yaml"
    ),
    "tcn_lstm": ModelMetadata(
        "tcn_lstm", "hybrid", True, "src.models.tcn_lstm", "Hybrid causal TCN plus LSTM comparison model.", "configs/training/tcn_lstm.yaml"
    ),
    "tcn_attention_bilstm": ModelMetadata(
        "tcn_attention_bilstm",
        "proposed",
        True,
        "src.models.tcn_attention_bilstm",
        "Proposed causal TCN plus temporal attention plus BiLSTM model.",
        "configs/training/tcn_attention_bilstm.yaml",
    ),
    # Ablation variants
    "bilstm": ModelMetadata(
        "bilstm",
        "ablation",
        True,
        "src.models.tcn_attention_bilstm",
        "Ablation: BiLSTM only (no TCN, no Attention).",
        "configs/training/ablation/bilstm.yaml",
    ),
    "attention_bilstm": ModelMetadata(
        "attention_bilstm",
        "ablation",
        True,
        "src.models.tcn_attention_bilstm",
        "Ablation: Attention + BiLSTM only (no TCN).",
        "configs/training/ablation/attention_bilstm.yaml",
    ),
    "tcn_attention": ModelMetadata(
        "tcn_attention",
        "ablation",
        True,
        "src.models.tcn_attention_bilstm",
        "Ablation: TCN + Attention only (no BiLSTM).",
        "configs/training/ablation/tcn_attention.yaml",
    ),
}


def list_models() -> list[dict]:
    return [MODEL_REGISTRY[name].to_dict() for name in sorted(MODEL_REGISTRY)]


def get_model_metadata(model_name: str) -> ModelMetadata:
    validate_model_name(model_name)
    return MODEL_REGISTRY[model_name]


def validate_model_name(model_name: str) -> bool:
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")
    return True


def get_model_builder(model_name: str) -> Callable:
    metadata = get_model_metadata(model_name)
    module = import_module(metadata.module)
    builder = getattr(module, "build_model")
    return builder
