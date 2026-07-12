"""PyTorch CUDA model builders used when TensorFlow GPU is unavailable."""

from __future__ import annotations

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, units: int = 32, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.LSTM(n_features, units, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(units, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


class GRURegressor(nn.Module):
    def __init__(self, n_features: int, units: int = 32, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.GRU(n_features, units, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(units, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels: int, filters: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, filters, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Conv1d(in_channels, filters, 1) if in_channels != filters else nn.Identity()
        self.norm = nn.LayerNorm(filters)

    def forward(self, x):
        residual = self.proj(x)
        out = nn.functional.pad(x, (self.pad, 0))
        out = torch.relu(self.conv(out))
        out = self.dropout(out)
        out = out + residual
        return self.norm(out.transpose(1, 2)).transpose(1, 2)


class TCNRegressor(nn.Module):
    def __init__(self, n_features: int, filters: int = 32, kernel_size: int = 3, dilations=None, dropout: float = 0.1):
        super().__init__()
        dilations = dilations or [1, 2, 4]
        blocks = []
        in_ch = n_features
        for dilation in dilations:
            blocks.append(CausalConvBlock(in_ch, filters, kernel_size, dilation, dropout))
            in_ch = filters
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(filters, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.blocks(x)
        x = x.mean(dim=2)
        return self.head(x).squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(self, lookback: int, n_features: int, d_model: int = 64, num_heads: int = 4, ff_dim: int = 128, num_blocks: int = 1, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, lookback, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True, activation="relu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_blocks)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.proj(x) + self.pos[:, : x.shape[1]]
        x = self.encoder(x)
        return self.head(x.mean(dim=1)).squeeze(-1)


class TCNLSTMRegressor(nn.Module):
    def __init__(self, n_features: int, filters: int = 32, kernel_size: int = 3, dilations=None, lstm_units: int = 32, dropout: float = 0.1):
        super().__init__()
        dilations = dilations or [1, 2, 4]
        blocks = []
        in_ch = n_features
        for dilation in dilations:
            blocks.append(CausalConvBlock(in_ch, filters, kernel_size, dilation, dropout))
            in_ch = filters
        self.blocks = nn.Sequential(*blocks)
        self.lstm = nn.LSTM(filters, lstm_units, batch_first=True)
        self.head = nn.Sequential(nn.Linear(lstm_units, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.blocks(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class TCNAttentionBiLSTMRegressor(nn.Module):
    def __init__(self, n_features: int, filters: int = 32, kernel_size: int = 3, dilations=None, attention_heads: int = 4, lstm_units: int = 32, dropout: float = 0.1, dense_units: int = 32):
        super().__init__()
        dilations = dilations or [1, 2, 4]
        blocks = []
        in_ch = n_features
        for dilation in dilations:
            blocks.append(CausalConvBlock(in_ch, filters, kernel_size, dilation, dropout))
            in_ch = filters
        self.blocks = nn.Sequential(*blocks)
        self.attn = nn.MultiheadAttention(filters, num_heads=attention_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(filters)
        self.bilstm = nn.LSTM(filters, lstm_units, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(lstm_units * 2, dense_units), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dense_units, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.blocks(x.transpose(1, 2)).transpose(1, 2)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm(x + attn_out)
        out, _ = self.bilstm(x)
        return self.head(out[:, -1]).squeeze(-1)


# --- Ablation variants ---

class BiLSTMRegressor(nn.Module):
    """Ablation: BiLSTM only (no TCN, no Attention)."""
    def __init__(self, n_features: int, lstm_units: int = 32, dropout: float = 0.1, dense_units: int = 32):
        super().__init__()
        self.bilstm = nn.LSTM(n_features, lstm_units, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(lstm_units * 2, dense_units), nn.ReLU(), nn.Linear(dense_units, 1), nn.Sigmoid())

    def forward(self, x):
        out, _ = self.bilstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class AttentionBiLSTMRegressor(nn.Module):
    """Ablation: Attention + BiLSTM only (no TCN)."""
    def __init__(self, n_features: int, attention_heads: int = 4, lstm_units: int = 32, dropout: float = 0.1, dense_units: int = 32):
        super().__init__()
        self.proj = nn.Linear(n_features, lstm_units * 2)
        self.attn = nn.MultiheadAttention(lstm_units * 2, num_heads=attention_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(lstm_units * 2)
        self.bilstm = nn.LSTM(lstm_units * 2, lstm_units, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(lstm_units * 2, dense_units), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dense_units, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.proj(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm(x + attn_out)
        out, _ = self.bilstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class TCNAttentionRegressor(nn.Module):
    """Ablation: TCN + Attention only (no BiLSTM)."""
    def __init__(self, n_features: int, filters: int = 32, kernel_size: int = 3, dilations=None, attention_heads: int = 4, dropout: float = 0.1, dense_units: int = 32):
        super().__init__()
        dilations = dilations or [1, 2, 4]
        blocks = []
        in_ch = n_features
        for dilation in dilations:
            blocks.append(CausalConvBlock(in_ch, filters, kernel_size, dilation, dropout))
            in_ch = filters
        self.blocks = nn.Sequential(*blocks)
        self.attn = nn.MultiheadAttention(filters, num_heads=attention_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(filters)
        self.head = nn.Sequential(nn.Linear(filters, dense_units), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dense_units, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.blocks(x.transpose(1, 2)).transpose(1, 2)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm(x + attn_out)
        return self.head(x.mean(dim=1)).squeeze(-1)


def build_torch_model(model_name: str, input_shape: tuple[int, int], config: dict) -> nn.Module:
    lookback, n_features = input_shape
    if model_name == "lstm":
        return LSTMRegressor(n_features, int(config.get("lstm_units", config.get("units", 32))), float(config.get("dropout", 0.1)))
    if model_name == "gru":
        return GRURegressor(n_features, int(config.get("gru_units", config.get("units", 32))), float(config.get("dropout", 0.1)))
    if model_name == "tcn":
        return TCNRegressor(n_features, int(config.get("filters", 32)), int(config.get("kernel_size", 3)), config.get("dilations", [1, 2, 4]), float(config.get("dropout", 0.1)))
    if model_name == "transformer":
        return TransformerRegressor(lookback, n_features, int(config.get("d_model", 64)), int(config.get("num_heads", 4)), int(config.get("ff_dim", 128)), int(config.get("num_blocks", 1)), float(config.get("dropout", 0.1)))
    if model_name == "tcn_lstm":
        return TCNLSTMRegressor(n_features, int(config.get("filters", 32)), int(config.get("kernel_size", 3)), config.get("dilations", [1, 2, 4]), int(config.get("lstm_units", 32)), float(config.get("dropout", 0.1)))
    if model_name == "tcn_attention_bilstm":
        return TCNAttentionBiLSTMRegressor(n_features, int(config.get("filters", 32)), int(config.get("kernel_size", 3)), config.get("dilations", [1, 2, 4]), int(config.get("attention_heads", 4)), int(config.get("lstm_units", 32)), float(config.get("dropout", 0.1)), int(config.get("dense_units", 32)))
    # Ablation variants
    if model_name == "bilstm":
        return BiLSTMRegressor(n_features, int(config.get("lstm_units", 32)), float(config.get("dropout", 0.1)), int(config.get("dense_units", 32)))
    if model_name == "attention_bilstm":
        return AttentionBiLSTMRegressor(n_features, int(config.get("attention_heads", 4)), int(config.get("lstm_units", 32)), float(config.get("dropout", 0.1)), int(config.get("dense_units", 32)))
    if model_name == "tcn_attention":
        return TCNAttentionRegressor(n_features, int(config.get("filters", 32)), int(config.get("kernel_size", 3)), config.get("dilations", [1, 2, 4]), int(config.get("attention_heads", 4)), float(config.get("dropout", 0.1)), int(config.get("dense_units", 32)))
    raise ValueError(f"Unsupported PyTorch model: {model_name}")
