"""Proposed TCN-Attention-BiLSTM model for congestion forecasting."""

from __future__ import annotations


def default_config() -> dict:
    return {
        "filters": 64,
        "kernel_size": 3,
        "dilations": [1, 2, 4, 8],
        "attention_heads": 4,
        "attention_key_dim": 16,
        "lstm_units": 32,
        "dropout": 0.1,
        "dense_units": 64,
        "learning_rate": 0.001,
    }


def get_model_metadata() -> dict:
    return {
        "name": "tcn_attention_bilstm",
        "role": "proposed",
        "future": "attention weight extraction for dashboard heatmap",
    }


def _tcn_block(tf, x, filters: int, kernel_size: int, dilation: int, dropout: float):
    residual = x
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation, activation="relu")(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation, activation="relu")(x)
    if residual.shape[-1] != filters:
        residual = tf.keras.layers.Conv1D(filters, 1, padding="same")(residual)
    x = tf.keras.layers.Add()([x, residual])
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)


def build_model(input_shape, config: dict | None = None):
    """Build the proposed TCN + temporal self-attention + BiLSTM regression model."""
    cfg = default_config()
    cfg.update(config or {})
    try:
        import tensorflow as tf
    except Exception as exc:
        raise ImportError("TensorFlow is required to build the TCN-Attention-BiLSTM model") from exc

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for dilation in cfg["dilations"]:
        x = _tcn_block(tf, x, int(cfg["filters"]), int(cfg["kernel_size"]), int(dilation), float(cfg["dropout"]))

    attn = tf.keras.layers.MultiHeadAttention(
        num_heads=int(cfg["attention_heads"]),
        key_dim=int(cfg["attention_key_dim"]),
        dropout=float(cfg["dropout"]),
        name="temporal_self_attention",
    )(x, x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(int(cfg["lstm_units"])))(x)
    x = tf.keras.layers.Dense(int(cfg["dense_units"]), activation="relu")(x)
    x = tf.keras.layers.Dropout(float(cfg["dropout"]))(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    model = tf.keras.Model(inputs, outputs, name="tcn_attention_bilstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(cfg["learning_rate"])),
        loss="mse",
        metrics=["mae"],
    )
    return model
