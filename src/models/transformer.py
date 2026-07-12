"""Attention-based Transformer baseline for time-series regression."""

from __future__ import annotations


def default_config() -> dict:
    return {
        "d_model": 64,
        "num_heads": 4,
        "ff_dim": 128,
        "num_blocks": 1,
        "dropout": 0.1,
        "dense_units": 32,
        "learning_rate": 0.001,
    }


def _encoder_block(tf, x, d_model: int, num_heads: int, ff_dim: int, dropout: float):
    attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=max(1, d_model // num_heads))(x, x)
    attn = tf.keras.layers.Dropout(dropout)(attn)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)
    ff = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    ff = tf.keras.layers.Dropout(dropout)(ff)
    ff = tf.keras.layers.Dense(d_model)(ff)
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff)


def build_model(input_shape, config: dict | None = None):
    """Build a compact Transformer encoder baseline for sequence regression."""
    cfg = default_config()
    cfg.update(config or {})
    try:
        import tensorflow as tf
    except Exception as exc:
        raise ImportError("TensorFlow is required to build the Transformer model") from exc

    lookback_steps = int(input_shape[0])
    d_model = int(cfg["d_model"])
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(d_model)(inputs)
    positions = tf.range(start=0, limit=lookback_steps, delta=1)
    position_embedding = tf.keras.layers.Embedding(input_dim=lookback_steps, output_dim=d_model)(positions)
    x = x + position_embedding
    for _ in range(int(cfg["num_blocks"])):
        x = _encoder_block(tf, x, d_model, int(cfg["num_heads"]), int(cfg["ff_dim"]), float(cfg["dropout"]))
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(int(cfg["dense_units"]), activation="relu")(x)
    x = tf.keras.layers.Dropout(float(cfg["dropout"]))(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    model = tf.keras.Model(inputs, outputs, name="transformer_smoke")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(cfg["learning_rate"])),
        loss="mse",
        metrics=["mae"],
    )
    return model
