"""Small causal Conv1D TCN-style model for smoke training."""

from __future__ import annotations


def build_model(input_shape, config: dict | None = None):
    config = config or {}
    try:
        import tensorflow as tf
    except Exception as exc:
        raise ImportError("TensorFlow is required to build the TCN model") from exc

    learning_rate = float(config.get("learning_rate", 0.001))
    filters = int(config.get("filters", 32))
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for dilation in (1, 2, 4):
        residual = x
        x = tf.keras.layers.Conv1D(filters, 3, padding="causal", dilation_rate=dilation, activation="relu")(x)
        x = tf.keras.layers.Dropout(float(config.get("dropout", 0.1)))(x)
        if residual.shape[-1] != filters:
            residual = tf.keras.layers.Conv1D(filters, 1, padding="same")(residual)
        x = tf.keras.layers.Add()([x, residual])
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    model = tf.keras.Model(inputs, outputs, name="tcn_smoke")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model
