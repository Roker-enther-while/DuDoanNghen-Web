"""Small Keras LSTM model for smoke training."""

from __future__ import annotations


def build_model(input_shape, config: dict | None = None):
    config = config or {}
    try:
        import tensorflow as tf
    except Exception as exc:
        raise ImportError("TensorFlow is required to build the LSTM model") from exc

    learning_rate = float(config.get("learning_rate", 0.001))
    units = int(config.get("units", 32))
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.LSTM(units)(inputs)
    x = tf.keras.layers.Dropout(float(config.get("dropout", 0.1)))(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    model = tf.keras.Model(inputs, outputs, name="lstm_smoke")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model
