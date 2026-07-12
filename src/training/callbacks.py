"""Small callback helpers for Keras training."""

from __future__ import annotations


from pathlib import Path


def make_keras_callbacks(config: dict | None = None, model_name: str | None = None, output_dir: str | Path = "outputs"):
    """Create optional Keras callbacks; returns [] if TensorFlow is unavailable."""
    config = config or {}
    try:
        import tensorflow as tf
    except Exception:
        return []
    callbacks = []
    if config.get("early_stopping", True):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(config.get("patience", config.get("early_stopping_patience", 2))),
                restore_best_weights=True,
            )
        )
    if config.get("reduce_lr_on_plateau", False):
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=float(config.get("reduce_lr_factor", 0.5)),
                patience=int(config.get("reduce_lr_patience", 2)),
                min_lr=float(config.get("min_learning_rate", 1e-6)),
            )
        )
    if config.get("model_checkpoint", True) and model_name:
        checkpoint_path = Path(output_dir) / "models" / model_name / "best_model.keras"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
            )
        )
    if config.get("csv_logger", False) and model_name:
        log_path = Path(output_dir) / "metrics" / f"{model_name}_training_log.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(tf.keras.callbacks.CSVLogger(str(log_path)))
    return callbacks
