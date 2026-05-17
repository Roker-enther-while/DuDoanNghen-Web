import os
import sys
from pathlib import Path

if sys.platform == "win32":
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    if tools_dir.exists():
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(tools_dir))
            except Exception:
                pass
        os.environ["PATH"] = str(tools_dir) + os.pathsep + os.environ.get("PATH", "")

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LSTM,
    LayerNormalization,
    MultiHeadAttention,
)
try:
    from .attention_layer import FeatureAttention, TemporalAttention
except ImportError:
    from attention_layer import FeatureAttention, TemporalAttention

def build_advanced_model(input_shape, horizon=5, num_filters=64, kernel_size=3, dilations=[1, 2, 4, 8], lstm_units=128, dropout_rate=0.3):
    """
    V3 SOTA Architecture: Dual-Stage Attention TCN-BiLSTM
    - Stage 1: Feature Attention (Input weighting)
    - TCN Block: Filters=64, Dilations=[1, 2, 4, 8]
    - BiLSTM Block: 128 Units
    - Stage 2: Temporal Attention (Sequence weighting)
    - Output: Multi-variate MIMO prediction
    """
    inputs = Input(shape=input_shape)

    # 1. Feature Attention (Stage 1)
    x = FeatureAttention(name="feature_attention")(inputs)

    # 2. TCN Block
    for idx, dilation_rate in enumerate(dilations):
        x = Conv1D(filters=num_filters,
                   kernel_size=kernel_size,
                   padding='causal',
                   activation='relu',
                   dilation_rate=dilation_rate,
                   name=f"tcn_conv_{idx+1}")(x)
        x = BatchNormalization(name=f"tcn_bn_{idx+1}")(x)
        x = Dropout(dropout_rate, name=f"tcn_drop_{idx+1}")(x)

    # 3. BiLSTM Block
    x = Bidirectional(LSTM(units=lstm_units, return_sequences=True), name="bilstm_layer")(x)
    x = Dropout(dropout_rate, name="bilstm_drop")(x)

    # 4. Temporal Attention (Stage 2)
    x = TemporalAttention(name="temporal_attention")(x)

    # 5. Dense Layers -> Output
    x = Dense(units=64, activation='relu', name="dense_1")(x)
    x = Dropout(dropout_rate, name="dense_drop")(x)
    
    # MIMO Output: (batch_size, horizon, 4 features)
    x = Dense(units=horizon * 4, name="output_dense_flat")(x)
    outputs = tf.keras.layers.Reshape((horizon, 4), name="output_mimo")(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="TCN_FeatureAttention_BiLSTM_TemporalAttention",
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['mae'])

    return model


build_advanced_model_legacy = build_advanced_model


def build_webtab_mhsa_model(
    input_shape,
    horizon=5,
    num_filters=64,
    kernel_size=3,
    dilations=[1, 2, 4, 8],
    lstm_units=128,
    dropout_rate=0.3,
    num_heads=4,
    key_dim=64,
    ffn_units=128,
    use_feature_attention=True,
    use_ffn=True,
):
    """
    TCN-BiLSTM with Multi-Head Self-Attention.

    The attention block uses only the encoded input window:
    query = key = value = sequence representation.
    """
    inputs = Input(shape=input_shape, name="input_window")
    x = FeatureAttention(name="feature_attention")(inputs) if use_feature_attention else inputs

    for idx, dilation_rate in enumerate(dilations):
        x = Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding="causal",
            activation="relu",
            dilation_rate=dilation_rate,
            name=f"tcn_conv_{idx+1}",
        )(x)
        x = BatchNormalization(name=f"tcn_bn_{idx+1}")(x)
        x = Dropout(dropout_rate, name=f"tcn_drop_{idx+1}")(x)

    x = Bidirectional(LSTM(units=lstm_units, return_sequences=True), name="bilstm_layer")(x)
    x = Dropout(dropout_rate, name="bilstm_drop")(x)

    attn_out = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout_rate,
        name="mhsa_temporal",
    )(query=x, key=x, value=x)
    x = Add(name="mhsa_residual")([x, attn_out])
    x = LayerNormalization(name="mhsa_residual_norm")(x)

    if use_ffn:
        ffn = Dense(ffn_units, activation="relu", name="mhsa_ffn_dense_1")(x)
        ffn = Dropout(dropout_rate, name="mhsa_ffn_drop")(ffn)
        ffn = Dense(int(x.shape[-1]), name="mhsa_ffn")(ffn)
        x = Add(name="mhsa_ffn_residual")([x, ffn])
        x = LayerNormalization(name="mhsa_ffn_norm")(x)

    x = GlobalAveragePooling1D(name="temporal_pooling")(x)
    x = Dense(units=64, activation="relu", name="dense_1")(x)
    x = Dropout(dropout_rate, name="dense_drop")(x)
    x = Dense(units=horizon * 4, name="output_dense_flat")(x)
    outputs = tf.keras.layers.Reshape((horizon, 4), name="output_mimo")(x)

    model = Model(inputs=inputs, outputs=outputs, name="TCN_BiLSTM_MHSA")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model

if __name__ == "__main__":
    # Test architecture
    SEQ_LEN = 60
    NUM_FEATURES = 13 # Match prepare_data_v2 (13-feature Multivariate)
    model = build_advanced_model((SEQ_LEN, NUM_FEATURES))
    model.summary()
