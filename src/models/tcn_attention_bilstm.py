import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Input
try:
    from .attention_layer import Attention
except ImportError:
    from attention_layer import Attention

def build_advanced_model(input_shape, horizon=1, num_filters=[32, 64, 128], kernel_size=3, lstm_units=128, dropout_rate=0.3):
    """
    Advanced TCN-Attention-BiLSTM Model Architecture.
    - TCN blocks for local pattern extraction with dilated convolutions.
    - Bidirectional LSTM for capturing long-term dependencies in both directions.
    - Attention mechanism to weight the most important time steps.
    """
    inputs = Input(shape=input_shape)

    # 1. TCN Blocks
    x = inputs
    for i, filters in enumerate(num_filters):
        dilation_rate = 2 ** i
        x = Conv1D(filters=filters,
                   kernel_size=kernel_size,
                   padding='causal',
                   activation='relu',
                   dilation_rate=dilation_rate,
                   name=f"tcn_conv_{i+1}")(x)
        x = BatchNormalization(name=f"tcn_bn_{i+1}")(x)
        x = Dropout(dropout_rate, name=f"tcn_drop_{i+1}")(x)

    # 2. BiLSTM Layer (Return Sequences for Attention)
    x = Bidirectional(LSTM(units=lstm_units, return_sequences=True), name="bilstm_layer")(x)
    x = Dropout(dropout_rate, name="lstm_drop")(x)

    # 3. Attention Mechanism
    x = Attention(name="attention_layer")(x)

    # 4. Dense Layers
    x = Dense(units=64, activation='relu', name="dense_1")(x)
    x = Dropout(dropout_rate, name="dense_drop")(x)
    outputs = Dense(units=horizon, name="output_dense")(x)

    model = Model(inputs=inputs, outputs=outputs, name="TCN_Attention_BiLSTM_Global")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['mae'])

    return model

if __name__ == "__main__":
    # Test architecture
    SEQ_LEN = 60
    NUM_FEATURES = 10 # Phase 10: 10-feature Multivariate
    model = build_advanced_model((SEQ_LEN, NUM_FEATURES))
    model.summary()
