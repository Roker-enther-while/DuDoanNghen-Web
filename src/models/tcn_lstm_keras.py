import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, Bidirectional

def build_tcn_lstm_model(input_shape, num_filters=[16, 32, 64], kernel_size=3, lstm_units=64, dropout_rate=0.2):
    """
    Xây dựng kiến trúc mô hình lai TCN-BiLSTM (Global Model).
    Hỗ trợ multivariate input. Có BatchNormalization sau mỗi Conv1D block.
    """
    model = Sequential(name="TCN_BiLSTM_Global")

    # ==========================================
    # 1. KHỐI TCN — với BatchNorm sau mỗi Conv
    # ==========================================
    for i, filters in enumerate(num_filters):
        dilation_rate = 2 ** i

        if i == 0:
            model.add(Conv1D(filters=filters,
                             kernel_size=kernel_size,
                             padding='causal',
                             activation='relu',
                             dilation_rate=dilation_rate,
                             input_shape=input_shape,
                             name=f"tcn_conv_{i+1}"))
        else:
            model.add(Conv1D(filters=filters,
                             kernel_size=kernel_size,
                             padding='causal',
                             activation='relu',
                             dilation_rate=dilation_rate,
                             name=f"tcn_conv_{i+1}"))

        model.add(BatchNormalization(name=f"tcn_bn_{i+1}"))
        model.add(Dropout(dropout_rate, name=f"tcn_drop_{i+1}"))

    # ==========================================
    # 2. KHỐI BiLSTM (Bidirectional LSTM)
    # ==========================================
    model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=False), name="bilstm_layer"))
    model.add(Dropout(dropout_rate, name="lstm_drop"))

    # ==========================================
    # 3. LỚP OUTPUT
    # ==========================================
    model.add(Dense(units=1, name="output_dense"))

    # ==========================================
    # 4. COMPILE
    # ==========================================
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['mae'])

    return model

# =========================================================
# SCRIPT TEST KIẾN TRÚC
# =========================================================
if __name__ == "__main__":
    SEQ_LEN = 60
    INPUT_FEATURES = 3  # multivariate: value + day_of_week + is_weekend
    INPUT_SHAPE = (SEQ_LEN, INPUT_FEATURES)

    print("Đang khởi tạo Global TCN-LSTM (BatchNorm + Dropout)...")
    model = build_tcn_lstm_model(input_shape=INPUT_SHAPE)
    model.summary()


# =========================================================
# SCRIPT SỬ DỤNG VÀ TEST KIẾN TRÚC MÔ HÌNH
# =========================================================
if __name__ == "__main__":
    # Cấu hình Dummy tương tự file tiền xử lý (Ví dụ: dữ liệu 60 phút, 1 feature)
    SEQ_LEN = 60
    INPUT_FEATURES = 1
    INPUT_SHAPE = (SEQ_LEN, INPUT_FEATURES)
    
    # Gọi hàm build mô hình
    print("Đang khởi tạo mô hình lai TCN-LSTM...")
    model = build_tcn_lstm_model(input_shape=INPUT_SHAPE)
    
    # In toàn quyền cấu trúc mạng và tổng lượng tham số
    print("\n--- CHI TIẾT KIẾN TRÚC MÔ HÌNH (NETWORK SUMMARY) ---")
    model.summary()
