import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None

def prepare_data_v2(
    df, col_name="value", window_size=60, horizon=1, train_ratio=0.8, use_log=True, filter_noise=False
):
    """
    Hàm chuẩn hóa và tạo đặc trưng nâng cao (v4) — WEB SYSTEM MULTIVARIATE 10-FEATURES
    Features: (QPS_Load, MA10, STD30, DIFF, LAG5, DOW, HR, WKND, RT, ERR_RATE)
    """
    df = df.copy()
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce").fillna(0)
    data_raw = df[col_name].values.reshape(-1, 1)
    data_raw = np.clip(data_raw, 0, None)

    # Lọc nhiễu Savitzky-Golay (Academic standard)
    if filter_noise and savgol_filter is not None:
        val_smooth = data_raw.flatten()
        if len(val_smooth) > 11:
            val_smooth = savgol_filter(val_smooth, window_length=11, polyorder=3)
        data_raw = val_smooth.reshape(-1, 1)

    if use_log:
        data_processed = np.log1p(data_raw)
    else:
        data_processed = data_raw

    # 1. Base Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(data_processed).flatten()
    scaled_series = pd.Series(scaled_values)

    # 2. Sinh đặc trưng (Feature Engineering)
    cpu_t = scaled_values
    cpu_ma10 = scaled_series.rolling(10, min_periods=1).mean().values
    cpu_std30 = scaled_series.rolling(30, min_periods=1).std().fillna(0).values
    cpu_diff = scaled_series.diff(1).fillna(0).values
    cpu_lag5 = scaled_series.shift(5).fillna(0).values

    # 3. Trích xuất đặc trưng lịch học (Temporal features)
    dt_series = None
    if "timestamp" in df.columns:
        try:
            dt_series = pd.to_datetime(df["timestamp"])
        except: pass
    elif isinstance(df.index, pd.DatetimeIndex):
        dt_series = df.index.to_series()

    N = len(df)
    if dt_series is not None:
        day_of_week = (dt_series.dt.dayofweek / 6.0).values
        hour = (dt_series.dt.hour / 23.0).values
        is_weekend = dt_series.dt.dayofweek.isin([5, 6]).astype(float).values
    else:
        day_of_week = np.zeros(N)
        hour = np.zeros(N)
        is_weekend = np.zeros(N)

    # 4. Phase 10 FINAL: Web Server Features (Response Time & Error Rate)
    # NCKH Audit: Normalize rt and err to [0, 1] for numerical stability
    rt_raw = df['Response_Time'].values if 'Response_Time' in df.columns else np.zeros(N)
    err_raw = df['Error_Rate_5xx'].values if 'Error_Rate_5xx' in df.columns else np.zeros(N)
    
    # Robust normalization (avoiding div by zero)
    rt = (rt_raw - rt_raw.min()) / (rt_raw.max() - rt_raw.min() + 1e-7)
    err = (err_raw - err_raw.min()) / (err_raw.max() - err_raw.min() + 1e-7)
    
    all_features = np.column_stack(
        [cpu_t, cpu_ma10, cpu_std30, cpu_diff, cpu_lag5, day_of_week, hour, is_weekend, rt, err]
    )
    num_features = all_features.shape[1]  # = 10

    # 5. Đóng gói Cửa Sổ Trượt (Sliding Window) với Dự báo Đa mốc (Horizon)
    X, y = [], []
    for i in range(window_size, len(all_features) - horizon + 1):
        X.append(all_features[i - window_size : i, :])
        # Lấy một chuỗi các giá trị tải trong tương lai (Horizon)
        future_y = cpu_t[i : i + horizon]
        y.append(future_y)

    X, y = np.array(X), np.array(y)
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, y_train, X_test, y_test, scaler
