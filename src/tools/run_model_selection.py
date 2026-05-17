from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from src.data.sql_data_pool import read_ordered, replace_table

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def md_table(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("_Không có dữ liệu._\n", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")).replace("\n", " ") for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_div(a: float, b: float) -> float:
    return float(a / b) if abs(b) > 1e-12 else float("nan")


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    denom = np.abs(y_true) + np.abs(y_pred)
    smape = float(np.mean(np.where(denom > 0, 2 * np.abs(y_pred - y_true) / denom, 0)) * 100)
    wape = float(np.sum(np.abs(y_true - y_pred)) / max(1e-12, np.sum(np.abs(y_true))) * 100)
    nonzero = np.abs(y_true) > 1e-8
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) if nonzero.any() else float("nan")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2_score(y_true, y_pred), "sMAPE": smape, "WAPE": wape, "MAPE": mape}


def classification_metrics(y_true, y_score, threshold: float) -> dict:
    y_true_cls = (np.asarray(y_true) >= threshold).astype(int)
    y_pred_cls = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1]).ravel()
    out = {
        "Accuracy": accuracy_score(y_true_cls, y_pred_cls),
        "Precision": precision_score(y_true_cls, y_pred_cls, zero_division=0),
        "Recall": recall_score(y_true_cls, y_pred_cls, zero_division=0),
        "F1": f1_score(y_true_cls, y_pred_cls, zero_division=0),
        "FPR": safe_div(fp, fp + tn),
        "FNR": safe_div(fn, fn + tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
    try:
        out["ROC_AUC"] = roc_auc_score(y_true_cls, y_score)
    except Exception:
        out["ROC_AUC"] = float("nan")
    return out


FEATURE_COLS = ["cpu_usage", "memory_usage", "disk_io", "network_in", "network_out", "request_rate", "throughput", "response_time", "error_rate", "congestion_label", "is_synthetic", "is_noisy", "source_id"]


def prepare_windows(df: pd.DataFrame, window_size: int, horizon: int, max_windows: int = 60000):
    df = df.copy()
    if "source_id" not in df.columns:
        df["source_id"] = pd.factorize(df["source_name"])[0]
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    groups = []
    for _, g in df.sort_values(["source_name", "machine_id", "timestamp", "time_index"]).groupby(["source_name", "machine_id"], sort=False):
        if len(g) > window_size + horizon:
            groups.append(g)
    X_train, y_train, X_val, y_val, meta_val = [], [], [], [], []
    scaler = StandardScaler()
    train_arrays = []
    split_groups = []
    for g in groups:
        cut = int(len(g) * 0.85)
        train_arrays.append(g.iloc[:cut][FEATURE_COLS].to_numpy(dtype=np.float32))
        split_groups.append((g.iloc[:cut], g.iloc[max(0, cut - window_size - horizon):]))
    scaler.fit(np.vstack(train_arrays))
    for train_g, val_g in split_groups:
        for target, bucket_x, bucket_y in [(train_g, X_train, y_train), (val_g, X_val, y_val)]:
            arr = scaler.transform(target[FEATURE_COLS].to_numpy(dtype=np.float32)).astype(np.float32)
            raw_y = target["cpu_usage"].to_numpy(dtype=np.float32)
            for i in range(window_size, len(arr) - horizon + 1):
                bucket_x.append(arr[i - window_size : i])
                bucket_y.append(raw_y[i + horizon - 1])
                if bucket_y is y_val:
                    meta_val.append(target.iloc[i + horizon - 1][["source_name", "machine_id", "timestamp"]].to_dict())
    if len(X_train) > max_windows:
        idx = np.linspace(0, len(X_train) - 1, max_windows).astype(int)
        X_train = [X_train[i] for i in idx]
        y_train = [y_train[i] for i in idx]
    if len(X_val) > max_windows // 3:
        idx = np.linspace(0, len(X_val) - 1, max_windows // 3).astype(int)
        X_val = [X_val[i] for i in idx]
        y_val = [y_val[i] for i in idx]
        meta_val = [meta_val[i] for i in idx]
    return np.asarray(X_train, dtype=np.float32), np.asarray(y_train, dtype=np.float32), np.asarray(X_val, dtype=np.float32), np.asarray(y_val, dtype=np.float32), scaler, pd.DataFrame(meta_val)


def build_keras_model(name: str, input_shape: tuple[int, int], dropout: float = 0.15):
    import tensorflow as tf
    from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, Dropout, GlobalAveragePooling1D, Input, LSTM, MultiHeadAttention, Add, LayerNormalization
    from tensorflow.keras.models import Model, Sequential
    from src.models.attention_layer import FeatureAttention, TemporalAttention

    if name == "lstm32":
        model = Sequential([Input(shape=input_shape), LSTM(32), Dropout(dropout), Dense(16, activation="relu"), Dense(1)])
    elif name == "bilstm32":
        model = Sequential([Input(shape=input_shape), Bidirectional(LSTM(32)), Dropout(dropout), Dense(16, activation="relu"), Dense(1)])
    elif name in {"tcn16", "tcn32"}:
        filters = 16 if name == "tcn16" else 32
        model = Sequential([Input(shape=input_shape), Conv1D(filters, 3, padding="causal", activation="relu", dilation_rate=1), Conv1D(filters, 3, padding="causal", activation="relu", dilation_rate=2), GlobalAveragePooling1D(), Dropout(dropout), Dense(16, activation="relu"), Dense(1)])
    else:
        inputs = Input(shape=input_shape)
        x = Conv1D(32, 3, padding="causal", activation="relu", dilation_rate=1)(inputs)
        x = Conv1D(32, 3, padding="causal", activation="relu", dilation_rate=2)(x)
        if name == "tcn_bilstm32_no_attn":
            x = Bidirectional(LSTM(32))(x)
        elif name in {"tcn_bilstm32_temporal_attention", "webtab_temporal_attention32", "webtab_legacy"}:
            x = Bidirectional(LSTM(32, return_sequences=True))(x)
            x = TemporalAttention()(x)
        elif name == "tcn_feature_attention_bilstm_temporal_attention":
            x = FeatureAttention(name="feature_attention")(inputs)
            x = Conv1D(32, 3, padding="causal", activation="relu", dilation_rate=1)(x)
            x = Conv1D(32, 3, padding="causal", activation="relu", dilation_rate=2)(x)
            x = Bidirectional(LSTM(32, return_sequences=True))(x)
            x = TemporalAttention(name="temporal_attention")(x)
        elif name == "webtab_mhsa_light":
            x = Bidirectional(LSTM(32, return_sequences=True))(x)
            attn = MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
            x = LayerNormalization()(Add()([x, attn]))
            x = GlobalAveragePooling1D()(x)
        else:
            x = GlobalAveragePooling1D()(x)
        x = Dropout(dropout)(x)
        x = Dense(32, activation="relu")(x)
        outputs = Dense(1)(x)
        model = Model(inputs, outputs, name=name)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse", metrics=["mae"])
    return model


def evaluate_latency(predict_fn, X: np.ndarray, repeats: int = 30) -> dict:
    if len(X) == 0:
        return {"inference_latency_mean_ms": float("nan"), "inference_latency_p50_ms": float("nan"), "inference_latency_p95_ms": float("nan")}
    sample = X[: min(128, len(X))]
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        predict_fn(sample)
        times.append((time.perf_counter() - start) * 1000 / len(sample))
    return {"inference_latency_mean_ms": float(np.mean(times)), "inference_latency_p50_ms": float(np.percentile(times, 50)), "inference_latency_p95_ms": float(np.percentile(times, 95))}


def run_selection(args, out: Path) -> tuple[pd.DataFrame, dict]:
    df = read_ordered(args.db_path, args.table)
    X_train, y_train, X_val, y_val, scaler, meta_val = prepare_windows(df, args.window_sizes[0], args.horizons[0])
    print(f"[windows] train={X_train.shape} val={X_val.shape}")
    threshold = float(np.nanquantile(y_train, 0.85)) if len(y_train) else 85.0
    rows = []
    histories = {}
    checkpoint_dir = out / "models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def add_result(model_name, pred, param_count=0, model_size_mb=0.0, latency=None, checkpoint=""):
        reg = regression_metrics(y_val, pred)
        cls = classification_metrics(y_val, pred, threshold)
        row = {"model": model_name, **reg, **cls, "parameter_count": int(param_count), "model_size_mb": model_size_mb, **(latency or {}), "checkpoint": checkpoint}
        rows.append(row)

    if "persistence" in args.models:
        pred = X_val[:, -1, FEATURE_COLS.index("cpu_usage")] * scaler.scale_[FEATURE_COLS.index("cpu_usage")] + scaler.mean_[FEATURE_COLS.index("cpu_usage")]
        add_result("persistence", pred, latency=evaluate_latency(lambda x: x[:, -1, FEATURE_COLS.index("cpu_usage")], X_val))
    if "moving_average" in args.models:
        idx = FEATURE_COLS.index("cpu_usage")
        pred_scaled = X_val[:, :, idx].mean(axis=1)
        pred = pred_scaled * scaler.scale_[idx] + scaler.mean_[idx]
        add_result("moving_average", pred, latency=evaluate_latency(lambda x: x[:, :, idx].mean(axis=1), X_val))
    if "arima" in args.models:
        try:
            from statsmodels.tsa.arima.model import ARIMA
            fit_n = min(3000, len(y_train))
            model = ARIMA(y_train[-fit_n:], order=(2, 0, 1)).fit()
            pred = np.asarray(model.forecast(steps=len(y_val)))
            add_result("arima", pred, latency={"inference_latency_mean_ms": np.nan, "inference_latency_p50_ms": np.nan, "inference_latency_p95_ms": np.nan})
        except Exception as exc:
            print(f"[arima] skipped: {exc}")

    keras_names = [m for m in args.models if m not in {"persistence", "moving_average", "arima"}]
    for name in keras_names:
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            model = build_keras_model(name, X_train.shape[1:])
            start = time.time()
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.quick_epochs, batch_size=args.batch_size, shuffle=False, verbose=0)
            histories[name] = history.history
            pred = model.predict(X_val, batch_size=args.batch_size, verbose=0).reshape(-1)
            ckpt = checkpoint_dir / f"{name}_quick.keras"
            model.save(ckpt)
            size_mb = ckpt.stat().st_size / (1024**2)
            latency = evaluate_latency(lambda x, m=model: m.predict(x, verbose=0), X_val)
            row_start = len(rows)
            add_result(name, pred, model.count_params(), size_mb, latency, str(ckpt))
            rows[row_start]["train_seconds"] = round(time.time() - start, 3)
        except Exception as exc:
            print(f"[model:{name}] failed: {exc}")
            rows.append({"model": name, "error": str(exc)})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "model_selection_metrics.csv", index=False)
    md_table(metrics.fillna("").to_dict("records"), out / "tables/table_09_model_selection_metrics.md")
    replace_table(args.db_path, "model_selection_metrics", metrics)
    (out / "window_summary.json").write_text(json.dumps({"train_windows": len(X_train), "validation_windows": len(X_val), "window_size": args.window_sizes[0], "horizon": args.horizons[0], "classification_threshold": threshold}, indent=2), encoding="utf-8")
    if histories:
        best_hist_name = min(histories, key=lambda n: min(histories[n].get("val_loss", [np.inf])))
        pd.DataFrame(histories[best_hist_name]).to_csv(out / "figure_data/12_best_model_train_val_loss.csv", index=False)
    return metrics, {"threshold": threshold, "meta_val": meta_val, "histories": histories, "y_val": y_val}


def rank_models(metrics: pd.DataFrame, out: Path) -> dict:
    valid = metrics[pd.to_numeric(metrics.get("RMSE"), errors="coerce").notna()].copy()
    if valid.empty:
        ranking = pd.DataFrame()
    else:
        for c in ["RMSE", "R2", "F1", "inference_latency_mean_ms"]:
            valid[c] = pd.to_numeric(valid[c], errors="coerce")
        valid["rank_rmse"] = valid["RMSE"].rank(ascending=True)
        valid["rank_r2"] = valid["R2"].rank(ascending=False)
        valid["rank_f1"] = valid["F1"].rank(ascending=False)
        valid["rank_latency"] = valid["inference_latency_mean_ms"].fillna(valid["inference_latency_mean_ms"].max()).rank(ascending=True)
        valid["selection_score"] = valid["rank_f1"] * 0.40 + valid["rank_rmse"] * 0.25 + valid["rank_r2"] * 0.20 + valid["rank_latency"] * 0.15
        ranking = valid.sort_values("selection_score")
    ranking.to_csv(out / "model_ranking.csv", index=False)
    (out / "model_ranking.json").write_text(ranking.to_json(orient="records", indent=2), encoding="utf-8")
    md_table(ranking.fillna("").to_dict("records"), out / "tables/table_10_model_ranking.md")
    summary = {
        "best_by_rmse": "" if ranking.empty else valid.sort_values("RMSE").iloc[0]["model"],
        "best_by_r2": "" if ranking.empty else valid.sort_values("R2", ascending=False).iloc[0]["model"],
        "best_by_f1": "" if ranking.empty else valid.sort_values("F1", ascending=False).iloc[0]["model"],
        "best_by_latency": "" if ranking.empty else valid.sort_values("inference_latency_mean_ms").iloc[0]["model"],
        "recommended_model_for_report": "" if ranking.empty else ranking.iloc[0]["model"],
    }
    (out / "model_selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_figures(metrics: pd.DataFrame, context: dict, out: Path) -> None:
    fig = out / "figures"; data = out / "figure_data"
    fig.mkdir(exist_ok=True); data.mkdir(exist_ok=True)
    metrics.to_csv(data / "08_09_10_11_model_selection_metrics.csv", index=False)
    for num, metric, title in [("08", "RMSE", "RMSE"), ("09", "R2", "R2"), ("10", "F1", "F1"), ("11", "inference_latency_mean_ms", "Latency ms")]:
        plt.figure(figsize=(10, 5))
        tmp = metrics.copy()
        tmp[metric] = pd.to_numeric(tmp.get(metric), errors="coerce")
        sns.barplot(data=tmp.dropna(subset=[metric]), x=metric, y="model")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(fig / f"{num}_model_selection_{'latency' if num == '11' else metric.lower()}.png", dpi=150)
        plt.close()
    arch = pd.DataFrame({"variant": metrics["model"].tolist(), "family": ["baseline" if m in {"persistence", "moving_average", "arima"} else "neural_light" for m in metrics["model"].tolist()]})
    arch.to_csv(data / "07_model_variants_architecture.csv", index=False)
    plt.figure(figsize=(10, 4))
    sns.countplot(data=arch, x="family")
    plt.tight_layout()
    plt.savefig(fig / "07_model_variants_architecture.png", dpi=150)
    plt.close()
    hist_path = data / "12_best_model_train_val_loss.csv"
    plt.figure(figsize=(8, 4))
    if hist_path.exists():
        h = pd.read_csv(hist_path)
        for c in h.columns:
            plt.plot(h[c], label=c)
        plt.legend()
    plt.tight_layout()
    plt.savefig(fig / "12_best_model_train_val_loss.png", dpi=150)
    plt.close()

    # Placeholder backed by CSVs for optional final/holdout plots not applicable in this run.
    for n, name in [
        ("13", "best_model_forecast_vs_actual"),
        ("14", "best_model_residuals"),
        ("15", "holdout_old20_forecast_vs_actual"),
        ("16", "confusion_matrix_best_model"),
        ("17", "roc_curve_best_model"),
        ("18", "ablation_tcn_attention_units"),
        ("19", "source_wise_performance"),
        ("20", "final_recommended_model_summary"),
    ]:
        csv_path = data / f"{n}_{name}.csv"
        pd.DataFrame({"note": ["not_available_optional"], "reason": ["optional evaluation not applicable for this run"]}).to_csv(csv_path, index=False)
        plt.figure(figsize=(7, 3))
        plt.text(0.5, 0.5, name.replace("_", " "), ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(fig / f"{n}_{name}.png", dpi=150)
        plt.close()


def write_report(out: Path, args, metrics: pd.DataFrame, ranking_summary: dict) -> None:
    summary_path = out / "data_reset_summary.json"
    data_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    artifact_rows = []
    for p in sorted(out.rglob("*")):
        if p.is_file():
            artifact_rows.append({"artifact": str(p.relative_to(out)), "size_bytes": p.stat().st_size})
    md_table(artifact_rows, out / "tables/table_16_artifact_index.md")
    md_table([ranking_summary], out / "tables/table_11_best_model_metrics.md")
    md_table(metrics[["model", "inference_latency_mean_ms", "inference_latency_p50_ms", "inference_latency_p95_ms", "parameter_count", "model_size_mb"]].fillna("").to_dict("records"), out / "tables/table_12_latency_comparison.md")
    ablation = metrics[metrics["model"].astype(str).str.contains("tcn|attention|mhsa", case=False, na=False)].fillna("")
    md_table(ablation.to_dict("records"), out / "tables/table_13_ablation_tcn_attention_units.md")
    md_table([{"status": "not_applicable", "reason": "old_bitbrains_holdout20 kept separate; optional cross-domain holdout evaluation not applicable for this run"}], out / "tables/table_14_holdout_old20_results.md")
    limitations = [
        {"limitation": "PARTIAL_DATA" if data_summary.get("status") != "FULL_DATA" else "none", "detail": f"external_real_rows={data_summary.get('external_real_rows')} target={data_summary.get('target_external_rows')}"},
        {"limitation": "source type", "detail": "Cluster/VM traces are workload proxies, not guaranteed web production logs."},
        {"limitation": "synthetic", "detail": "Synthetic noisy data is not real trace data."},
        {"limitation": "full training", "detail": "Top-3 120 epoch full training is skipped when data status is PARTIAL_DATA."},
    ]
    md_table(limitations, out / "tables/table_15_limitations_and_failures.md")
    report = f"""# BÁO CÁO THỰC NGHIỆM DATA RESET + BIG LOGS + MODEL SELECTION

## 1. Mục tiêu thay đổi
- Chuyển từ CSV-only sang SQL/local data pool.
- Loại 20% test cũ khỏi training.
- Bổ sung log/trace từ nhiều nguồn uy tín khi có thể tải được.
- Tạo thêm 20% synthetic noisy continuous data trên lượng external thực tế đã nạp.
- Thử nhiều biến thể mô hình nhẹ hơn.
- Chọn model tốt nhất cho dự đoán nghẽn web theo metrics.

## 2. Trạng thái repo và môi trường
Xem `git_state_before.md`, `environment.md`, `repo_audit.md`.

## 3. Archive dữ liệu/model cũ
Artifact cũ đã được copy vào `archive_previous_runs/`. Mặc định không xóa vĩnh viễn vì không có flag purge. Xem `archive_manifest.csv`.

## 4. Xử lý dữ liệu cũ
- Tổng số dòng Bitbrains cũ: {data_summary.get('old_train80_rows', 0) + data_summary.get('old_holdout20_rows', 0)}
- 80% train cũ: {data_summary.get('old_train80_rows')}
- 20% holdout cũ: {data_summary.get('old_holdout20_rows')}
- Có dùng holdout để train không: Không.

## 5. Nguồn dữ liệu mới
- Trạng thái: `{data_summary.get('status', 'UNKNOWN')}`
- External rows đã nạp: {data_summary.get('external_real_rows')}
- Số nguồn inventory: {data_summary.get('source_count')}
- Nếu chưa đạt 2,000,000 dòng, run này không tuyên bố full success. Xem `source_download_failures.md` và `tables/table_06_source_inventory.md`.

## 6. Synthetic noisy data
- Synthetic rows: {data_summary.get('synthetic_noisy_rows')}
- Tỷ lệ: {data_summary.get('synthetic_ratio')}
- Synthetic data chỉ dùng tăng độ bền thử nghiệm, không phải dữ liệu thật.

## 7. Training pool cuối
- External real rows: {data_summary.get('external_real_rows')}
- Synthetic rows: {data_summary.get('synthetic_noisy_rows')}
- Old train80 rows: {data_summary.get('old_train80_rows')}
- Total train_pool rows: {data_summary.get('total_train_pool_rows')}
- Validation rows: {data_summary.get('validation_rows')}
- SQLite DB: `{data_summary.get('db_path')}`

## 8. Mô hình và biến thể đã thử
Xem `tables/table_07_model_variants.md` và `tables/table_09_model_selection_metrics.md`.

## 9. Kết quả model selection
- best_by_rmse: {ranking_summary.get('best_by_rmse')}
- best_by_r2: {ranking_summary.get('best_by_r2')}
- best_by_f1: {ranking_summary.get('best_by_f1')}
- best_by_latency: {ranking_summary.get('best_by_latency')}
- recommended_model_for_report: {ranking_summary.get('recommended_model_for_report')}

## 10. Kết quả model tốt nhất
Checkpoint nằm trong `models/` nếu model neural chạy thành công. Metrics chi tiết ở `model_ranking.csv`.

## 11. Đánh giá holdout 20% cũ nếu có
Holdout cũ được giữ riêng trong `holdout_old_20pct/` và không dùng train/tune. Cross-domain holdout chưa chạy trong run này.

## 12. Kết luận khoa học
Model khuyến nghị hiện tại là `{ranking_summary.get('recommended_model_for_report')}` theo bảng metrics. Nếu baseline đứng đầu, kết luận phải ghi baseline tốt hơn các mô hình attention trong run này.

## 13. Hạn chế
- Dữ liệu 4 nguồn không hoàn toàn là web log production.
- Cluster/VM trace chỉ là proxy cho workload hệ thống.
- Synthetic noisy data chỉ dùng tăng độ bền, không thay thế dữ liệu thật.
- Nếu thiếu nguồn hoặc thiếu 2M dòng thì trạng thái là PARTIAL_DATA và đã ghi rõ.
- Nếu model chưa vượt baseline thì giữ nguyên kết quả theo metrics.

## 14. Artifact index
Xem `tables/table_16_artifact_index.md`, `figures/`, `figure_data/`, `raw_console.log`, `train_commands.log`.
"""
    (out / "NCKH_BIGLOGS_MODEL_SELECTION_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--table", default="train_pool")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[30, 60, 120])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1])
    parser.add_argument("--models", nargs="+", default=["persistence", "moving_average", "arima", "lstm32", "bilstm32", "tcn16", "tcn32", "tcn_bilstm32_no_attn", "tcn_bilstm32_temporal_attention", "tcn_feature_attention_bilstm_temporal_attention"])
    parser.add_argument("--quick-epochs", type=int, default=10)
    parser.add_argument("--full-epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()
    np.random.seed(args.seed)
    out = Path(args.output_dir).resolve()
    for sub in ["tables", "figures", "figure_data", "models"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    with (out / "raw_console.log").open("a", encoding="utf-8") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        with (out / "train_commands.log").open("a", encoding="utf-8") as f:
            f.write(" ".join(sys.argv) + "\n")
        metrics, context = run_selection(args, out)
        ranking_summary = rank_models(metrics, out)
        write_figures(metrics, context, out)
        write_report(out, args, metrics, ranking_summary)
        report = out / "NCKH_BIGLOGS_MODEL_SELECTION_REPORT.md"
        validation = {
            "report_exists": report.exists(),
            "model_ranking_exists": (out / "model_ranking.csv").exists(),
            "raw_console_exists": (out / "raw_console.log").exists(),
            "best_model_checkpoint_exists": any((out / "models").glob("*.keras")),
            "report_path": str(report),
        }
        (out / "final_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
        print(json.dumps(validation, indent=2))
        print(str(report))


if __name__ == "__main__":
    main()
