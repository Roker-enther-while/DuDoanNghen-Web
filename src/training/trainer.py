"""Train, evaluate, and persist model comparison artifacts."""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.training.callbacks import make_keras_callbacks
from src.training.data_loader import (
    convert_to_train_dtype,
    get_data_summary,
    load_window_data,
    subset_split,
    subset_window_data,
    validate_window_data,
)
from src.training.metrics import alert_metrics, regression_metrics
from src.training.metrics import resolve_alert_threshold
from src.training.gpu_memory import configure_tensorflow_gpu
from src.training.registry import get_model_builder, get_model_metadata
from src.training.torch_trainer import train_torch_model


def load_training_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def set_seed(seed: int, backend: str = "tensorflow") -> None:
    random.seed(seed)
    np.random.seed(seed)
    if backend == "torch":
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        return
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def _history_dict(history) -> dict:
    if history is None:
        return {}
    return {key: [float(x) for x in values] for key, values in history.history.items()}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _write_predictions(path: Path, y_true, y_pred, timestamps=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"y_true": np.asarray(y_true).reshape(-1), "y_pred": np.asarray(y_pred).reshape(-1)})
    if timestamps is not None and len(timestamps) == len(frame):
        frame.insert(0, "timestamp", timestamps)
    frame.to_csv(path, index=False)


def train_and_evaluate(
    model_name: str,
    data_path: str | Path,
    config: dict | None = None,
    overrides: dict | None = None,
) -> dict[str, Any]:
    """Run one model, save predictions/metrics/model, and return a manifest row."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    config = dict(config or {})
    overrides = {key: value for key, value in (overrides or {}).items() if value is not None}
    config.update(overrides)

    backend = config.get("backend", "tensorflow")
    seed = int(config.get("seed", 42))
    set_seed(seed, backend=backend)
    output_dir = Path(config.get("output_dir", "outputs"))
    artifact_name = config.get("artifact_name", model_name)
    gpu_plan_payload = {"backend": backend, "note": "baseline/no deep backend initialized"}
    if backend != "torch":
        gpu_plan_payload = configure_tensorflow_gpu(config).to_dict()
    metadata = get_model_metadata(model_name)
    builder = get_model_builder(model_name)

    raw_data = load_window_data(data_path)
    validate_window_data(raw_data)
    summary = get_data_summary(raw_data)
    train_data = convert_to_train_dtype(raw_data, "float32")
    train_subset = subset_split(train_data, "train", config.get("max_train_samples"), "head")
    val_subset = subset_split(train_data, "val", config.get("max_val_samples"), "head")
    test_strategy = config.get("evaluation_sample_strategy", "head")
    test_subset = subset_split(train_data, "test", config.get("max_test_samples"), test_strategy)

    X_train, y_train = train_subset["X"], train_subset["y"]
    X_val, y_val = val_subset["X"], val_subset["y"]
    X_test, y_test = test_subset["X"], test_subset["y"]
    feature_columns = raw_data.get("feature_columns", [])

    threshold_cfg = config.get("alert_threshold", 0.70)
    if isinstance(threshold_cfg, dict):
        threshold_mode = threshold_cfg.get("mode", "fixed")
        threshold_value = float(threshold_cfg.get("value", 0.70))
        reference_split = threshold_cfg.get("reference_split", "val")
    else:
        threshold_mode = "fixed"
        threshold_value = float(threshold_cfg)
        reference_split = "fixed"
    reference_map = {"train": y_train, "val": y_val, "test": y_test, "fixed": y_val}
    alert_threshold = resolve_alert_threshold(reference_map.get(reference_split, y_val), threshold_mode, threshold_value)

    started = time.perf_counter()
    history = {}
    model_path: str | None = None
    status = "success"
    error = None
    inference_time = 0.0
    torch_train_time = None

    try:
        if metadata.category == "baseline":
            model = builder(config)
            model.fit(X_train, y_train, feature_columns=feature_columns)
            inference_started = time.perf_counter()
            y_pred = model.predict(X_test)
            inference_time = time.perf_counter() - inference_started
            model_path = model.save(output_dir / "models" / artifact_name)
        elif backend == "torch":
            model_path, history, y_pred, torch_train_time, inference_time, gpu_plan_payload = train_torch_model(
                model_name, X_train, y_train, X_val, y_val, X_test, config, output_dir
            )
        else:
            model = builder(input_shape=X_train.shape[1:], config=config)
            print(f"Model summary for {model_name}:")
            model.summary()
            fit_kwargs = {
                "x": X_train,
                "y": y_train,
                "epochs": int(config.get("epochs", 2)),
                "batch_size": int(config.get("batch_size", 64)),
                "verbose": int(config.get("verbose", 1)),
                "callbacks": make_keras_callbacks(config, model_name=model_name, output_dir=output_dir),
            }
            if len(X_val):
                fit_kwargs["validation_data"] = (X_val, y_val)
            keras_history = model.fit(**fit_kwargs)
            history = _history_dict(keras_history)
            inference_started = time.perf_counter()
            y_pred = model.predict(X_test, batch_size=int(config.get("batch_size", 64)), verbose=0).reshape(-1)
            inference_time = time.perf_counter() - inference_started
            model_dir = output_dir / "models" / artifact_name
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = str(model_dir / "model.keras")
            model.save(model_path)
    except NotImplementedError:
        raise
    except Exception as exc:
        status = "failed"
        error = str(exc)
        y_pred = np.zeros_like(y_test, dtype=np.float32)

    elapsed = torch_train_time if torch_train_time is not None else time.perf_counter() - started
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float32).reshape(-1), 0.0, 1.0)
    y_test_eval = np.asarray(y_test, dtype=np.float32).reshape(-1)[: len(y_pred)]
    reg = regression_metrics(y_test_eval, y_pred)
    alert = alert_metrics(y_test_eval, y_pred, threshold=alert_threshold)

    if config.get("structured_output", False):
        prediction_path = output_dir / "predictions" / artifact_name / "test_predictions.csv"
        history_path = output_dir / "metrics" / artifact_name / "history.json"
        metrics_path = output_dir / "metrics" / artifact_name / "final_metrics.json"
    else:
        prediction_path = output_dir / "predictions" / f"{artifact_name}_test_predictions.csv"
        history_path = output_dir / "metrics" / f"{artifact_name}_history.json"
        metrics_path = output_dir / "metrics" / f"{artifact_name}_metrics.json"
    _write_predictions(prediction_path, y_test_eval, y_pred, test_subset.get("timestamps"))
    _write_json(history_path, {"model": model_name, "history": history})
    payload = {
        "model": model_name,
        "category": metadata.category,
        "status": status,
        "error": error,
        "train_time_seconds": float(elapsed),
        "inference_time_seconds": float(inference_time),
        "metrics": reg,
        "alert_metrics": alert,
        "alert_threshold": alert_threshold,
        "alert_threshold_mode": threshold_mode,
        "alert_threshold_value": threshold_value,
        "alert_threshold_reference_split": reference_split,
        "evaluation_sample_strategy": test_strategy,
        "evaluation_sample_count": int(len(y_test_eval)),
        "prediction_path": str(prediction_path),
        "model_path": model_path,
        "history_path": str(history_path),
        "data_summary": summary,
        "config": config,
        "gpu_memory_plan": gpu_plan_payload,
        "target_notice": "NASA target is a proxy congestion score, not a measured congestion label.",
    }
    _write_json(metrics_path, payload)
    return payload | {"metrics_path": str(metrics_path)}


def comparison_markdown(results: list[dict[str, Any]]) -> str:
    successful = [r for r in results if r.get("status") == "success"]
    best_rmse = min(successful, key=lambda r: r["metrics"]["rmse"]) if successful else None
    best_f1 = max(successful, key=lambda r: r["alert_metrics"]["f1"]) if successful else None
    deep_models = [r for r in successful if r.get("category") not in {"baseline"}]
    baseline_models = [r for r in successful if r.get("category") == "baseline"]
    best_deep = min(deep_models, key=lambda r: r["metrics"]["rmse"]) if deep_models else None
    best_baseline = min(baseline_models, key=lambda r: r["metrics"]["rmse"]) if baseline_models else None
    data_summary = next((r.get("data_summary") for r in results if r.get("data_summary")), None)
    small_warning = ""
    if data_summary and data_summary.get("train", {}).get("X_shape", [0])[0] <= 1000:
        small_warning = "This run used a smoke/small artifact; do not draw scientific conclusions from the ranking."
    lines = [
        "# Model Comparison",
        "",
        "Target is a proxy congestion score for NASA HTTP logs, not a measured congestion label.",
    ]
    if small_warning:
        lines.extend(["", f"Warning: {small_warning}"])
    lines.extend([
        "",
        "| model | category | status | threshold | test_strategy | true_pos | pred_pos | MAE | RMSE | R2 | alert_f1 | warning |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for r in results:
        alert_info = r.get("alert_metrics", {})
        lines.append(
            f"| {r['model']} | {r['category']} | {r['status']} | "
            f"{r.get('alert_threshold_mode', 'fixed')}={r.get('alert_threshold', 0.0):.6f} | "
            f"{r.get('evaluation_sample_strategy', '')} | "
            f"{alert_info.get('alert_positive_count_true', 0)} | {alert_info.get('alert_positive_count_pred', 0)} | "
            f"{r['metrics']['mae']:.6f} | {r['metrics']['rmse']:.6f} | {r['metrics']['r2']:.6f} | "
            f"{alert_info.get('f1', 0.0):.6f} | {alert_info.get('warning') or ''} |"
        )
    lines.extend(["", "## Best Models"])
    lines.append(f"- Best by RMSE: {best_rmse['model']} ({best_rmse['metrics']['rmse']:.6f})" if best_rmse else "- Best by RMSE: n/a")
    lines.append(f"- Best by alert F1: {best_f1['model']} ({best_f1['alert_metrics']['f1']:.6f})" if best_f1 else "- Best by alert F1: n/a")
    lines.append(f"- Best deep model by RMSE: {best_deep['model']} ({best_deep['metrics']['rmse']:.6f})" if best_deep else "- Best deep model by RMSE: n/a")
    lines.append(f"- Best baseline by RMSE: {best_baseline['model']} ({best_baseline['metrics']['rmse']:.6f})" if best_baseline else "- Best baseline by RMSE: n/a")
    proposed = next((r for r in successful if r["model"] == "tcn_attention_bilstm"), None)
    lines.extend(["", "## Automatic Checks"])
    for other in ["tcn_lstm", "transformer", "tcn"]:
        target = next((r for r in successful if r["model"] == other), None)
        if proposed and target:
            verdict = "yes" if proposed["metrics"]["rmse"] < target["metrics"]["rmse"] else "no"
            lines.append(f"- Proposed beats {other} by RMSE: {verdict}")
    if best_baseline and proposed:
        verdict = "yes" if proposed["metrics"]["rmse"] < best_baseline["metrics"]["rmse"] else "no"
        lines.append(f"- Proposed beats best baseline by RMSE: {verdict}")
    lines.extend(
        [
            "",
            "## Notes",
            "- Quick/smoke training is intentionally small and not tuned.",
            "- Baselines are simple controls and should not be treated as optimized forecasting models.",
            "- If deep models underperform naive baselines, likely causes include limited data, short training, proxy target dynamics, and untuned hyperparameters.",
        ]
    )
    proxy_quality_path = Path("outputs") / "metrics" / "proxy_target_quality.json"
    if proxy_quality_path.exists():
        try:
            proxy_quality = json.loads(proxy_quality_path.read_text(encoding="utf-8"))
            for note in proxy_quality.get("notes", []):
                lines.append(f"- Proxy target note: {note}")
        except Exception:
            pass
    return "\n".join(lines) + "\n"


def write_model_comparison(results: list[dict[str, Any]], output_dir: str | Path = "outputs", output_tag: str | None = None) -> tuple[str, str]:
    output_dir = Path(output_dir)
    prefix = f"{output_tag}_" if output_tag else ""
    json_path = output_dir / "metrics" / f"{prefix}model_comparison.json"
    md_path = output_dir / "reports" / f"{prefix}model_comparison.md"
    _write_json(json_path, {"models": results})
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(comparison_markdown(results), encoding="utf-8")
    return str(json_path), str(md_path)


def write_dashboard_payload(
    results: list[dict[str, Any]],
    data_artifact_path: str | Path,
    output_dir: str | Path = "outputs",
    output_tag: str | None = None,
) -> tuple[str, str]:
    """Write dashboard-ready JSON/CSV without building a UI."""
    output_dir = Path(output_dir)
    successful = [r for r in results if r.get("status") == "success"]
    best = min(successful, key=lambda r: r["metrics"]["rmse"]) if successful else None
    best_f1 = max(successful, key=lambda r: r["alert_metrics"]["f1"]) if successful else None
    proposed = next((r for r in successful if r["model"] == "tcn_attention_bilstm"), None)
    data_summary = next((r.get("data_summary") for r in results if r.get("data_summary")), {})
    rows = []
    for r in results:
        rows.append(
            {
                "model": r["model"],
                "category": r.get("category"),
                "status": r.get("status"),
                "train_time_seconds": r.get("train_time_seconds", 0.0),
                "inference_time_seconds": r.get("inference_time_seconds", 0.0),
                "mae": r.get("metrics", {}).get("mae"),
                "rmse": r.get("metrics", {}).get("rmse"),
                "r2": r.get("metrics", {}).get("r2"),
                "alert_f1": r.get("alert_metrics", {}).get("f1"),
                "alert_threshold": r.get("alert_threshold"),
                "alert_threshold_mode": r.get("alert_threshold_mode"),
                "alert_positive_count_true": r.get("alert_metrics", {}).get("alert_positive_count_true"),
                "alert_positive_count_pred": r.get("alert_metrics", {}).get("alert_positive_count_pred"),
                "evaluation_sample_strategy": r.get("evaluation_sample_strategy"),
                "prediction_path": r.get("prediction_path"),
                "history_path": r.get("history_path"),
                "model_path": r.get("model_path"),
            }
        )
    target_distribution_summary = None
    target_distribution_path = output_dir / "metrics" / "target_distribution.json"
    if target_distribution_path.exists():
        try:
            target_distribution_summary = json.loads(target_distribution_path.read_text(encoding="utf-8"))
        except Exception:
            target_distribution_summary = None
    first_result = results[0] if results else {}
    payload = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_type": output_tag or "default",
        "data_artifact_path": str(data_artifact_path),
        "data_summary": data_summary,
        "model_comparison": rows,
        "best_model": best["model"] if best else None,
        "best_by_rmse": best["model"] if best else None,
        "best_by_f1": best_f1["model"] if best_f1 else None,
        "proposed_model_result": proposed,
        "threshold_info": {
            "mode": first_result.get("alert_threshold_mode"),
            "value": first_result.get("alert_threshold_value"),
            "resolved_threshold": first_result.get("alert_threshold"),
            "reference_split": first_result.get("alert_threshold_reference_split"),
        },
        "target_distribution_summary": target_distribution_summary,
        "metrics_by_model": {r["model"]: {"metrics": r.get("metrics", {}), "alert_metrics": r.get("alert_metrics", {})} for r in results},
        "prediction_paths": {r["model"]: r.get("prediction_path") for r in results},
        "history_paths": {r["model"]: r.get("history_path") for r in results},
        "warning_notes": [
            "NASA target is a proxy congestion score, not a measured congestion label.",
            "Quick/smoke training is not a scientific final comparison.",
        ],
    }
    web_dir = output_dir / "web"
    web_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{output_tag}_" if output_tag else ""
    json_path = web_dir / f"{prefix}model_dashboard_payload.json"
    csv_path = web_dir / f"{prefix}model_comparison_table.csv"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return str(json_path), str(csv_path)
