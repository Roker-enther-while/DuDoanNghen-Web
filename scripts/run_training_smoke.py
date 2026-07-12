"""Run smoke training for baseline and first deep comparison models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.registry import get_model_metadata
from src.training.trainer import load_training_config, train_and_evaluate, write_dashboard_payload, write_model_comparison


SMOKE_MODELS = ["naive_last_value", "moving_average", "lstm", "gru", "tcn"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "processed" / "nasa_http" / "windows" / "windows_fp16.npz"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "training" / "smoke.yaml"))
    parser.add_argument("--models", nargs="+", default=SMOKE_MODELS)
    args = parser.parse_args(argv)

    config = load_training_config(args.config)
    models = config.get("models", args.models)
    results = []
    for model_name in models:
        print(f"Running smoke model: {model_name}")
        try:
            results.append(train_and_evaluate(model_name, args.data, config))
        except Exception as exc:
            try:
                category = get_model_metadata(model_name).category
            except Exception:
                category = "unknown"
            results.append(
                {
                    "model": model_name,
                    "category": category,
                    "status": "failed",
                    "error": str(exc),
                    "train_time_seconds": 0.0,
                    "inference_time_seconds": 0.0,
                    "metrics": {"mae": 0.0, "rmse": 0.0, "r2": 0.0},
                    "alert_metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "tp": 0, "fp": 0, "tn": 0, "fn": 0},
                    "prediction_path": None,
                    "model_path": None,
                }
            )
            print(f"FAILED {model_name}: {exc}")
    output_tag = config.get("output_tag")
    comparison_json, comparison_md = write_model_comparison(results, config.get("output_dir", "outputs"), output_tag=output_tag)
    dashboard_json, dashboard_csv = write_dashboard_payload(results, args.data, config.get("output_dir", "outputs"), output_tag=output_tag)
    print(json.dumps({"comparison_json": comparison_json, "comparison_md": comparison_md, "dashboard_json": dashboard_json, "dashboard_csv": dashboard_csv}, indent=2))
    return 0 if all(r["status"] == "success" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
