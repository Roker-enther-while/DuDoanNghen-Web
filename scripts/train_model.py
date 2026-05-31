"""CLI to train or evaluate one registered model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.data_loader import get_data_summary, load_window_data, validate_window_data
from src.training.registry import list_models
from src.training.trainer import load_training_config, train_and_evaluate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "processed" / "nasa_http" / "windows" / "windows_fp16.npz"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "training" / "smoke.yaml"))
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)

    if args.list_models:
        print(json.dumps(list_models(), indent=2))
        return 0

    data = load_window_data(args.data)
    validate_window_data(data)
    print("Data summary:")
    print(json.dumps(get_data_summary(data), indent=2))

    config = load_training_config(args.config)
    overrides = {
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "max_test_samples": args.max_test_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "output_dir": args.output_dir,
    }
    result = train_and_evaluate(args.model, args.data, config, overrides)
    print(json.dumps({"status": result["status"], "metrics_path": result["metrics_path"], "prediction_path": result["prediction_path"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
