"""Evaluate a predictions CSV with regression and alert metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.metrics import alert_metrics, regression_metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.predictions)
    result = {
        "predictions": args.predictions,
        "metrics": regression_metrics(frame["y_true"], frame["y_pred"]),
        "alert_metrics": alert_metrics(frame["y_true"], frame["y_pred"], args.threshold),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
