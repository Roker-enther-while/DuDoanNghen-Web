"""Analyze target distribution and proxy target quality."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.data_loader import load_window_data, validate_window_data
from src.training.target_diagnostics import proxy_target_quality, target_distribution_report, write_target_reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args(argv)
    data = load_window_data(args.data)
    validate_window_data(data)
    distribution = target_distribution_report(data)
    quality = proxy_target_quality(data["y_train"], data["y_val"], data["y_test"])
    paths = write_target_reports(distribution, quality, args.output_dir)
    print(json.dumps(paths, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
