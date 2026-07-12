"""Prepare NASA HTTP logs into time-series and normalized window artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.nasa_http import add_congestion_proxy_target, aggregate_requests, iter_nasa_log_records
from src.data.preprocess import chronological_split, normalize_splits, save_scaler
from src.data.quality import generate_quality_report, write_quality_report
from src.data.schemas import NASA_FEATURE_COLUMNS, NASA_TARGET_COLUMN
from src.data.windowing import build_windows_from_normalized


def prepare_nasa_http(
    input_paths: list[Path],
    output_root: Path,
    window_minutes: int = 1,
    limit_lines: int | None = None,
    lookback_steps: int = 60,
    horizon_steps: int = 15,
) -> dict:
    frames = []
    total_stats = {"total_lines": 0, "parsed_lines": 0, "skipped_lines": 0}
    for path in input_paths:
        frame, stats = iter_nasa_log_records(path, limit_lines=limit_lines)
        frames.append(frame)
        for key, value in stats.to_dict().items():
            total_stats[key] += value
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    timeseries = aggregate_requests(raw, window_minutes=window_minutes)
    timeseries, target_metadata = add_congestion_proxy_target(timeseries, horizon_steps=horizon_steps)

    output_root.mkdir(parents=True, exist_ok=True)
    ts_csv = output_root / f"timeseries_{window_minutes}min.csv"
    timeseries.to_csv(ts_csv, index=False)
    ts_parquet = output_root / f"timeseries_{window_minutes}min.parquet"
    parquet_written = False
    try:
        timeseries.to_parquet(ts_parquet, index=False)
        parquet_written = True
    except Exception as exc:
        print(f"WARNING: parquet output skipped: {exc}")

    report = generate_quality_report(timeseries, total_stats, NASA_FEATURE_COLUMNS)
    write_quality_report(
        report,
        ROOT / "outputs" / "metrics" / "data_quality_report.json",
        ROOT / "outputs" / "reports" / "data_quality_report.md",
    )

    train, val, test, split_metadata = chronological_split(timeseries)
    split_dir = output_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(split_dir / "train_timeseries.csv", index=False)
    val.to_csv(split_dir / "val_timeseries.csv", index=False)
    test.to_csv(split_dir / "test_timeseries.csv", index=False)

    scaler, normalized_outputs = normalize_splits(
        train,
        val,
        test,
        NASA_FEATURE_COLUMNS,
        NASA_TARGET_COLUMN,
        output_root / "normalized",
    )
    save_scaler(output_root / "scaler.json", scaler)
    window_info = build_windows_from_normalized(
        {split: paths["fp16"] for split, paths in normalized_outputs.items()},
        output_root / "windows" / "windows_fp16.npz",
        lookback_steps,
        {"lookback_steps": lookback_steps, "horizon_steps": horizon_steps, "allow_context_overlap": False},
    )
    metadata = {
        "parse_stats": total_stats,
        "timeseries_csv": str(ts_csv),
        "timeseries_parquet": str(ts_parquet) if parquet_written else None,
        "quality_report": str(ROOT / "outputs" / "metrics" / "data_quality_report.json"),
        "split": split_metadata,
        "target": target_metadata,
        "normalized_outputs": normalized_outputs,
        "scaler": str(output_root / "scaler.json"),
        "windows": window_info,
        "feature_columns": NASA_FEATURE_COLUMNS,
        "target_column": NASA_TARGET_COLUMN,
    }
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", default=[str(ROOT / "data" / "raw" / "nasa_http" / "NASA_access_log_Jul95.gz")])
    parser.add_argument("--output-root", default=str(ROOT / "data" / "processed" / "nasa_http"))
    parser.add_argument("--window-minutes", type=int, default=1)
    parser.add_argument("--limit-lines", type=int, default=None)
    parser.add_argument("--lookback-steps", type=int, default=60)
    parser.add_argument("--horizon-steps", type=int, default=15)
    args = parser.parse_args(argv)
    metadata = prepare_nasa_http(
        [Path(p) for p in args.input],
        Path(args.output_root),
        args.window_minutes,
        args.limit_lines,
        args.lookback_steps,
        args.horizon_steps,
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
