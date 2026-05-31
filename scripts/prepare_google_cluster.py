"""Prepare a local Google Cluster sample or BigQuery export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.google_cluster import aggregate_google_cluster_sample, google_cluster_guidance, read_google_cluster_sample


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--window-minutes", type=int, default=5)
    parser.add_argument("--output", default=str(ROOT / "data" / "processed" / "google_cluster" / "timeseries_5min.csv"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.input:
        print(google_cluster_guidance())
        return 0
    if args.dry_run:
        print(json.dumps({"input": args.input, "window_minutes": args.window_minutes, "guidance": google_cluster_guidance()}, indent=2))
        return 0

    df = read_google_cluster_sample(args.input)
    prepared = aggregate_google_cluster_sample(df, args.window_minutes)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output, index=False)
    print(json.dumps({"output": str(output), "rows": len(prepared)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
