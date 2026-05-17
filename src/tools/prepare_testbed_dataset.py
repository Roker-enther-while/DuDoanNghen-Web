from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.schema_harmonizer import harmonize_dataframe
from src.data.sql_data_pool import replace_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Harmonize labeled testbed CSV and optionally write it to the SQLite pool.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="Data/testbed/testbed_harmonized.csv")
    parser.add_argument("--db-path", default="")
    parser.add_argument("--table", default="testbed_pool")
    parser.add_argument("--source-name", default="docker_prometheus_testbed")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    mapping: dict = {}
    out = harmonize_dataframe(df, args.source_name, mapping_out=mapping)
    if "congestion_label" in df.columns:
        out["congestion_label"] = pd.to_numeric(df["congestion_label"], errors="coerce").fillna(out["congestion_label"]).astype(int)
    for extra in ["label_reason", "label_rule_version", "load_profile"]:
        if extra in df.columns:
            out[extra] = df[extra]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    summary = {"output": str(output), "rows": len(out), "columns": list(out.columns), "mapping": mapping}
    if args.db_path:
        replace_table(args.db_path, args.table, out)
        summary["db_path"] = args.db_path
        summary["table"] = args.table
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

