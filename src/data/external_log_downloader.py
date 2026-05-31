from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class SourceCandidate:
    source_name: str
    source_url: str
    license_terms: str
    local_sample: str
    notes: str


SOURCES = [
    SourceCandidate(
        "Alibaba Cluster Trace Program",
        "https://github.com/alibaba/clusterdata",
        "Research use terms in repository; some full downloads require survey or scripts.",
        "Data/Alibaba_Cluster_Trace_Sample.csv",
        "Local bundled sample used when full trace is unavailable.",
    ),
    SourceCandidate(
        "Google Borg Cluster Data",
        "https://github.com/google/cluster-data",
        "Google cluster trace public dataset terms; 2019 trace is commonly accessed through BigQuery/Kaggle mirrors.",
        "Data/Google_Cluster_Trace_Sample.csv",
        "Local bundled sample used; full BigQuery export is not attempted by default.",
    ),
    SourceCandidate(
        "Microsoft Azure Public Dataset",
        "https://github.com/Azure/AzurePublicDataset",
        "Microsoft public dataset repository terms.",
        "Data/Azure_Cloud_VM_Trace_Sample.csv",
        "Local bundled VM trace sample used.",
    ),
    SourceCandidate(
        "Bitbrains/GWA workload traces",
        "http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains",
        "GWA/TU Delft dataset terms; access may require site availability.",
        "Data/Bitbrains_FastStorage_Trace_Sample.csv",
        "Only first 80% of the old Bitbrains file is allowed in training; holdout is excluded.",
    ),
    SourceCandidate(
        "AWS CloudWatch Trace Sample",
        "https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/working_with_metrics.html",
        "Local repository sample; not a public production trace release.",
        "Data/AWS_CloudWatch_Trace_Sample.csv",
        "Fallback fourth non-holdout local sample if public sources cannot be fetched.",
    ),
]


def collect_local_sources(repo_root: str | Path, include_old_bitbrains_train: bool = False) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    repo = Path(repo_root)
    inventory = []
    failures = []
    frames = []
    for src in SOURCES:
        path = repo / src.local_sample
        if not path.exists():
            failures.append(
                {
                    "source_name": src.source_name,
                    "source_url": src.source_url,
                    "problem": f"Local sample not found: {src.local_sample}",
                }
            )
            continue
        if "Bitbrains" in src.source_name and not include_old_bitbrains_train:
            continue
        try:
            df = pd.read_csv(path)
            frames.append(df.assign(__source_name=src.source_name, __source_path=str(path)))
            inventory.append(
                {
                    "source_name": src.source_name,
                    "source_url": src.source_url,
                    "license_terms": src.license_terms,
                    "downloaded_file_path": str(path),
                    "raw_rows": len(df),
                    "selected_rows": len(df),
                    "columns": ", ".join(map(str, df.columns)),
                    "timestamp_column": "timestamp" if "timestamp" in {c.lower() for c in df.columns} else "",
                    "machine_service_id_column": "not present in local sample",
                    "metrics_available": ", ".join([c for c in df.columns if c.lower() != "timestamp"]),
                    "problems_limitations": src.notes,
                }
            )
        except Exception as exc:
            failures.append({"source_name": src.source_name, "source_url": src.source_url, "problem": str(exc)})
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, inventory, failures
