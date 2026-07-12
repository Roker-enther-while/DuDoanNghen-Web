# Azure VM Trace (v2) — Provenance

- **Source**: Zenodo record 14564935 ("DataCenter-Traces-Datasets")
- **URL**: https://zenodo.org/records/14564935
- **Downloaded**: 2026-07-12
- **File**: vm_cpu_readings_month_aggregated_cpu_mem.csv (307 KB)
- **MD5**: b14d5002a7d2f7e0033a0deae63d2ced
- **License**: CC-BY 4.0
- **Original dataset**: Microsoft Azure Public Dataset v2 (https://github.com/Azure/AzurePublicDataset)
- **Processing**: Derived/processed by third party. Sum values grouped every 300 seconds for one month. CPU usage computed using core_count of each VM. Each column includes total consumption of all data center VMs.
- **Schema**: timestamp (seconds from epoch, 300s steps), cpu_usage (total), assigned_mem (total)
- **Note**: The timestamp column is in seconds from start (0, 300, 600, ...), not absolute timestamps. Original dataset: https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md
