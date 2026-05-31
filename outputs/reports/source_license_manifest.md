# Source License Manifest

Only sources with explicit license and citation metadata are allowed into the pipeline.

## Valid Sources

| source_id | type | license | research | derived_features | attribution | PII risk | raw default |
|---|---|---|---:|---:|---:|---|---:|
| nasa_http_1995 | real_web_log | Internet Traffic Archive redistributable trace permission | True | True | True | medium | True |
| zanbil_web_logs | real_web_log | CC0-1.0 Public Domain Dedication | True | True | False | medium | False |
| synthetic_stress_public_baseline | synthetic_generated | Project-generated derived benchmark | True | True | True | low | False |

## Rejected Or Disabled Sources

- `google_cluster_2011`: valid_disabled - Disabled by default; use local official sample/export only.
- `google_cluster_2019`: valid_disabled - Future source for telemetry-like resource pressure; use BigQuery/sample exports.
