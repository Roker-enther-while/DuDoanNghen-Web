| limitation | detail |
|---|---|
| PARTIAL_DATA | external_real_rows=None target=None |
| source type | Cluster/VM traces are workload proxies, not guaranteed web production logs. |
| synthetic | Synthetic noisy data is not real trace data. |
| full training | Top-3 120 epoch full training is skipped when data status is PARTIAL_DATA. |
