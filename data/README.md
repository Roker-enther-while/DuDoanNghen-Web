# Data Directory

## Cấu trúc

```
data/
├── raw/                    # Raw data (không commit)
│   └── nasa_http/          # NASA HTTP 1995 logs
│       ├── NASA_access_log_Jul95.gz
│       └── NASA_access_log_Aug95.gz
├── processed/              # Processed data
│   ├── nasa_http_3m/       # NASA 3-month processed
│   │   └── windows/
│   │       └── windows_fp16.npz
│   └── synthetic_stress/   # Synthetic stress benchmark
│       └── windows/
│           └── windows_fp16.npz
└── README.md               # This file
```

## Data sources

### NASA HTTP 1995 (real public)
- **Source:** NASA Kennedy Space Center HTTP Server Logs
- **URL:** https://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html
- **License:** Internet Traffic Archive redistributable trace permission
- **Files:** NASA_access_log_Jul95.gz, NASA_access_log_Aug95.gz
- **Size:** ~3.46M raw log lines

### Zanbil (planned)
- **Source:** Online Shopping Store Web Server Logs
- **URL:** https://doi.org/10.7910/DVN/3QBYB5
- **License:** CC0-1.0 Public Domain
- **Status:** RAW MISSING - place at data/raw/zanbil/access.log

### Synthetic Stress (generated)
- **Scenarios:** flash_crowd, burst_traffic, error_surge, slow_ramp, periodic_spike, mixed_incident
- **Samples:** 1800 (6 scenarios × 300 samples)
- **Status:** Generated from public baseline

## Cách đặt data

1. Tải NASA HTTP logs từ Internet Traffic Archive
2. Đặt vào `data/raw/nasa_http/`
3. Chạy: `python scripts/prepare_nasa_http.py`
4. Chạy: `python scripts/run_data_pipeline.py --config configs/data/nasa_http_3m.yaml`
