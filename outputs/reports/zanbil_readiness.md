# Zanbil Readiness

- Raw path: C:\Users\dhp01\OneDrive\Máy tính\TCN-Attention-BiLSTM\data\raw\zanbil\access.log
- Raw exists: False
- File size bytes: 0
- Parser ready: False
- Parsed sample count: 0
- Source governance ready: True
- PII policy ready: True
- Ready for prepare: False

## Next Steps
1. Download the dataset from the source declared in configs/data/public_sources.yaml.
2. Place the authorized raw log at data/raw/zanbil/access.log.
3. Do not commit the raw log unless project policy explicitly allows it.
4. Run: python scripts/prepare_zanbil_logs.py --input data/raw/zanbil/access.log --config configs/data/zanbil_logs.yaml
