# Zanbil Raw Data Input

## Trạng thái hiện tại

**Status:** BLOCKED — Chưa có raw file

## Cách bổ sung

### 1. Tải Zanbil dataset

Từ Harvard Dataverse:
```
https://doi.org/10.7910/DVN/3QBYB5
```

Hoặc từ Kaggle:
```
kaggle datasets download -d eliasdabbas/web-server-access-logs
```

### 2. Đặt file vào đúng vị trí

```
data/raw/zanbil/access.log
```

### 3. Chạy pipeline sau khi có file

```bash
# Kiểm tra readiness
python scripts/check_zanbil_readiness.py

# Chuẩn hóa logs
python scripts/prepare_zanbil_logs.py --input data/raw/zanbil/access.log --config configs/data/zanbil_logs.yaml

# Build multi-source dataset
python scripts/build_multi_source_dataset.py --config configs/data/multi_source_web_logs.yaml

# Train lại
python scripts/train_model.py --config configs/training/tcn_attention_bilstm_full_120.yaml
```

## License

Zanbil dataset: CC0-1.0 Public Domain Dedication

## Citation

Zaker, Farzin, 2019, Online Shopping Store - Web Server Logs, Harvard Dataverse, V1, https://doi.org/10.7910/DVN/3QBYB5
