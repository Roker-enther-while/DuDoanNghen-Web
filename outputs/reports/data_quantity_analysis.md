# Phân Tích Vấn Đề Số Liệu Training

## 1. Tình Trạng Hiện Tại

### Dữ liệu hiện có

| Metric | Giá trị | Đánh giá quốc tế |
|---|---|---|
| Raw log lines | 3,461,613 | Trung bình |
| Time steps (1-min) | 89,265 | Thấp cho deep learning |
| Training windows | 62,425 | Thấp cho model phức tạp |
| Nguồn dữ liệu | 1 (NASA 1995) | Thiếu đa dạng |
| Thời gian dữ liệu | 2 tháng (Jul-Aug 1995) | Quá ngắn |
| Telemetry | Chỉ traffic features | Thiếu CPU/RAM/response time |

### Kết quả hiện tại

| Metric | Giá trị | Vấn đề |
|---|---|---|
| R² | 0.331 | Model chỉ giải thích 33% phương sai |
| MAE | 0.043 | Chấp nhận được |
| RMSE | 0.056 | Chấp nhận được |
| Recall (p90) | 0.007 | Gần như không phát hiện nghẽn |
| F1 (p90) | 0.015 | Vô nghĩa |
| Positive cases | 1,765/13,330 (13.2%) | Imbalanced |

### Tại sao kết quả chưa đủ uy tín quốc tế

1. **Dữ liệu quá ít** — 62K training windows, trong khi các paper quốc tế thường dùng hàng trăm ngàn đến hàng triệu samples
2. **1 nguồn duy nhất** — không có cross-domain validation
3. **Dữ liệu cũ** — 1995, web patterns khác hoàn toàn hiện tại (HTTP/1.0, no HTTPS, no CDN, no SPA)
4. **Proxy target** — không phải measured congestion thật
5. **Thiếu baseline comparison** — chưa có ARIMA, Prophet, hay các SOTA models
6. **R² thấp** — 0.331 không đủ để publish ở conference/journal uy tín

## 2. Các Nguồn Dữ Liệu Có Thể Bổ Sung

### Ưu tiên 1: Zanbil (đã có framework)

| Thông tin | Chi tiết |
|---|---|
| Tên | Online Shopping Store Web Server Logs |
| Nguồn | Harvard Dataverse |
| License | CC0-1.0 (Public Domain) |
| Kích thước | ~3GB+ |
| URL | https://doi.org/10.7910/DVN/3QBYB5 |
| Kaggle mirror | https://www.kaggle.com/datasets/eliasdabbas/web-server-access-logs |
| Trạng thái | Đã khai báo, raw chưa có |
| Hành động | Tải về → đặt tại data/raw/zanbil/access.log |

**Ưu điểm:**
- License mở (CC0)
- Đã có parser framework trong repo
- Tăng gấp đôi dữ liệu training
- Web e-commerce patterns (hiện đại hơn NASA)

### Ưu tiên 2: Public Web Log Datasets khác

| Dataset | Nguồn | Kích thước | License | Ghi chú |
|---|---|---|---|---|
| Calgary HTTP | ita.ee.lbl.gov | ~200MB | Research | Tương tự NASA |
| ClarkNet HTTP | ita.ee.lbl.gov | ~200MB | Research | ISP web server |
| WorldCup98 | ita.ee.lbl.gov | ~2GB | Research | FIFA World Cup traffic |
| Microsoft IIS Logs | Various | Varies | Research | Windows web server |
| WebTraffSim | Synthetic | Large | Open | Synthetic nhưng realistic |

### Ưu tiên 3: Google Cluster Trace (đã khai báo, disabled)

| Thông tin | Chi tiết |
|---|---|
| Tên | Google ClusterData2011/2019 Borg Trace |
| License | CC-BY-4.0 |
| Có CPU/RAM | ✅ Có system telemetry |
| Vấn đề | Resource trace, không phải web access log |
| Hành động | Enable làm supplementary telemetry source |

## 3. Giải Pháp Tăng Dữ Liệu

### A. Thêm Zanbil Dataset (Ưu tiên cao nhất)

**Tác động:** Tăng training data lên ~2-3x

**Bước thực hiện:**
1. Tải Zanbil dataset từ Kaggle: `kaggle datasets download -d eliasdabbas/web-server-access-logs`
2. Giải nén → đặt `access.log` tại `data/raw/zanbil/access.log`
3. Chạy: `python scripts/prepare_zanbil_logs.py --input data/raw/zanbil/access.log --config configs/data/zanbil_logs.yaml`
4. Chạy: `python scripts/build_multi_source_dataset.py --config configs/data/multi_source_web_logs.yaml`
5. Train lại với NASA+Zanbil

### B. Thêm Calgary/ClarkNet/WorldCup (Ưu tiên trung bình)

**Tác động:** Tăng training data lên ~5-10x

**Bước thực hiện:**
1. Tải từ Internet Traffic Archive
2. Thêm vào `configs/data/public_sources.yaml`
3. Viết parser tương ứng
4. Chuẩn hóa features giống NASA
5. Build multi-source dataset

### C. Data Augmentation (Ưu tiên trung bình)

**Kỹ thuật:**
- **Time warping** — thay đổi tốc độ thời gian
- **Window shifting** — dịch cửa sổ thời gian
- **Noise injection** — thêm Gaussian noise nhỏ
- **SMOTE cho time series** — oversampling minority class
- **Synthetic minority oversampling** — tạo thêm positive cases

**Tác động:** Tăng effective training data lên 2-5x mà không cần thêm raw data

### D. Sử dụng Google Cluster cho Telemetry (Ưu tiên thấp)

**Tác động:** Thêm CPU/RAM features → target closer to measured congestion

**Vấn đề:** Google Cluster là resource trace, không phải web access log. Cần map features.

## 4. Kế Hoạch Hành Động

### Phase 1: Thêm Zanbil (1-2 ngày)

```
1. Tải Zanbil dataset
2. Đặt tại data/raw/zanbil/access.log
3. Chạy prepare_zanbil_logs.py
4. Chạy build_multi_source_dataset.py
5. Train tcn_attention_bilstm với NASA+Zanbil
6. So sánh kết quả
```

**Kỳ vọng:** R² tăng từ 0.33 → 0.45-0.55

### Phase 2: Thêm Calgary/ClarkNet (3-5 ngày)

```
1. Tải Calgary + ClarkNet logs
2. Viết parser cho format mới
3. Chuẩn hóa features
4. Build 4-source dataset (NASA+Zanbil+Calgary+ClarkNet)
5. Train lại
```

**Kỳ vọng:** R² tăng lên 0.55-0.65

### Phase 3: Data Augmentation (2-3 ngày)

```
1. Implement time warping
2. Implement SMOTE cho time series
3. Tăng positive cases (congestion events)
4. Train với augmented data
```

**Kỳ vọng:** Recall tăng từ 0.007 → 0.3-0.5

### Phase 4: Google Cluster Telemetry (5-7 ngày)

```
1. Tải Google Cluster sample
2. Map resource metrics → web congestion proxy
3. Kết hợp với web log features
4. Train multi-modal model
```

**Kỳ vọng:** Target closer to measured congestion

## 5. Kỳ Vọng Sau Khi Tăng Dữ Liệu

| Metric | Hiện tại | Sau Phase 1 | Sau Phase 2 | Sau Phase 3 |
|---|---|---|---|---|
| Training samples | 62K | ~150K | ~400K | ~800K |
| Sources | 1 | 2 | 4 | 4+augmented |
| R² | 0.331 | 0.45-0.55 | 0.55-0.65 | 0.60-0.70 |
| Recall (calibrated) | 0.979 | 0.95+ | 0.95+ | 0.90+ |
| F1 (calibrated) | 0.866 | 0.85+ | 0.88+ | 0.85+ |

## 6. Yêu Cầu Cho Paper Quốc Tế

Để publish ở conference/journal uy tín (ví dụ: IEEE, ACM, Elsevier):

| Yêu cầu | Hiện tại | Cần đạt |
|---|---|---|
| Dataset size | 62K windows | 200K+ windows |
| Data sources | 1 | 3+ |
| Data age | 1995 | 2015+ hoặc multi-era |
| R² | 0.331 | 0.60+ |
| Baseline comparison | 5 models | 8+ models (thêm ARIMA, Prophet, Transformer variants) |
| Ablation study | Chưa có | Cần có |
| Cross-validation | Chưa có | Cần k-fold |
| Statistical significance | Chưa có | Cần p-value |

## 7. Kết Luận

Vấn đề chính **không phải model** mà là **dữ liệu**:

1. **Quả ít** — 62K windows không đủ cho model phức tạp
2. **Quả cũ** — 1995 data không representative cho web hiện đại
3. **1 nguồn** — không có generalization
4. **Proxy target** — không phải ground truth

**Giải pháp ưu tiên:** Thêm Zanbil dataset trước (nhanh nhất, đã có framework), sau đó thêm Calgary/ClarkNet/WorldCup.
