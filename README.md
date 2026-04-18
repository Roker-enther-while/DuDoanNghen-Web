# WebTAB: Mô hình Lai TCN-FTAtt-BiLSTM Dự báo Nghẽn Mạng Web (V6.0 Gold)

> [!IMPORTANT]
> **Phiên bản Nghiên cứu**: Đây là bản thực thi chính thức của khung công tác **WebTAB** (Web-focused Temporal Attention Bidirectional). Giải pháp SOTA (State-of-the-Art) cho quản trị hạ tầng web chủ động, kết hợp tích chập thời gian dilated, bộ nhớ tái hồi dài hạn và cơ chế chú ý kép Feature-Temporal (FTAtt).

---

## 📖 Giới thiệu Tổng quan
Trong các hạ tầng Cloud-native hiện đại, việc dự báo tải chính xác là điều kiện tiên quyết để đảm bảo tuân thủ SLA (Service Level Agreement) và tối ưu hóa năng lượng. Nghiên cứu này đề xuất **TCN-FTAtt-BiLSTM**, một kiến trúc deep learning lai độc quyền giúp bắt kịp các đỉnh nhọn truy cập đột biến (Flash Crowds) với độ trễ cực thấp.

### Điểm nhấn Công nghệ:
- **TCN (Temporal Convolutional Network):** Trích xuất đặc trưng cục bộ với trường thu nhận (receptive field) lớn linh hoạt.
- **FTAtt (Feature-Temporal Dual Attention):** Đột phá của phiên bản V6, tách biệt việc gắn trọng số cho 13 đặc trưng vật lý và các bước thời gian, giúp giảm độ phức tạp tính toán và tăng tốc độ suy luận.
- **BiLSTM:** Học các phụ thuộc thời gian hai chiều để bảo toàn ngữ cảnh dài hạn.
- **Half-Precision (Float16):** Tận dụng tối đa Tensor Cores trên GPU RTX 4060 để đạt tốc độ xử lý vượt trội.

---

## 🚀 Thông số Hiệu năng (Benchmarks)
Dựa trên tập dữ liệu thực tế **8.5 triệu mẫu** log telemetry:

| Chỉ số | Kết quả V6 | Đánh giá |
| :--- | :---: | :--- |
| **Độ chính xác (R2)** | **0.982** | Xuất sắc |
| **Độ trễ (Latency)** | **2.35ms** | Real-time tối ưu |
| **Failure Recall** | **94.2%** | Bám đỉnh cực tốt |
| **SLA Compliance** | **93.5%** | An toàn hạ tầng |


---

## 🛠️ Hướng dẫn Sử dụng nhanh

### 1. Cài đặt môi trường
Yêu cầu Python 3.10+ và TensorFlow 2.10 (Native Windows GPU).
```bash
pip install -r requirements.txt
```

### 2. Huấn luyện mô hình (V6 Ultra-Fast)
```bash
python src/tools/train_ultrafast_v6.py
```

### 3. Tạo biểu đồ nghiên cứu
```bash
python src/tools/generate_v6_sota_figures.py
```

---

## ✍️ Tác giả & Bản quyền
Dự án được phát triển cho mục đích Nghiên cứu Khoa học và Luận văn tốt nghiệp. Mọi hành vi sao chép cần được trích dẫn theo chuẩn IEEE/APA.

**Từ khóa:** Web congestion, TCN, BiLSTM, Feature-Temporal Attention, AIOps, Cloud Optimization.
