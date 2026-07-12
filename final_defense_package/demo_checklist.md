# DEMO CHECKLIST

## 5 PHÚT TRƯỚC KHI BẢO VỆ

### Kiểm tra máy tính
- [ ] Laptop sạc đầy pin
- [ ] Mở sẵn terminal/command prompt
- [ ] Mở sẵn thư mục project
- [ ] Kiểm tra kết nối internet (nếu cần)

### Mở sẵn các file
- [ ] Dashboard HTML: `outputs/web/research_defense_dashboard.html`
- [ ] Metrics JSON: `outputs/metrics/full_120_v2_tcn_attention_bilstm/final_metrics.json`
- [ ] Audit Report: `outputs/reports/23_final_hard_audit_report.md`
- [ ] Figures: `outputs/figures/prediction_vs_actual.png`

### Chạy thử lệnh
```bash
# Kiểm tra pytest
cd "C:\Users\dhp01\OneDrive\Máy tính\TCN-Attention-BiLSTM"
python -m pytest -q

# Kiểm tra demo
python scripts/run_demo.py --sample synthetic --model best --explain --save-report
```

### Chuẩn bị fallback
- [ ] Screenshot dashboard nếu không mở được HTML
- [ ] Screenshot figures nếu không mở được PNG
- [ ] Bản in metrics JSON nếu cần

---

## LỆNH DEMO

### 1. Mở Dashboard
```bash
start outputs/web/research_defense_dashboard.html
```

### 2. Chạy Terminal Demo
```bash
python scripts/run_demo.py --sample synthetic --model best --explain --save-report
```

### 3. Xem Metrics
```bash
cat outputs/metrics/full_120_v2_tcn_attention_bilstm/final_metrics.json
```

### 4. Chạy Tests
```bash
python -m pytest -q
```

### 5. Xem Figures
```bash
start outputs/figures/prediction_vs_actual.png
start outputs/figures/model_comparison_rmse.png
```

### 6. Xem Audit Report
```bash
cat outputs/reports/23_final_hard_audit_report.md
```

---

## NẾU DEMO LỖI

### Dashboard không mở
→ Dùng screenshot đã chụp sẵn
→ Đọc metrics từ JSON

### Terminal demo lỗi
→ Đọc kết quả từ `outputs/metrics/18_terminal_demo_output.json`
→ Hiển thị ảnh figures

### pytest fail
→ Giải thích: "Tests kiểm tra logic code, không ảnh hưởng đến kết quả model đã train"

### Không có internet
→ Demo offline, dùng file local

---

## THỨ TỰ TRÌNH BÀY

1. Mở slide giới thiệu
2. Nói về vấn đề nghiên cứu
3. Giới thiệu dataset và pipeline
4. Giải thích mô hình TCN-Attention-BiLSTM
5. **Chuyển sang demo:**
   - Mở dashboard HTML
   - Chạy terminal demo
   - Hiển thị figures
6. Trình bày kết quả
7. Nói về hạn chế
8. Kết luận

---

## GHI CHÚ

- Nói chậm, rõ ràng
- Nhìn thầy khi trả lời
- Không đọc slide word-for-word
- Nếu không biết câu trả lời: "Em sẽ tìm hiểu thêm ạ"
