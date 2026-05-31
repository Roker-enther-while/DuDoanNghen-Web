# BÁO CÁO THỰC NGHIỆM DATA RESET + BIG LOGS + MODEL SELECTION

## 1. Mục tiêu thay đổi
- Chuyển từ CSV-only sang SQL/local data pool.
- Loại 20% test cũ khỏi training.
- Bổ sung log/trace từ nhiều nguồn uy tín khi có thể tải được.
- Tạo thêm 20% synthetic noisy continuous data trên lượng external thực tế đã nạp.
- Thử nhiều biến thể mô hình nhẹ hơn.
- Chọn model tốt nhất cho dự đoán nghẽn web theo metrics.

## 2. Trạng thái repo và môi trường
Xem `git_state_before.md`, `environment.md`, `repo_audit.md`.

## 3. Archive dữ liệu/model cũ
Artifact cũ đã được copy vào `archive_previous_runs/`. Mặc định không xóa vĩnh viễn vì không có flag purge. Xem `archive_manifest.csv`.

## 4. Xử lý dữ liệu cũ
- Tổng số dòng Bitbrains cũ: 0
- 80% train cũ: None
- 20% holdout cũ: None
- Có dùng holdout để train không: Không.

## 5. Nguồn dữ liệu mới
- Trạng thái: `UNKNOWN`
- External rows đã nạp: None
- Số nguồn inventory: None
- Nếu chưa đạt 2,000,000 dòng, run này không tuyên bố full success. Xem `source_download_failures.md` và `tables/table_06_source_inventory.md`.

## 6. Synthetic noisy data
- Synthetic rows: None
- Tỷ lệ: None
- Synthetic data chỉ dùng tăng độ bền thử nghiệm, không phải dữ liệu thật.

## 7. Training pool cuối
- External real rows: None
- Synthetic rows: None
- Old train80 rows: None
- Total train_pool rows: None
- Validation rows: None
- SQLite DB: `None`

## 8. Mô hình và biến thể đã thử
Xem `tables/table_07_model_variants.md` và `tables/table_09_model_selection_metrics.md`.

## 9. Kết quả model selection
- best_by_rmse: moving_average
- best_by_r2: moving_average
- best_by_f1: lstm32
- best_by_latency: persistence
- recommended_model_for_report: moving_average

## 10. Kết quả model tốt nhất
Checkpoint nằm trong `models/` nếu model neural chạy thành công. Metrics chi tiết ở `model_ranking.csv`.

## 11. Đánh giá holdout 20% cũ nếu có
Holdout cũ được giữ riêng trong `holdout_old_20pct/` và không dùng train/tune. Cross-domain holdout chưa chạy trong run này.

## 12. Kết luận khoa học
Model khuyến nghị hiện tại là `moving_average` theo bảng metrics. Nếu baseline đứng đầu, kết luận phải ghi baseline tốt hơn các mô hình attention trong run này.

## 13. Hạn chế
- Dữ liệu 4 nguồn không hoàn toàn là web log production.
- Cluster/VM trace chỉ là proxy cho workload hệ thống.
- Synthetic noisy data chỉ dùng tăng độ bền, không thay thế dữ liệu thật.
- Nếu thiếu nguồn hoặc thiếu 2M dòng thì trạng thái là PARTIAL_DATA và đã ghi rõ.
- Nếu model chưa vượt baseline thì giữ nguyên kết quả theo metrics.

## 14. Artifact index
Xem `tables/table_16_artifact_index.md`, `figures/`, `figure_data/`, `raw_console.log`, `train_commands.log`.
