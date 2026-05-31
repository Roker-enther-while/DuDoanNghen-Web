# CLAUDE.md — TCN-Attention-BiLSTM Project Rules

## PROJECT CHARTER

### Project name

Dự đoán nghẽn hệ thống web bằng mô hình trí tuệ nhân tạo dựa trên chuỗi thời gian

### Core problem

Dự đoán sớm nguy cơ nghẽn hệ thống web dựa trên chuỗi thời gian từ log/telemetry web, phân tích request count, response time, throughput, CPU usage, memory usage, error rate.

### Original goal

Xây dựng pipeline AI minh bạch để dự đoán proxy congestion score từ log web công khai, huấn luyện TCN-Attention-BiLSTM, đánh giá real public proxy result, hiệu chỉnh threshold cảnh báo, kiểm thử synthetic stress benchmark và trình bày dashboard minh bạch.

### Current accepted goal

Mục tiêu cuối cùng: biến repository này thành một hệ thống nghiên cứu + thực nghiệm + demo hoàn chỉnh, đúng phạm vi đề cương, có thể bảo vệ/trình bày được trước giảng viên. Không chỉ train model rồi in số liệu; sản phẩm phải có pipeline dữ liệu, mô hình, đánh giá, biểu đồ, báo cáo, demo cảnh báo sớm và bằng chứng kiểm thử rõ ràng.

### Final product goal

Một repo có thể chạy lại pipeline thực nghiệm đầy đủ: data pipeline, source governance, training framework, baseline models, TCN-Attention-BiLSTM, threshold calibration, synthetic stress benchmark, recommendation engine, demo cảnh báo sớm, final reports, dashboard minh bạch.

### Success criteria

- Data pipeline NASA HTTP → chuỗi thời gian hoạt động
- Train/val/test split theo thời gian, không data leakage
- Float16 storage, float32 training, Mixed Precision
- 6 mô hình đã train và so sánh
- TCN-Attention-BiLSTM full 120 epoch
- MAE, RMSE, R², Precision, Recall, F1, confusion matrix
- Threshold calibration tách biệt
- Synthetic stress benchmark tách riêng
- Recommendation Engine hoạt động
- Demo cảnh báo sớm hoạt động
- Dashboard dùng số thật
- Source governance/license manifest
- pytest pass

### Required deliverables

- Data pipeline (NASA HTTP → windows_fp16.npz)
- Source governance/license manifest
- Training framework (PyTorch CUDA)
- Baseline models (Moving Average, LSTM, GRU, TCN, Transformer)
- TCN-Attention-BiLSTM model
- Threshold calibration
- Synthetic stress benchmark
- Recommendation Engine
- Demo cảnh báo sớm
- Final reports (research summary, artifact manifest, gap analysis)
- Dashboard (số thật)
- Runbook
- Tài liệu báo cáo 5 chương

### Non-goals / out-of-scope

- Không gọi NASA target là measured congestion
- Không gọi synthetic là real-world
- Không claim cross-source khi chưa có Zanbil raw
- Không dùng số demo giả làm kết quả chính
- Không train multi-source khi Zanbil chưa có
- Không nhầm sang VietMIRA, multimedia retrieval, delivery route, BBC GoodFood, scheduler
- Không claim production auto-scaling
- Không claim TurboQuant nếu chưa implement

### Hard constraints

- Chỉ dùng số thật từ artifact đã train
- Proxy target phải ghi rõ trong mọi report/dashboard
- Synthetic tách riêng khỏi real public result
- Không bịa multi-source
- Không bịa measured congestion
- Threshold calibration phải giải thích tách biệt
- Không train lại nếu chỉ cần finalization

### Quality bar

- pytest pass
- JSON không NaN/Infinity
- Dashboard dùng số thật
- Report ghi rõ giới hạn
- Không có số demo giả

## WORKFLOW RULES

Trước khi làm việc, đọc: CLAUDE.md, NEXT_STEP.md, AGENT_REPORT.md, PHASE_LOG.md.

Cuối lượt, cập nhật AGENT_REPORT.md và PHASE_LOG.md.

Không commit khi verification chưa pass.

## NGUYÊN TẮC LÀM VIỆC

1. Không bịa kết quả.
2. Không tạo checkpoint giả.
3. Không ghi rằng đã train full nếu thực tế chỉ smoke test.
4. Không dùng dữ liệu không rõ nguồn nếu có rủi ro bản quyền.
5. Nếu thiếu dữ liệu thật, phải ghi rõ trạng thái BLOCKED/PARTIAL và dùng dữ liệu public hoặc synthetic benchmark có nhãn rõ ràng.
6. Mọi kết quả phải có file evidence.
7. Mọi bước phải chạy được bằng command cụ thể.
8. Không phá code đang chạy.
9. Không xóa file quan trọng nếu chưa backup hoặc chưa chắc chắn.
10. Mọi mở rộng được phép sáng tạo, nhưng phải nằm trong phạm vi đề cương.

## KNOWN PROJECT FACTS

- Branch: Không có git repo trong thư mục dự án
- Model chính: TCN-Attention-BiLSTM
- Data: NASA HTTP 1995
- Target: proxy congestion score
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Backend: PyTorch CUDA
- Epochs: 120/120
- Best epoch: 30
