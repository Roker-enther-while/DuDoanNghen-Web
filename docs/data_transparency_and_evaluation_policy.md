# Data Transparency And Evaluation Policy

## 1. Nguyen tac du lieu

- Chi dua dataset vao pipeline khi co license, citation, va provenance ro rang.
- Khong dung dataset scraped trai phep hoac dataset khong xac dinh duoc quyen su dung.
- Khong publish raw web logs neu co rui ro PII nhu IP, hostname, query string, user-agent.
- Client identifier phai duoc hash bang salt cuc bo truoc khi tao processed artifact.
- Query string va user-agent phai bi drop mac dinh tru khi co ly do nghien cuu va policy rieng.
- Moi artifact processed phai co `source_id` hoac metadata truy nguoc ve source manifest.

## 2. Phan loai ket qua

- `real_public_test`: danh gia tren tap test tu public real logs, vi du NASA HTTP.
- `cross_source_test`: train tren mot hoac nhieu source va test tren source khac de kiem tra generalization.
- `synthetic_stress_test`: danh gia co kiem soat tren stress benchmark sinh tu public baseline.

Khong tron ba nhom ket qua nay thanh mot ket luan duy nhat. Synthetic stress chi dung de do kha nang phat hien tinh huong stress co kiem soat, khong duoc goi la ket qua real-world.

## 3. Cach trinh bay minh bach

- NASA HTTP khong co CPU, memory, response_time that; target hien tai la proxy congestion score.
- Synthetic stress benchmark la mo phong co gan `is_synthetic=true`, `scenario_name`, va `generation_config`.
- Google Cluster Trace la resource workload trace, khong phai web access log.
- Moi bang ket qua phai ghi ro source, split mode, threshold policy, target type, va positive case count.
- Neu dung threshold theo quantile, can ghi split tham chieu va gia tri threshold da resolve.

## 4. Dieu kien duoc ket luan

Chi ket luan model A tot hon model B khi:

- Cung data artifact va split.
- Cung target type.
- Cung threshold policy.
- Cung metric va cach tinh metric.
- Test set co du positive cases cho alert metric.
- License/provenance/citation day du trong source manifest.
- Synthetic va real result duoc bao cao rieng.

Neu bat ky dieu kien nao chua dat, chi duoc goi la diagnostic hoac stress benchmark result, khong goi la ket luan khoa hoc cuoi cung.
