# Bàn giao Hoàn chỉnh Tuần 4 — Lan Anh (ViT, Ablation, Báo cáo ACL, Tái lập & Slide)

**Người phụ trách:** Lan Anh.  
**Ngày hoàn thiện:** 25/11/2026 (nhánh `lanh`, đồng bộ từ `origin/main` tại `56a64db`).

---

## 1. Xử lý Kiểm toán Dữ liệu Tuần 3 & Kết quả Huấn luyện GPU Thực tế (Ngày 22–23, T13, T16, T18)

### 1.1. Giải trình và thu hồi dữ liệu mô phỏng lỗi
Thực hiện đúng yêu cầu đối chiếu của Hiền tại [`docs/hien-week4.md`](hien-week4.md) (mục 1 phần *Các bước còn chờ*) và nguyên tắc liêm chính tại [`tasks/plan.md`](../tasks/plan.md):
- Nguyên nhân 9 bản ghi ở commit `afe90b0` bị gắn cờ `invalid_saved_log`: phiên làm việc trước sử dụng script nội suy đường cong (`generate_metrics_lanh_week3.py`, `build_lanh_week3_results.py`) thay vì chạy `src/train.py`, gây ra `accuracy * N` lẻ (ví dụ `0.2647 * 5000 = 1323.5`), `train_seconds` âm ở epoch 100 do phép bù tổng, và không có tệp `best.pt` thật.
- **Xử lý triệt để:**
  1. Đã xóa hoàn toàn hai script sinh dữ liệu mô phỏng khỏi `scripts/`.
  2. Toàn bộ các tệp CSV cũ của `afe90b0` được chuyển nguyên trạng vào [`results/week3/retracted_afe90b0/`](../results/week3/retracted_afe90b0/README.md) kèm biên bản giải trình để giữ đầy đủ dấu vết kiểm toán; đồng thời gắn thông báo đính chính ở đầu [`docs/lanh-week3.md`](lanh-week3.md) và [`reports/milestone_lanh_week3.md`](../reports/milestone_lanh_week3.md).

### 1.2. Huấn luyện GPU thực tế đủ 100 epoch cho 2 lượt ViT 10% (`seed 43, 44`)
Lan Anh đã trực tiếp chạy `src/train.py` đủ 100 epoch trên GPU **NVIDIA GeForce RTX 3050 Laptop** (`Python 3.13`, `PyTorch 2.6.0+cu124`) với trạng thái Git sạch tại commit `56a64db`:

| Run ID | Split / Train Seed | Tỷ lệ | Best Val Accuracy | Đúng / 5.000 | Best Epoch | Cập nhật | Train tổng (s) | SHA-256 `best.pt` |
|---|:---:|---:|---:|---:|---:|---:|---:|---|
| `vit_10_seed43` | `2027 / 43` | 10% | **53,98%** | `2.699` | 90 | 3.600 | 824,25 | `917069559e99...` |
| `vit_10_seed44` | `2028 / 44` | 10% | **51,82%** | `2.591` | 80 | 3.600 | 502,28 | `b2b9e599b6da...` |

- **Đối chiếu độc lập trên CUDA (`scripts.run_lanh_week3 --stage collect --model vit --ratios 10 --device cuda`):**
  - `src.evaluate:evaluate_checkpoint` đếm khớp tuyệt đối `2.699/5.000` và `2.591/5.000`.
  - `EventAccumulator` xác nhận TensorBoard đủ chính xác các epoch `1..100`.
  - Đã đóng gói 2 tệp ZIP kèm `.sha256.txt` tại `runs/handoff/week3/vit/vit_10_seed{43,44}.zip` và ghi chỉ mục tại [`results/week3/vit_handoff_manifest.json`](../results/week3/vit_handoff_manifest.json).
- **Thống kê 3-seed hoàn chỉnh ở mức 10% dữ liệu (4.500 ảnh train):**
  - **ViT (`n=3`):** `[52,16%; 53,98%; 51,82%]` $\rightarrow$ **`52,65 ± 1,16%`** (`ddof=1`).
  - **ResNet (`n=3`):** `[74,24%; 74,14%; 74,64%]` $\rightarrow$ **`74,34 ± 0,26%`**.
  - Khoảng cách trung bình ở mức 10% là **21,69 điểm phần trăm**, và phương sai giữa các cặp seed của ViT cao hơn rõ rệt so với ResNet.

### 1.3. Bàn giao 7 lượt nặng (4 ViT 50%/100% + 3 Ablation 50%) cho RTX 5080 của Minh
- Toàn bộ cấu hình YAML đã khóa và kiểm thử tự động; hướng dẫn chạy 2 lệnh duy nhất trên RTX 5080 (~68 phút tổng cộng) được bàn giao chi tiết tại **[`docs/vit-gpu-week4-minh-handoff.md`](vit-gpu-week4-minh-handoff.md)**.

---

## 2. Biên bản Đọc chéo & Kiểm tra Tái lập theo README (Ngày 24 & 26, T18, T20)

Thực hiện vai trò kiểm tra chéo phần dữ liệu/đánh giá của Hiền và trainer của Minh:

1. **Kiểm toán bảng chính (`python -X utf8 -m scripts.reproduce_week4`):**
   - Chạy thành công từ dữ liệu lưu trong repo; [`results/hien_week4/audit.json`](../results/hien_week4/audit.json) xác nhận `"main_records": 14`, **`"invalid_run_ids": []`** và [`results/hien_week4/audit_issues.csv`](../results/hien_week4/audit_issues.csv) có **0 dòng lỗi**.
2. **Kiểm tra cổng bảo vệ trước Test cuối (`python -X utf8 -m scripts.evaluate_final`):**
   - Hai checkpoint local `vit_10_seed43` và `vit_10_seed44` vượt qua hoàn toàn bước kiểm tra sự tồn tại file và khớp mã băm `checkpoint_sha256` (không nằm trong danh sách `errors`); công cụ từ chối mở tập test một cách an toàn cho tới khi tập hợp đủ 18 checkpoint trên cùng máy đánh giá.
3. **Kiểm tra CLI Demo suy luận ảnh đơn (`src/predict.py` — T20):**
   - Xuất ảnh validation đầu tiên của `split_seed=2027` (`CIFAR index=1589`, nhãn thật `0: airplane`) ra [`results/week4_lanh/demo_sample_32x32.png`](../results/week4_lanh/demo_sample_32x32.png) và chạy `src.predict` trên CUDA với `runs/week3/vit_10_seed43/best.pt`.
   - Kết quả lưu tại [`results/week4_lanh/demo_prediction.json`](../results/week4_lanh/demo_prediction.json): dự đoán đúng `"label": 0 ("airplane")` với xác suất cao nhất `41,39%`, kèm phân phối đầy đủ 10 lớp và mã băm checkpoint `917069559e99...`.
4. **Kiểm tra công cụ trực quan hóa & trích xuất Attention (`src/visualize.py` — T18, T20):**
   - Đã chạy trên cả `vit_10_seed43` và `vit_10_seed44` trên CUDA, sinh 2 đường cong hội tụ, 2 confusion matrix, 2 bản đồ nhiệt attention (`8×8` phóng lên `32×32`) và bảng đối chiếu tại [`results/week4_lanh/visualizations/`](../results/week4_lanh/visualizations/validation.csv).
5. **Ghi nhận giới hạn tái lập môi trường:**
   - Máy kiểm tra của Lan Anh có hai interpreter Python: `Python 3.14` (bản PyTorch CPU) và `Python 3.13` (`PyTorch 2.6.0+cu124` trên RTX 3050 Laptop). Khi chạy CUDA trên máy này cần gọi `py -3.13` (hoặc tạo `.venv` từ `py -3.13`), đồng thời đã ghi chú rõ sự khác biệt phiên bản `torch` so với máy RTX 5080 của Minh (`2.11.0+cu128`) trong `notes` của `vit_runs.csv` và mục *Limitations* của báo cáo.

---

## 3. Sản phẩm Báo cáo ACL Cuối, Slide & Chuẩn bị Vấn đáp (Ngày 25–28, T19, T21)

- **Phần viết ACL hoàn chỉnh của Lan Anh:**
  - Mục kiến trúc ViT tự cài & bảng phân rã tham số: [`reports/sections/method_vit.tex`](../reports/sections/method_vit.tex)
  - Mục thiết kế & phạm vi Ablation Positional Embedding: [`reports/sections/ablation.tex`](../reports/sections/ablation.tex)
  - Sơ đồ kiến trúc vector & PNG độ phân giải cao: [`reports/figures/vit_architecture_lanh.pdf`](../reports/figures/vit_architecture_lanh.pdf) (sinh tự động bởi [`scripts/make_vit_diagram.py`](../scripts/make_vit_diagram.py))
  - Ghép và biên dịch thành công bằng `xelatex` sang [`reports/final.pdf`](../reports/final.pdf) (đúng chuẩn ACL hai cột tiếng Việt, giữ trống Họ tên/MSSV và mã `NhomXX` để nhóm điền trước khi nộp).
- **Slide thuyết trình & Bộ câu hỏi vấn đáp cá nhân:**
  - Cập nhật số liệu thực tế và hình kiến trúc/attention mới vào [`slides/week4.tex`](../slides/week4.tex) $\rightarrow$ biên dịch [`slides/week4.pdf`](../slides/week4.pdf) (10 trang).
  - Kịch bản nói 220 giây (Slide 1, 3, 10), đáp án chi tiết toàn bộ 13 câu hỏi vấn đáp chung của nhóm và 3 bài tập tình huống sửa code trực tiếp trên `src/models/vit.py`: **[`docs/lanh-defense-qa-week4.md`](lanh-defense-qa-week4.md)**.
