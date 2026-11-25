# Bàn giao tuần 4 của Minh

Phân công của nhóm:
- **Lan Anh** phụ trách mô hình, mã và kết quả ViT/ablation. Minh hỗ trợ vận hành các lượt chạy này trên GPU.
- **Hiền** phụ trách dữ liệu, mã đánh giá, thống kê, trực quan hóa và kết quả test cuối. Minh hỗ trợ vận hành GPU nhưng công việc và kết quả đánh giá vẫn ghi nhận cho Hiền.
- **Minh** phụ trách mô hình ResNet, thí nghiệm mở rộng Swin, trainer, kiểm tra tái lập và demo.

## Thí nghiệm và bằng chứng

Repo dùng CIFAR-10 với ảnh RGB 32×32. Mỗi cặp seed tách phân tầng 45.000 ảnh train và 5.000 ảnh validation từ tập train gốc, giữ riêng 10.000 ảnh test chính thức. Hai baseline dùng chung manifest ở từng cặp `(split_seed, train_seed)`: `(2026, 42)`, `(2027, 43)`, `(2028, 44)`. Mỗi checkpoint `best.pt` chọn theo validation accuracy cao nhất, hòa điểm giữ epoch sớm hơn. Mỗi lượt huấn luyện 100 epoch, batch size 128; ba tỷ lệ dữ liệu (10%, 50%, 100%) có lần lượt 3.600, 17.600 và 35.200 bước cập nhật.

| Mô hình | Cấu hình Minh phụ trách giải thích | Phạm vi đánh giá |
|---|---|---|
| ResNet | Stem 3×3 stride 1, ba stage kênh 16/32/64, 3 basic block mỗi stage, projection shortcut 1×1 khi đổi shape, global average pooling, Linear 10 lớp. 272.474 tham số. SGD, LR 0,05, momentum 0,9, weight decay 0,0005. | Baseline chính: 10%, 50%, 100% × 3 cặp seed. Chọn `best.pt` trên validation, đánh giá một lần trên 10.000 ảnh test sau khi khóa danh sách 18 checkpoint. |
| Swin | `SwinTransformer` của `torchvision`, patch 2×2, cửa sổ 4×4, 3 stage kênh 26/52/104, mỗi stage 2 block, 2/4/8 head. 261.824 tham số. AdamW, LR 0,0003, weight decay 0,05, stochastic depth 0,1, dropout 0. | Thí nghiệm mở rộng chỉ trên validation: 10% và 100% × 3 cặp seed (cùng 1 lượt smoke 1 epoch từ tuần 3 để lưu dấu vết kỹ thuật). Không đưa vào đánh giá test chính. |

Nguồn kiến trúc và khai báo dùng AI nằm trong `reports/final.tex`, `reports/references.bib` và đầu từng file mô hình. Minh kiểm tra trực tiếp phép cộng residual, nhánh projection khi đổi shape và cách truyền cấu hình vào `SwinTransformer` của `torchvision`. Chín CSV ViT/ablation sinh từ số đặt trước trong bản tuần 3 (commit `56a64db`) đã được thay bằng kết quả thu trực tiếp từ checkpoint và `metrics.csv` thật qua `--stage collect`.

## Tái lập từ môi trường sạch và chạy demo

Chạy từ gốc repo trên Windows với Python 3.13, dữ liệu CIFAR-10 tại `data/` và ba manifest tại `results/splits/`. Nếu kiểm tra trên máy khác, giải nén gói `runs/handoff/week4/minh_week4.zip` tại gốc repo (đối chiếu SHA-256 trong `runs/handoff/week4/minh_week4.sha256.txt` và `results/minh_week4/handoff.json`) để khôi phục đúng cây thư mục `runs/`.

```powershell
python -m venv runs/week4_env
runs/week4_env/Scripts/python -m pip install -r requirements-dev.txt
runs/week4_env/Scripts/python -m pip check
runs/week4_env/Scripts/python -X utf8 -m src.data --verify --verify-labels
runs/week4_env/Scripts/python -X utf8 -m src.predict --image results/minh_week4/demo_input.png --checkpoint runs/week2/resnet_d100_s42/best.pt --out results/minh_week4/demo_resnet.json --device cpu
```

Demo dùng ảnh validation đầu tiên của manifest 2026 (chỉ số CIFAR-10 `23171`). Trên CPU trong `runs/week4_env`, checkpoint ResNet 100% seed 42 dự đoán đúng nhãn **airplane** với xác suất `0,999908` (99,99%), xuất đầy đủ phân phối xác suất 10 lớp, thiết bị chạy và mã băm SHA-256 của `best.pt`. Bản đồ attention của ViT dùng công cụ trực quan hóa của Hiền và hàm `get_attention_weights()` của Lan Anh.

## Lệnh hoàn tất quy trình tuần 4

Lệnh dưới đây thực hiện tuần tự: tiếp tục từ `last.pt` (nếu còn dở) hoặc xác nhận đủ 100 epoch, thu kết quả validation thật (`collect` cho ViT, ablation, ResNet, Swin), chạy `scripts.reproduce_week4 --strict`, huấn luyện lặp lại ResNet 10% seed 42 trong `runs/week4_env` để so khớp bitwise trọng số và lịch sử học, đóng gói `minh_week4.zip`, chạy demo `src.predict`, xuất phân tích lỗi ghép cặp trên cùng ảnh validation bằng `scripts.analyze_hien_week3`, kiểm tra danh sách 18 checkpoint đã khóa bằng `scripts.evaluate_final`, đánh giá tập test cuối (khi bật `-ExecuteTest`), sinh bảng `.tex` từ CSV thật qua `scripts.week4_tables`, biên dịch PDF và chạy toàn bộ `pytest`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/finish_week4.ps1 -ExecuteTest
```

Khi MiKTeX `lualatex` trên máy thiếu gói `lua-uni-data`, script tự động dùng `runs/week4_tex/tectonic.exe -X compile` (XeTeX + BibTeX, hỗ trợ đầy đủ Times New Roman, Arial và tiếng Việt) cho `reports/final.tex`, `reports/milestone.tex` và `slides/week4.tex`.

| Sản phẩm đầu ra | Cách đối chiếu |
|---|---|
| `results/hien_week4/` (`per_seed.csv`, `summary.csv`, `checkpoint_registry.csv`, `audit_issues.csv`, `audit.json`) | Kiểm tra toàn bộ 28 bản ghi validation (18 baseline chính, 3 ablation, 6 Swin mở rộng, 1 Swin smoke): `audit_issues.csv` rỗng, đủ 3 cặp seed mỗi nhóm, khớp SHA-256 của `best.pt` và manifest. |
| `results/minh_week4/verified.csv` và `handoff.json` | Đối chiếu SHA-256 `best.pt`, số ảnh đúng trên 5.000 validation, `best_epoch`, `manifest_sha256` của cả 27 lượt 100 epoch; xác nhận lượt lặp lại `resnet_10_seed42` khớp tuyệt đối tensor (`best.pt`, `last.pt`) và lịch sử học. |
| `runs/handoff/week4/minh_week4.zip` và `.sha256.txt` | Gói nén nội bộ giữ nguyên đường dẫn `runs/` (checkpoint, config, metadata, `metrics.csv`, TensorBoard) kèm `verified.csv` và mã băm SHA-256. |
| `results/minh_week4/paired/` | Kiểm tra đủ 18/18 baseline validation (`missing=0`) và ghép lỗi trên cùng 5.000 ảnh validation ở mức 50% qua ba cặp seed (`d50_s42`, `d50_s43`, `d50_s44`), kèm ma trận nhầm lẫn và bản đồ attention 20 ảnh cố định. |
| `results/final_test/` | Do Hiền phụ trách phương pháp và kết quả: giữ `selected_registry.csv` (18 checkpoint đã khóa trước test), `runs.csv` (10.000 ảnh test mỗi lượt) và `summary.csv` (mean ± SD mẫu `ddof=1`). |
| `reports/final.pdf`, `reports/milestone.pdf`, `slides/week4.pdf` và `results/hien_week4/hien_week4_internal.zip` | Bảng và số liệu chèn trực tiếp từ các tệp `.tex` do `scripts.week4_tables` tính từ CSV thật; đóng gói kèm `package_manifest.json` và `.sha256.txt`. |

## Phần trình bày của Minh (3 phút 30 giây)

- **90 giây — Slide ResNet:** Đi theo một ảnh 32×32 qua stem 3×3 và ba stage 16/32/64. Giải thích nhánh tắt giữ đường truyền gradient, projection 1×1 đổi kênh và kích thước để cộng được hai tensor. Đối chiếu 272.474 tham số của ResNet với 274.282 của ViT (lệch 0,66%).
- **60 giây — Slide huấn luyện & checkpoint:** Giải thích CrossEntropyLoss nhận logits thô, SGD + cosine annealing cho ResNet, kiểm tra `manifest_sha256` khi resume, tách vai trò `best.pt` (suy luận/đánh giá) và `last.pt` (khôi phục optimizer, scheduler, RNG).
- **60 giây — Slide tái lập & demo:** Trình bày ảnh validation cố định chỉ số 23171, đầu ra xác suất 10 lớp và SHA-256 của `src.predict`, kết quả tái lập bitwise ResNet 10% seed 42; nếu hội đồng hỏi về Swin, giải thích patch 2×2, patch merging qua lưới 16×16 → 8×8 → 4×4 và cửa sổ dịch chuyển 4×4 trong phạm vi mở rộng trên validation.

## Các mục còn cần cả nhóm xác nhận trước khi nộp

1. Điền đầy đủ họ tên, MSSV của Lan Anh, Hiền, Minh và mã nhóm vào `reports/final.tex`, `reports/milestone.tex`, `slides/week4.tex`, sau đó đổi tên bản nộp theo quy định môn (`NhomXX_DeTai03.pdf`).
2. Cả ba thành viên đọc duyệt chéo số liệu cuối, xác nhận tỷ lệ đóng góp thực tế và lời khai báo sử dụng AI.
3. Tổ chức diễn tập chung toàn nhóm 10–12 phút để chốt thời lượng chuyển ý giữa ba người.
