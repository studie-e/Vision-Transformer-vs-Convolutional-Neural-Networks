# Bàn giao vận hành GPU RTX 5080 (Tuần 4) — 7 lượt ViT & Ablation còn lại

**Người giao:** Lan Anh (chủ trì kiến trúc ViT, cấu hình YAML và kiểm thử).  
**Người nhận vận hành GPU:** Minh (hạ tầng RTX 5080, `torch==2.11.0+cu128`).  
**Người nhận kết quả & đánh giá cuối:** Hiền (audit, ghép lỗi, attention đa lớp, test cuối).

---

## 1. Bối cảnh và trạng thái hiện tại

1. **Đã hoàn thành & xác minh độc lập bằng checkpoint thật:**
   - **3 lượt ViT P0 (`seed=42`, `split_seed=2026`):** `week2_vit_{10,50,100}_seed42` (đã chạy trên RTX 5080 ở Tuần 2: `52,16% / 67,66% / 75,90%`).
   - **2 lượt ViT 10% P1 (`seed=43, 44`):** `vit_10_seed43` và `vit_10_seed44` (Lan Anh đã trực tiếp chạy đủ 100 epoch trên GPU NVIDIA GeForce RTX 3050 Laptop trong Tuần 4, sinh đầy đủ `best.pt`, `last.pt`, `metrics.csv`, TensorBoard và gói ZIP kèm SHA-256).
   - Như vậy mức **ViT 10% đã đủ 3/3 seed thật**.

2. **7 lượt nặng còn lại nhờ Minh chạy trên RTX 5080** (để đồng nhất môi trường tốc độ cao với P0, ước tính tổng thời gian chỉ **~68 phút** trên RTX 5080 thay vì ~3,8 giờ trên laptop RTX 3050):
   - **4 lượt ViT chính (T13):**
     - `configs/week3/vit_50_seed43.yaml` (`split_seed=2027, seed=43, ratio=0.50`, ~8 phút)
     - `configs/week3/vit_100_seed43.yaml` (`split_seed=2027, seed=43, ratio=1.00`, ~15 phút)
     - `configs/week3/vit_50_seed44.yaml` (`split_seed=2028, seed=44, ratio=0.50`, ~8 phút)
     - `configs/week3/vit_100_seed44.yaml` (`split_seed=2028, seed=44, ratio=1.00`, ~15 phút)
   - **3 lượt Ablation bỏ Positional Embedding ở 50% dữ liệu (T16, 268.042 tham số):**
     - `configs/week3/ablation_vit_50_seed42.yaml` (`split_seed=2026, seed=42, ratio=0.50`, ~7,5 phút)
     - `configs/week3/ablation_vit_50_seed43.yaml` (`split_seed=2027, seed=43, ratio=0.50`, ~7,5 phút)
     - `configs/week3/ablation_vit_50_seed44.yaml` (`split_seed=2028, seed=44, ratio=0.50`, ~7,5 phút)

---

## 2. Lệnh thực thi chuẩn trên máy RTX 5080 của Minh

Chạy từ thư mục gốc repo trong môi trường CUDA của Minh (`Python 3.13.7`, `torch==2.11.0+cu128`):

### Bước 1: Đồng bộ nhánh `lanh` và kiểm tra dữ liệu CIFAR-10
```powershell
git fetch origin
git checkout lanh
git pull origin lanh
python -X utf8 -m src.data --verify --verify-labels
```

### Bước 2: Chạy 4 lượt ViT 50% và 100% (`seed 43, 44`)
Script `scripts/run_lanh_week3.py` đã được bổ sung cờ `--ratios` để chỉ chạy đúng 4 lượt `50%` và `100%` (tự động hỗ trợ `--resume` nếu ngắt quãng):
```powershell
python -X utf8 -m scripts.run_lanh_week3 --stage train --model vit --ratios 50 100 --device cuda
```
*(Nếu trên máy Minh cũng muốn chạy lại luôn 2 lượt `10%` trên RTX 5080 để cả 9 lượt ViT đồng nhất tuyệt đối phần cứng RTX 5080 + `torch 2.11.0+cu128`, chỉ cần bỏ `--ratios 50 100` — mất thêm vỏn vẹn ~3,5 phút!)*

### Bước 3: Đối chiếu độc lập và đóng gói ZIP cho ViT
- Nếu đã có đủ cả 6 thư mục `runs/week3/vit_{10,50,100}_seed{43,44}` trên máy Minh (giải nén 2 gói ZIP `vit_10_seed{43,44}` của Lan Anh vào `runs/week3/` hoặc chạy luôn trên RTX 5080):
```powershell
python -X utf8 -m scripts.run_lanh_week3 --stage collect --model vit --device cuda
```
- Nếu chỉ thu thập riêng 4 lượt `50%` và `100%`:
```powershell
python -X utf8 -m scripts.run_lanh_week3 --stage collect --model vit --ratios 50 100 --device cuda
```

### Bước 4: Chạy và thu thập 3 lượt Ablation bỏ Positional Embedding (`seed 42, 43, 44`)
```powershell
python -X utf8 -m scripts.run_lanh_week3 --stage train --model ablation --device cuda
python -X utf8 -m scripts.run_lanh_week3 --stage collect --model ablation --device cuda
```
Lệnh `collect --model ablation` sẽ tự động kiểm tra `best.pt` trên CUDA, xác nhận 100 epoch TensorBoard, đóng gói 3 file ZIP kèm `.sha256.txt` tại `runs/handoff/week3/ablation/`, đồng thời ghi `results/week3/ablation_runs.csv`, `results/week3/ablation_summary.csv` và `results/ablation.csv`.

---

## 3. Kiểm tra tính hợp lệ trước khi chuyển Hiền đánh giá Test cuối

Sau khi hoàn tất Bước 3 và Bước 4, chạy công cụ kiểm toán Tuần 4 của Hiền ngay trên máy Minh:

```powershell
python -X utf8 -m scripts.reproduce_week4
python -X utf8 -m scripts.evaluate_final
```

**Tiêu chuẩn nghiệm thu bắt buộc:**
1. `results/hien_week4/audit.json` báo `"main_records": 18`, `"invalid_run_ids": []`, `"local_main_checkpoints": 18`.
2. `scripts.evaluate_final` (chưa truyền `--execute`) in `"errors": []`.
3. Chuyển gói `runs/handoff/week3/vit/` và `runs/handoff/week3/ablation/` kèm `handoff_manifest.json` cho Hiền để Hiền khóa danh sách và chạy đánh giá tập test cuối (`--execute`) một lần duy nhất.
