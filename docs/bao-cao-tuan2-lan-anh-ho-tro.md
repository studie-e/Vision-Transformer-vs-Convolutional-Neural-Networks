# Báo cáo Thực thi Thí nghiệm Tuần 2 — Vision Transformer (ViT)
## Ghi nhận: Hiền và Minh hỗ trợ Lan Anh

---

## 1. Bối cảnh và Phân công Hỗ trợ Nội bộ

- **Mô hình và Kiến trúc:** Toàn bộ mã nguồn Vision Transformer (ViT 274.282 tham số) do **Lan Anh** tự cài đặt hoàn chỉnh từ đầu bằng PyTorch ([`src/models/vit.py`](../src/models/vit.py)), đã kiểm thử gradient, parameter parity với ResNet (chênh 0,66%) và khảo sát siêu tham số ban đầu.
- **Hạ tầng phần cứng:** Theo thỏa thuận hàng đợi tài nguyên trong [kế hoạch chung](../tasks/plan.md), máy tính của **Minh** có hạ tầng GPU RTX (CUDA). Do đó, **Minh đảm nhận vận hành chạy 3 lượt huấn luyện 100 epoch của ViT trên GPU hỗ trợ Lan Anh** để đảm bảo tiến độ chung của nhóm.
- **Dữ liệu và Nghiệm thu:** **Hiền** chuẩn bị bộ cấu hình chuẩn (`configs/week2/`), pipeline dữ liệu CIFAR-10 phân tầng theo seed, và chịu trách nhiệm tiếp nhận toàn bộ thư mục run từ Minh để đối chiếu dữ liệu, vẽ biểu đồ learning curve, ma trận nhầm lẫn và bản đồ chú ý (attention map).

---

## 2. Cấu hình Thí nghiệm Chốt Tuần 2 (T09, T10)

Cả nhóm đã thống nhất khóa chặt chẽ toàn bộ giao thức thí nghiệm trước khi mở loạt chạy chính:

| Hạng mục | Quy ước chốt | Ghi chú kỹ thuật |
|---|---|---|
| **Dữ liệu** | CIFAR-10 gốc | 45.000 ảnh train, 5.000 ảnh validation (phân tầng); 10.000 test giữ riêng cho bản cuối. |
| **Các mức dữ liệu** | 10%, 50%, 100% | Tương ứng 4.500, 22.500, 45.000 ảnh. Các tập con lồng nhau ($10\% \subset 50\% \subset 100\%$). |
| **Seed cố định** | `split_seed=2026`, `seed=42` | Cả ViT và ResNet dùng chung đúng danh sách ảnh cho từng mức tỷ lệ. |
| **Ngân sách train** | 100 epochs, batch size 128 | Lịch giảm tốc độ học: `CosineAnnealingLR` theo epoch. |
| **Cấu hình ViT (Lan Anh)** | **AdamW, lr=0.0003, weight_decay=0.05** | Dropout 0.1, 4 blocks, 4 heads, $D=96$, MLP ratio 1.5 (274.282 params). |
| **Cấu hình ResNet (Minh)** | **SGD, lr=0.05, momentum=0.9, weight_decay=0.0005** | Width 16, 3 blocks/stage (272.474 params, chênh lệch 0,66%). |
| **Tiêu chí Checkpoint** | Top-1 Accuracy trên 5.000 ảnh val | Chọn `best.pt` theo validation accuracy cao nhất. **Nếu hòa điểm thì giữ epoch sớm hơn** (`val_accuracy > best`). |
| **Tệp cấu hình** | `configs/week2/` trên nhánh `hien` / `lanh` | `vit_10.yaml`, `vit_50.yaml`, `vit_100.yaml`. |

---

## 3. Hướng dẫn Từng bước cho Minh Vận hành trên GPU

### Bước 1: Đồng bộ mã nguồn & Kiểm tra Dữ liệu
```powershell
# Chuyển sang nhánh lanh và kéo commit mới nhất
git checkout lanh
git pull origin lanh

# Xác thực dữ liệu CIFAR-10 và các file manifest
python -X utf8 -m src.data --verify --verify-labels
```
*(Yêu cầu: Terminal báo `[OK]` cho cả 3 seed 2026, 2027, 2028).*

### Bước 2: Kiểm tra Thiết bị GPU CUDA
```powershell
python -c "import torch; print('CUDA Ready:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"
```

### Bước 3: Lệnh Thực thi 3 Lượt Huấn luyện ViT (100 Epochs)

#### 🚀 Cách 1: Chạy tự động liên tục cả 3 lượt (Khuyên dùng để cắm máy chạy ban đêm):
```powershell
python -X utf8 -m src.train --config configs/week2/vit_10.yaml --run-dir runs/week2_vit_10_seed42 --device cuda; `
python -X utf8 -m src.train --config configs/week2/vit_50.yaml --run-dir runs/week2_vit_50_seed42 --device cuda; `
python -X utf8 -m src.train --config configs/week2/vit_100.yaml --run-dir runs/week2_vit_100_seed42 --device cuda
```

#### Cách 2: Chạy riêng rẽ từng mức tỷ lệ:
```powershell
# 1. ViT 10% (4.500 ảnh train — 100 epoch)
python -X utf8 -m src.train --config configs/week2/vit_10.yaml --run-dir runs/week2_vit_10_seed42 --device cuda

# 2. ViT 50% (22.500 ảnh train — 100 epoch)
python -X utf8 -m src.train --config configs/week2/vit_50.yaml --run-dir runs/week2_vit_50_seed42 --device cuda

# 3. ViT 100% (45.000 ảnh train — 100 epoch)
python -X utf8 -m src.train --config configs/week2/vit_100.yaml --run-dir runs/week2_vit_100_seed42 --device cuda
```

---

## 4. Xử lý Sự cố & Khôi phục (Resume)

Nếu tiến trình huấn luyện bị ngắt giữa chừng (mất điện, tắt máy đột ngột), Minh không cần chạy lại từ đầu mà dùng tùy chọn `--resume` trỏ vào `last.pt` của đúng thư mục đó. Mô hình sẽ nạp lại đầy đủ trọng số, optimizer, scheduler và trạng thái RNG:

```powershell
# Ví dụ nối lại lượt 100% nếu bị ngắt:
python -X utf8 -m src.train --config configs/week2/vit_100.yaml --run-dir runs/week2_vit_100_seed42 --device cuda --resume runs/week2_vit_100_seed42/last.pt
```

---

## 5. Quy trình Nghiệm thu và Bàn giao cho Hiền

### 5.1. Kiểm tra 5 Tệp Bắt buộc trong Mỗi Thư mục Run
Mỗi thư mục hoàn thành (`runs/week2_vit_*_seed42/`) phải có đầy đủ:
1. `config.yaml`: Cấu hình thực tế đã dùng.
2. `metadata.json`: Lưu thông tin máy, commit Git, thời gian UTC, số tham số và hash SHA-256 mã nguồn / manifest.
3. `metrics.csv`: Bảng log chi tiết loss/accuracy từng epoch.
4. `best.pt`: Checkpoint có validation accuracy cao nhất.
5. `last.pt`: Checkpoint epoch 100 cuối cùng.

### 5.2. Hiền Tiếp nhận và Xử lý Bằng Chứng (T12)
Sau khi nhận thư mục từ Minh, Hiền chạy lệnh trực quan hóa tự động:
```powershell
python -X utf8 -m src.visualize `
  --run runs/week2_vit_10_seed42 `
  --run runs/week2_vit_50_seed42 `
  --run runs/week2_vit_100_seed42 `
  --out results/hien_week2/vit_main `
  --label validation
```
Script này tự động:
- Kiểm tra tính hợp lệ của checkpoint và đối chiếu độ khớp với log.
- Xuất biểu đồ đường cong học tập (`_curve.png`).
- Xuất ma trận nhầm lẫn (`_confusion.png`).
- Xuất bản đồ attention chồng lên ảnh gốc CIFAR-10 (`_attention.png`) từ phương thức `model.get_attention_weights()`.
- Cập nhật trạng thái `completed` trong [run_matrix.csv](../results/hien_week2/run_matrix.csv).

---

## 6. Ghi nhận Đóng góp và Liêm chính Học thuật

- **Lan Anh:** Tự cài đặt toàn bộ mô hình ViT, thiết kế module trích xuất attention weights, thực hiện khảo sát siêu tham số ban đầu, soạn thảo bản thảo phương pháp luận kiến trúc.
- **Minh (Hỗ trợ):** Cung cấp tài nguyên GPU cá nhân (RTX), vận hành chạy 3 lượt huấn luyện 100 epoch theo hàng đợi dùng chung.
- **Hiền (Hỗ trợ):** Khóa cấu hình YAML, kiểm tra phân tầng dữ liệu, tiếp nhận checkpoint và trích xuất dữ liệu trực quan hóa cho báo cáo.
