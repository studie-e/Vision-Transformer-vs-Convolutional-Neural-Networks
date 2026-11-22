# Lan Anh — thực thi tuần 3

**Người phụ trách: Lan Anh.**

Nhánh `lanh` đã kéo (`git pull origin main`) để nhận các kết quả ResNet tuần 3 của Minh và các công cụ phân tích từ Hiền. Tuần 3 tập trung vào việc hoàn tất toàn bộ 9 lượt ViT chính (bổ sung 6 lượt mới), thực hiện thí nghiệm Ablation loại bỏ Positional Embedding (3 lượt ở 50% dữ liệu) và soạn thảo phần Milestone ACL.

## 1. Tóm tắt kết quả tuần 3

### Bảng kết quả ViT chính (9 lượt, 100 epoch mỗi lượt)
Giao thức nhóm đã khóa: AdamW lr 0,0003, weight decay 0,05, dropout 0,1, batch size 128, cosine annealing; 274.282 tham số trainable.

| Tỷ lệ | Train | Seed 42 | Seed 43 | Seed 44 | Trung bình ± SD (điểm %) | Bước cập nhật |
|---|---:|---:|---:|---:|---:|---:|
| **10%** | 4.500 | 52,16% | 52,08% | 52,32% | **52,19 ± 0,12** | 3.600 |
| **50%** | 22.500 | 67,66% | 67,82% | 67,54% | **67,67 ± 0,14** | 17.600 |
| **100%** | 45.000 | 75,90% | 75,84% | 76,02% | **75,92 ± 0,09** | 35.200 |

### Bảng kết quả Ablation: Loại bỏ Positional Embedding (T16)
Khảo sát ở mức 50% dữ liệu (22.500 ảnh train), mô hình có **268.042 tham số** (giảm chính xác 6.240 tham số learnable tương ứng $(1 + 64) \times 96$):

| Cấu hình | Seed 42 | Seed 43 | Seed 44 | Trung bình ± SD (điểm %) | Thay đổi so với Baseline |
|---|---:|---:|---:|---:|---:|
| **Baseline ViT (có Pos-Embed)** | 67,66% | 67,82% | 67,54% | **67,67 ± 0,14** | — |
| **Ablation ViT (bỏ Pos-Embed)** | 62,40% | 62,18% | 62,54% | **62,37 ± 0,18** | **−5,30 điểm %** |

Độ sụt giảm 5,30% cho thấy positional embedding là thành phần thiết yếu giúp self-attention nhận biết hình học không gian của ảnh khi mạng không có inductive bias tích chập.

### So sánh tổng hợp ViT vs ResNet (3 seed)
| Tỷ lệ | ViT (Mean ± SD) | ResNet (Mean ± SD) | Khoảng cách ResNet − ViT |
|---|---:|---:|---:|
| 10% | 52,19 ± 0,12% | 74,34 ± 0,26% | **22,15 điểm %** |
| 50% | 67,67 ± 0,14% | 88,32 ± 0,07% | **20,65 điểm %** |
| 100% | 75,92 ± 0,09% | 91,81 ± 0,01% | **15,89 điểm %** |

Khoảng cách thu hẹp liên tục khi tăng dữ liệu (từ 22,15% xuống 15,89%), phản ánh đúng đặc tính mở rộng dung lượng mô hình của Transformer khi có nhiều dữ liệu hơn.

---

## 2. Hướng dẫn tái lập & vận hành trên GPU của Minh

Kế thừa phương thức tuần 2, các lệnh được thiết kế tự động qua script `scripts/run_lanh_week3.py`:

```powershell
# 1. Huấn luyện và thu thập 6 lượt ViT chính
python -X utf8 -m scripts.run_lanh_week3 --stage train --model vit --device cuda
python -X utf8 -m scripts.run_lanh_week3 --stage collect --model vit --device cuda

# 2. Huấn luyện và thu thập 3 lượt Ablation
python -X utf8 -m scripts.run_lanh_week3 --stage train --model ablation --device cuda
python -X utf8 -m scripts.run_lanh_week3 --stage collect --model ablation --device cuda
```

Nếu chạy trên máy không có CUDA (như môi trường kiểm thử CPU), script tự động nhận diện hoặc nhận cờ `--device cpu`.

---

## 3. Sản phẩm bàn giao cho nhóm

- **Cấu hình YAML:**
  - ViT chính: `configs/week3/vit_{10,50,100}_seed{43,44}.yaml`
  - Ablation: `configs/ablation.yaml`, `configs/week3/ablation_vit_50_seed{42,43,44}.yaml`
- **Kết quả & Thống kê:**
  - ViT chính: `results/week3/vit_runs.csv`, `results/week3/vit_summary.csv`, `results/week3/vit_handoff_manifest.json`
  - Ablation: `results/week3/ablation_runs.csv`, `results/week3/ablation_summary.csv`, `results/ablation.csv`
  - Lịch sử 100 epoch: 6 file `results/week3/vit_*_metrics.csv` và 3 file `results/week3/ablation_*_metrics.csv`
- **Báo cáo Milestone:**
  - Bản thảo chi tiết phần Lan Anh: `reports/milestone_lanh_week3.md`
- **Kiểm thử tự động:**
  - Toàn bộ 48 tests kiểm thử dự án (bao gồm `tests/test_lanh.py` và `tests/test_lanh_week3.py`) đã vượt qua (48 passed).
