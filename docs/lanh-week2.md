# Bàn giao Tuần 2 — Lan Anh (ViT 10%, 50%, 100% dữ liệu)

## 1. Phạm vi công việc Tuần 2 (T09, T10)

- **Thời gian thực hiện:** Tuần 2 theo lộ trình kế hoạch [lan-anh.md](../tasks/roadmaps/lan-anh.md).
- **Trách nhiệm:** Lan Anh phụ trách triển khai 3 lượt huấn luyện chính của Vision Transformer trên 3 tỷ lệ dữ liệu (10%, 50%, 100%) với seed ghép cặp đầu tiên (`split_seed=2026, train_seed=42`).
- **Phối hợp:**
  - Khóa siêu tham số và recipe chung với Minh (T09).
  - Xuất ma trận attention mẫu và cung cấp checkpoint cho Hiền trực quan hóa (T12).
  - Viết bản thảo phần kiến trúc Method–ViT ([reports/method_lanh.md](../reports/method_lanh.md)).

---

## 2. Kết quả Khảo sát Siêu tham số (Tuning) Ngày 8

Trước khi chạy chính, Lan Anh đã thực hiện khảo sát độc lập 3 công thức huấn luyện trên tập con 10% (4.500 ảnh, 3 epochs) với seed 42 để tìm recipe tối ưu cho ViT:

| Cấu hình | Optimizer | Learning Rate | Weight Decay | Val Accuracy | Val Loss | Nhận xét |
|---|---|---|---|:---:|:---:|---|
| `sgd_lr005` | SGD (momentum 0.9) | 0.05 | 0.0005 | 25,42% | 1.9568 | Hội tụ chậm hơn trên Transformer |
| `sgd_lr002` | SGD (momentum 0.9) | 0.02 | 0.0005 | 26,24% | 1.9491 | Ổn định hơn lr=0.05 |
| **`adamw_lr001`** | **AdamW** | **0.001** | **0.01** | **29,22%** | **1.8890** | **Vượt trội rõ rệt (+3,8%)** |

> **Quyết định khóa giao thức (T09):** Khóa `AdamW (lr=0.001, weight_decay=0.01)` cùng lịch giảm dần `CosineAnnealingLR` trong 5 epochs cho toàn bộ các lượt chạy chính của ViT.

---

## 3. Ba Lượt Chạy Chính Tuần 2 (T10)

Cả 3 lượt chạy dùng cùng kiến trúc ViT 274.282 tham số (`embed_dim=96, depth=4, heads=4, mlp_ratio=1.5`), cùng seed `(split_seed=2026, train_seed=42)` và cùng quy tắc chọn checkpoint theo validation accuracy:

| Run ID | Tỷ lệ dữ liệu | Số ảnh train | Epochs | Best Val Acc | Best Epoch | Train Acc | Thời gian | Thư mục lưu |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| `vit_d10_s42` | 10% | 4.500 | 5 | **32,00%** (1.600/5.000) | 5 | 31,82% | 1,8 phút | `runs/vit_d10_s42/` |
| `vit_d50_s42` | 50% | 22.500 | 5 | **50,64%** (2.532/5.000) | 5 | 47,47% | 9,1 phút | `runs/vit_d50_s42/` |
| `vit_d100_s42`| 100% | 45.000 | 5 | **56,26%** (2.813/5.000) | 5 | 52,85% | 16,0 phút | `runs/vit_d100_s42/` |

### Phân tích hiện tượng Data Scaling sơ bộ:
- **Từ 10% lên 50% (tăng gấp 5 lần dữ liệu):** Val Accuracy tăng mạnh từ **32,00% lên 50,64%** (+18,64 điểm %).
- **Từ 50% lên 100% (tăng gấp đôi dữ liệu):** Val Accuracy tiếp tục tăng từ **50,64% lên 56,26%** (+5,62 điểm %).
- **Tổng mức tăng trưởng:** Tăng **+24,26%** khi quy mô dữ liệu mở rộng từ 4.500 lên 45.000 ảnh.
- Kết quả này cung cấp minh chứng thực nghiệm rõ rệt cho giả thuyết: Vision Transformer phụ thuộc mạnh vào lượng dữ liệu huấn luyện do thiếu inductive bias tích chập. Khi có nhiều dữ liệu hơn, mạng tự học được các biểu diễn không gian và mẫu đặc trưng hiệu quả hơn.

---

## 4. Dữ liệu Bàn giao cho Hiền (T12)

1. **Ma trận Attention mẫu:** Lưu tại `results/lanh_week2_attentions.pt` gồm các tensor `[10, 4, 65, 65]` (10 ảnh đại diện cho 10 lớp CIFAR-10) để Hiền vẽ bản đồ nhiệt attention chồng lên ảnh gốc.
2. **Checkpoints:** `runs/vit_d{10,50,100}_s42/best.pt` và `last.pt`.
3. **Log TensorBoard & CSV:** Lưu chi tiết tại từng thư mục run và `results/lanh_week2_summary.csv`.

---

## 5. Câu hỏi tự kiểm tra vấn đáp Tuần 2 (Lan Anh)

1. **Vì sao AdamW vượt trội hơn SGD trên Vision Transformer?**
   *Trả lời:* ViT không có inductive bias tích chập (không có sẵn giả định về tính địa phương và tương đương tịnh tiến). Ma trận attention có độ nhạy gradient rất khác nhau giữa các token và giữa các head. AdamW với tốc độ học thích nghi theo từng tham số (adaptive learning rates) và cơ chế weight decay tách rời giúp các tầng self-attention ổn định và hội tụ nhanh hơn nhiều so với SGD momentum.
2. **Hiện tượng Data Scaling thể hiện thế nào qua 3 mức dữ liệu 10%, 50%, 100%?**
   *Trả lời:* Khi lượng dữ liệu tăng từ 4.500 lên 45.000 ảnh, accuracy của ViT tăng trưởng mạnh mẽ vì mô hình có đủ mẫu để học các mối quan hệ không gian phức tạp giữa các patch mà không cần phụ thuộc vào bias cấu trúc sẵn có.
3. **Pre-LN mang lại ưu thế gì cho việc hội tụ của ViT?**
   *Trả lời:* Pre-LN đặt LayerNorm trên nhánh residual trước attention và MLP, giúp gradient đi thẳng qua đường tắt (residual stream) mà không bị co dãn liên tục bởi LayerNorm, ngăn ngừa hiện tượng vanishing/exploding gradient khi mạng bắt đầu huấn luyện.
