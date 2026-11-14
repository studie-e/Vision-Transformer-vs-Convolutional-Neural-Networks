# Đoạn đóng góp Proposal — Lan Anh (ViT và Ablation)

Tệp này chứa nội dung Lan Anh bàn giao cho Hiền để cập nhật vào `reports/proposal.tex` (bản nộp chính thức theo template ACL).

---

## 1. Mô tả kiến trúc ViT tự cài (Method — ViT)

Vision Transformer (ViT) được **tự cài đặt hoàn toàn bằng PyTorch** theo kiến trúc gốc của \citet{dosovitskiy2020image}, tối ưu cho kích thước ảnh nhỏ 32×32 của CIFAR-10. Nhóm không sử dụng bất kỳ thư viện ViT hoàn chỉnh có sẵn nào. Các thành phần tự cài bao gồm:

- **Patch Embedding:** Sử dụng `nn.Conv2d(3, 96, kernel_size=4, stride=4)` để chia ảnh 32×32 thành lưới $8 \times 8 = 64$ patch, mỗi patch được chiếu tuyến tính thành vector không gian nhúng $D = 96$.
- **CLS Token:** Một vector tham số học được `[1, 1, 96]` nối vào đầu chuỗi patch token, tạo thành chuỗi $N = 65$ token. Vector này đóng vai trò đại diện biểu diễn toàn cục phục vụ phân loại.
- **Positional Embedding:** Tham số học được 1D có kích thước `[1, 65, 96]` cộng trực tiếp vào chuỗi token sau phép chiếu patch, cung cấp thông tin không gian mà self-attention không có sẵn.
- **Transformer Encoder Blocks (4 khối):** Thiết kế theo chuẩn Pre-LN:
  - $\mathbf{x}' = \mathbf{x} + \mathrm{MHSA}(\mathrm{LayerNorm}(\mathbf{x}))$
  - $\mathbf{x}'' = \mathbf{x}' + \mathrm{MLP}(\mathrm{LayerNorm}(\mathbf{x}'))$
  - Multi-Head Self-Attention (MHSA) có 4 head song song, mỗi head có $d_k = 24$, hệ số co dãn $\frac{1}{\sqrt{24}} \approx 0.204$.
  - MLP gồm 2 lớp tuyến tính với hàm kích hoạt GELU và tỷ lệ mở rộng $\mathrm{mlp\_ratio} = 1.5$ ($D \to 144 \to D$).
  - Dropout $p = 0.1$ sau attention và MLP.
- **Classification Head:** Trích xuất CLS token ở đầu ra tầng cuối, chuẩn hóa qua `nn.LayerNorm(96)` và đưa qua `nn.Linear(96, 10)` để xuất logits 10 lớp (chưa qua softmax, tương thích chuẩn `CrossEntropyLoss`).

---

## 2. Thống kê và khớp số tham số (Khớp ResNet)

Nhóm đã tính toán và kiểm tra chính xác số tham số học được (trainable parameters):

| Mô hình | Cấu hình | Số tham số | Chênh lệch so với ResNet |
|---|---|---:|:---:|
| **ResNet (Minh)** | Width 16, 3 stages × 3 basic blocks, projection shortcuts | 272.474 | Baseline |
| **ViT tự cài (Lan Anh)** | Patch 4×4, $D=96$, depth 4, 4 heads, MLP ratio 1.5 | **274.282** | **+0,66% (1.808 tham số)** |
| **ViT Ablation (No Pos)** | Bỏ positional embedding ($D=96$, depth 4) | 268.042 | −1,63% |

> **Kết luận:** Chênh lệch số tham số giữa ViT và ResNet đạt **0,66%**, nằm sâu trong ngưỡng mục tiêu đề xuất $\le 10\%$, đảm bảo tính công bằng cao cho việc so sánh inductive bias.

---

## 3. Cơ chế xuất Attention Map

Nhằm phục vụ phân tích trực quan hóa của Hiền, mô hình ViT tích hợp phương thức `model.get_attention_weights()`:
- Trả về danh sách 4 tensor (tương ứng 4 encoder block).
- Mỗi tensor có kích thước $[B, 4, 65, 65]$, trong đó mỗi hàng biểu diễn phân phối xác suất chú ý của 1 token lên 65 token còn lại (đã qua Softmax theo chiều cuối).
- Đã được kiểm thử lưu trữ mẫu trên batch validation CIFAR-10 tại `runs/lanh_week1_smoke/sample_attention.pt`.

---

## 4. Kết quả Pilot tuần 1 của Lan Anh

- **Suite kiểm thử:** 9 test tự động trong `tests/test_lanh.py` đạt 100%; forward, backward, gradient flow, parameter budget, overfit batch nhỏ đều thành công.
- **Chạy pilot trên dữ liệu thật:** Huấn luyện 1 epoch trên 4.500 ảnh CIFAR-10 (tỷ lệ 10%, split seed 2026, train seed 42):
  - Train loss: 2.2947, Train accuracy: 17,40%
  - Val loss: 2.1755, Val accuracy: **20,34% (1.017/5.000 ảnh)**
  - Thời gian: 16,57 giây trên CPU (36 steps).
  - Evaluator `src.evaluate` và checkpoint của Trainer khớp chính xác 100%.
- Minh chứng kỹ thuật đầy đủ được lưu tại `results/lanh_pilot_runs.csv`, `results/lanh_pilot_summary.csv`, và `results/lanh_week1_checks.json`.
