# Ghi chú tuần 1 — Hiền

## Ngày 1 — Đọc bài Swin Transformer và chốt dữ liệu

### Tóm tắt bài đọc: Swin Transformer (Liu et al., 2021)

**Nguồn:** Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021).
Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.
arXiv:2103.14030. <https://arxiv.org/abs/2103.14030>

#### Ý tưởng cốt lõi

Swin Transformer giải quyết hai hạn chế của ViT gốc khi áp dụng cho ảnh độ phân giải cao:
1. **Chi phí bậc hai theo số patch** — ViT tính self-attention trên toàn bộ patch, tốn O(N²) với N là số patch. Swin giới hạn attention trong từng cửa sổ cục bộ (window) cố định M×M patch, giảm còn O(N·M²).
2. **Thiếu tính phân cấp** — ViT dùng patch đơn kích thước suốt; Swin gộp patch (patch merging) để tạo feature map mịn→thô như CNN, tương thích với các tác vụ dense (detection, segmentation).

#### Thành phần kiến trúc

- **Patch partition:** chia ảnh thành patch 4×4, embed tuyến tính → feature map C kênh.
- **Patch merging:** gộp 2×2 patch lại sau mỗi stage, giảm không gian, tăng kênh; giống pooling trong ResNet.
- **W-MSA (Window Multi-head Self-Attention):** chia feature map thành các cửa sổ M×M (mặc định M=7); tính self-attention độc lập trong từng cửa sổ. Không có thông tin qua ranh giới cửa sổ ở bước này.
- **SW-MSA (Shifted Window MSA):** dịch cửa sổ đi floor(M/2) vị trí trên lưới token để tạo kết nối giữa các cửa sổ liền kề ở lớp tiếp theo. Dùng cyclic shift + masking để tính hiệu quả trên một lần attention.
- **Relative position bias:** thêm bias tương đối dựa trên khoảng cách patch trong cửa sổ, không dùng tọa độ tuyệt đối như ViT.
- **MLP block:** hai lớp Linear với GELU, giống ViT; layer norm đặt trước (Pre-LN).

#### Liên hệ đề tài — áp dụng cho ảnh 32×32 CIFAR

| Vấn đề | Nhận xét |
|---|---|
| Lưới 8×8 token, M=7 → 2×2 cửa sổ nếu padding thành 14×14 | Rất ít cửa sổ, không khai thác được Shifted Window đầy đủ; cần giảm M xuống 2 hoặc 4 |
| Patch 4×4 trên 32×32 → 8×8 patch | Vẫn dùng được; nhóm dự kiến ViT dùng patch 4×4 cho ảnh 32×32 |
| Patch merging giảm 4× spatial | Sau 1–2 stage spatial còn 4×4 hoặc 2×2; phù hợp cho classification head global average pooling |
| Relative position bias | Vẫn áp dụng được; đây là một khác biệt thiết kế của Swin so với ViT không có positional bias |
| Số tham số | Swin-Tiny ~28M; nhóm cần dùng mô hình nhỏ hơn nhiều để khớp ResNet ~272K |

**Kết luận:** Swin cho ảnh nhỏ cần chỉnh patch size, window size và số stage/kênh. Đây là thí nghiệm bổ sung (P2), không phải thay thế bảng chính ViT–ResNet. Nhóm sẽ dùng thư viện có trích dẫn, ghi rõ số tham số và recipe; nếu không khớp ngân sách thì trình bày thành thí nghiệm thăm dò.

#### Giới hạn khi áp dụng cho ảnh nhỏ

- Shifted Window và Relative Position Bias thiết kế cho ảnh lớn (224×224 trở lên); cần kiểm nghiệm hiệu quả khi feature map nhỏ, không giả định trước kết quả.
- Khó so sánh công bằng với ViT/ResNet nếu số tham số và recipe khác nhau đáng kể.
- Kết quả trên CIFAR-10 với Swin nhỏ được coi là thí nghiệm khám phá, không phải bằng chứng nhân quả về inductive bias.

---

## Chốt dữ liệu: CIFAR-10

Nhóm chốt **CIFAR-10, ảnh gốc 32×32**. Nguồn chính thức: <https://www.cs.toronto.edu/~kriz/cifar.html>

| Phân chia | Số ảnh | Ghi chú |
|---|---|---|
| Train gốc | 50.000 | Tách thành 45k train / 5k validation |
| Test | 10.000 | Chỉ dùng đánh giá cuối, không tuning |
| Train chính | 45.000 | Phân tầng theo 10 lớp |
| Validation | 5.000 | Phân tầng theo 10 lớp, không giao nhau với train |

Tỷ lệ tập con (tính trên 45.000 train):
- 10% → 4.500 ảnh (450 ảnh/lớp)
- 50% → 22.500 ảnh (2.250 ảnh/lớp)
- 100% → 45.000 ảnh (4.500 ảnh/lớp)

Tập con lồng nhau: 10% ⊂ 50% ⊂ 100%. Hai mô hình ViT và ResNet dùng **cùng danh sách ảnh** từ manifest.

