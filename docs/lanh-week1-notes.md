# Ghi chú tuần 1 — Lan Anh

## Ngày 1 — Đọc bài ViT và tự cài mô hình

### Tóm tắt bài đọc: An Image is Worth 16x16 Words (Dosovitskiy et al., 2021)

**Nguồn:** Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X.,
Unterthiner, T., ... & Houlsby, N. (2021).
An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
ICLR 2021. <https://arxiv.org/abs/2010.11929>

#### Ý tưởng cốt lõi

Vision Transformer (ViT) áp dụng kiến trúc Transformer — vốn thiết kế cho NLP — trực tiếp lên ảnh. Thay vì dùng convolution, ViT:

1. **Chia ảnh thành patch** (ví dụ 16×16 pixel) — mỗi patch tương đương một "word" trong NLP.
2. **Chiếu tuyến tính** mỗi patch thành embedding vector.
3. **Thêm positional embedding** (learnable) để mã hoá vị trí.
4. **Xử lý qua Transformer encoder** chuẩn (self-attention + MLP).
5. **Phân loại** qua CLS token prepend ở đầu chuỗi.

#### Thành phần kiến trúc

| Thành phần | Mô tả | Vai trò |
|---|---|---|
| Patch Embedding | Chia ảnh H×W thành (H/P × W/P) patch, flatten + Linear | Biến ảnh thành chuỗi token |
| CLS Token | Token learnable gắn vào đầu chuỗi | Tổng hợp thông tin toàn cục cho phân loại |
| Positional Embedding | Learnable, cộng vào mỗi token | Mã hoá vị trí — attention không có thông tin thứ tự |
| Multi-Head Self-Attention | Q, K, V từ mỗi token; attention = softmax(QK^T/√d) × V | Token trao đổi thông tin toàn cục |
| MLP | Linear → GELU → Linear trong mỗi block | Biến đổi phi tuyến cho từng token |
| Layer Normalization | Trước attention và MLP (Pre-LN) | Ổn định training |
| Residual Connection | x + f(x) | Giúp gradient lan truyền |

#### Kết quả chính từ bài báo

- **ViT cần dữ liệu lớn** (JFT-300M, ImageNet-21K) để vượt CNN. Với dữ liệu nhỏ/trung bình, CNN có lợi thế nhờ inductive bias (locality, translation equivariance).
- **Khi đủ dữ liệu**, ViT vượt BiT-ResNet và EfficientNet trên ImageNet.
- Positional embedding learnable 1D hoạt động tốt ngang 2D (với ViT), cho thấy ViT tự học được spatial structure từ data.
- Attention distance tăng theo depth: layer đầu chú ý cục bộ, layer sau chú ý toàn cục.

#### Liên hệ đề tài — áp dụng cho ảnh 32×32 CIFAR-10

| Vấn đề | Nhận xét |
|---|---|
| Ảnh nhỏ 32×32 | Dùng patch 4×4 thay vì 16×16 → 64 patch, chuỗi vừa phải |
| Ít dữ liệu (4.500–45.000) | Dự kiến CNN có lợi thế ở 10%, khoảng cách thu hẹp ở 100% — nhưng viết theo số liệu thật |
| Không pretrained | Train from scratch, ViT có thể yếu hơn so với fine-tune — phù hợp câu hỏi đề tài |
| Inductive bias | ViT thiếu locality bias → cần nhiều dữ liệu hơn để học pattern cục bộ; ResNet có sẵn |

#### Giới hạn khi áp dụng

- ViT nhỏ (~274K params) rất khác ViT-Base (~86M) trong bài gốc. Kết quả trên CIFAR-10 không thể so sánh trực tiếp với bảng trong paper.
- Train from scratch trên dataset nhỏ là thiết lập bất lợi cho ViT — đây là ý nghĩa thí nghiệm, không phải hạn chế thiết kế.
- Batch size, learning rate, augmentation có thể ảnh hưởng khác nhau giữa ViT và CNN — cùng recipe là quy ước nhóm, không phải tối ưu cho từng kiến trúc.

---

## Tự cài ViT

Đã tự cài toàn bộ kiến trúc ViT tại `src/models/vit.py`. Chi tiết các thành phần và kết quả kiểm tra xem [bàn giao tuần 1](lanh-week1.md).

### Cấu hình chốt

- **embed_dim=96, depth=4, heads=4, mlp_ratio=1.5** → 274.282 tham số
- Khớp ResNet 272.474 tham số, chênh 0.7%
- Patch 4×4 trên ảnh 32×32 → 64 patch + 1 CLS = 65 token

### Sơ đồ kiến trúc ViT đã cài

```
Ảnh [B, 3, 32, 32]
    │
    ▼
PatchEmbedding (Conv2d 4×4, stride 4)
    │ → [B, 64, 96]
    ▼
Prepend CLS token
    │ → [B, 65, 96]
    ▼
+ Positional Embedding (learnable)
    │ → [B, 65, 96]
    ▼
Dropout
    │
    ▼
┌─ Transformer Block ×4 ──────────────┐
│  LayerNorm → Multi-Head Attention   │
│  (4 heads, head_dim=24, scale=1/√24)│
│  + Residual                          │
│  LayerNorm → MLP (96→144→96, GELU)  │
│  + Residual                          │
└──────────────────────────────────────┘
    │
    ▼
CLS token [B, 96]
    │
    ▼
LayerNorm → Linear(96, 10)
    │
    ▼
Logits [B, 10]
```
