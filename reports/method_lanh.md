# Phương pháp luận: Kiến trúc Vision Transformer và Phân tích Inductive Bias

> **Tài liệu bàn giao phần Method — Lan Anh (Tuần 2)**  
> Dùng để Hiền tổng hợp vào phần **Method** và **Analysis** của báo cáo theo định dạng ACL.

---

## 1. Kiến trúc Vision Transformer (ViT) Tự Cài Đặt

Mô hình Vision Transformer được tự cài đặt bằng PyTorch theo bài báo gốc của \citet{dosovitskiy2020image}, nhưng được điều chỉnh để phù hợp với độ phân giải nhỏ $32 \times 32$ của CIFAR-10 và ngân sách tham số tương đương ResNet (~272K tham số).

```
          Ảnh đầu vào x ∈ ℝ^{32 × 32 × 3}
                         │
                         ▼
           Patch Embedding (Conv2d 4×4)
                    [B, 64, 96]
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
   [x_class] ∈ ℝ^{1 × 96}         x_p E ∈ ℝ^{64 × 96}
        └────────────────┬────────────────┘
                         ▼  (Concat)
                    [B, 65, 96]
                         │
                         ▼  (+ E_pos ∈ ℝ^{65 × 96})
             Chuỗi token z_0 ∈ ℝ^{B × 65 × 96}
                         │
                         ▼
        ┌── Transformer Encoder Block (×4) ───────┐
        │  z'_{l-1} = z_{l-1} + MHSA(LN(z_{l-1}))  │
        │  z_l      = z'_{l-1} + MLP(LN(z'_{l-1})) │
        └────────────────┬────────────────────────┘
                         │
                         ▼
              Trích xuất token CLS: z_L^0
                         │
                         ▼
             LayerNorm → Linear(96, 10)
                         │
                         ▼
                  Logits y ∈ ℝ^{10}
```

### 1.1. Patch Embedding và Chuỗi Token Đầu Vào
Với ảnh $x \in \mathbb{R}^{H \times W \times C}$ ($H=W=32, C=3$) và kích thước patch $P=4$:
- Số lượng patch là $N = \frac{HW}{P^2} = \frac{32 \times 32}{16} = 64$.
- Mỗi patch $x_p^i \in \mathbb{R}^{P^2 C} = \mathbb{R}^{48}$ được chiếu tuyến tính qua ma trận nhúng $E \in \mathbb{R}^{48 \times D}$ ($D=96$). Trong cài đặt, phép toán này được thực thi tối ưu bằng một lớp `nn.Conv2d(3, 96, kernel_size=4, stride=4)`.
- Một token phân lớp học được $\mathbf{x}_{\text{class}} \in \mathbb{R}^D$ được ghép vào đầu chuỗi:
  $$\mathbf{z}_0 = [\mathbf{x}_{\text{class}}; \, \mathbf{x}_p^1 E; \, \mathbf{x}_p^2 E; \, \dots; \, \mathbf{x}_p^N E] + \mathbf{E}_{\text{pos}}$$
  trong đó $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ là positional embedding 1D có thể học được.
- Chiều dài chuỗi token cố định là $N+1 = 65$, kích thước tensor: $[B, 65, 96]$.

### 1.2. Khối Transformer Encoder (Pre-LN)
Mỗi khối Transformer Encoder gồm hai thành phần con chính với cấu trúc Pre-LayerNorm (Pre-LN):
1. **Multi-Head Self-Attention (MHSA):**
   - Số attention heads: $h = 4$.
   - Chiều mỗi head: $d_k = D / h = 96 / 4 = 24$.
   - Các ma trận chiếu $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$.
   - Tọa độ chú ý được tính bằng Scaled Dot-Product Attention:
     $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
     với hệ số tỉ lệ $\frac{1}{\sqrt{24}} \approx 0.2041$.
2. **Multi-Layer Perceptron (MLP):**
   - Hai lớp tuyến tính với hàm kích hoạt phi tuyến GELU:
     $$\text{MLP}(u) = \text{GELU}(u W_1 + b_1) W_2 + b_2$$
   - Tỷ lệ mở rộng: $\text{mlp\_ratio} = 1.5 \implies$ chiều ẩn $D_{\text{mlp}} = 96 \times 1.5 = 144$.
3. **Pre-LN Residual Connections:**
   $$\mathbf{z}'_l = \mathbf{z}_{l-1} + \text{MHSA}(\text{LayerNorm}(\mathbf{z}_{l-1}))$$
   $$\mathbf{z}_l = \mathbf{z}'_l + \text{MLP}(\text{LayerNorm}(\mathbf{z}'_l))$$

### 1.3. Classification Head
Đầu ra sau khối encoder thứ 4 lấy riêng biểu diễn của token CLS $\mathbf{z}_L^0 \in \mathbb{R}^D$:
$$y = \text{Linear}(\text{LayerNorm}(\mathbf{z}_L^0)) \in \mathbb{R}^{10}$$
Đầu ra là logits thô (chưa qua softmax), được đưa trực tiếp vào hàm mất mát `nn.CrossEntropyLoss`.

---

## 2. So sánh Tham số và Độ tương đồng với ResNet

Để bảo đảm tính công bằng khi so sánh vai trò của **inductive bias**, hai mô hình được chuẩn hóa số lượng tham số có thể học sao cho chênh lệch không vượt quá ngưỡng quy định ($\le 10\%$):

| Thành phần | ResNet-16 (Minh) | ViT Nhỏ (Lan Anh) |
|---|---|---|
| **Stem / Embedding** | Conv $3\times3$ (432 params) + BN (32) = 464 | Conv $4\times4$ (1.248) + PosEmbed (6.240) + CLS (96) = 7.584 |
| **Thân mô hình** | 9 BasicBlocks (3 stages × 3 blocks) = 271.360 | 4 TransformerBlocks = 265.536 |
| **Phân loại (Head)** | Linear 64 $\to$ 10 (650 params) | LayerNorm (192) + Linear 96 $\to$ 10 (970) = 1.162 |
| **Tổng tham số trainable** | **272.474** | **274.282** |
| **Chênh lệch tỉ đối** | *Baseline* | **+0,66% (1.808 tham số)** |

---

## 3. Vai trò của Inductive Bias trong So sánh Thực nghiệm

Sự khác biệt cốt lõi giữa hai kiến trúc nằm ở mức độ giả định cấu trúc không gian (inductive bias):

1. **Convolutional Neural Networks (ResNet):**
   - **Tính địa phương (Locality):** Phép tích chập $3\times3$ giả định các pixel lân cận có mức độ tương quan cao hơn các pixel ở xa.
   - **Tính tương đương tịnh tiến (Translation Equivariance):** Chia sẻ trọng số (weight sharing) trên toàn bộ không gian ảnh giúp mạng nhận diện mẫu đặc trưng bất kể vị trí xuất hiện.
   - *Hệ quả:* CNN học rất nhanh và ổn định ngay cả khi tập dữ liệu nhỏ (10%), vì mạng không cần tốn dung lượng tham số để "học lại" quy luật hình học không gian 2D.

2. **Vision Transformers (ViT):**
   - **Tự do toàn cục (Global Context):** Self-attention cho phép mọi token tương tác với mọi token khác ngay từ tầng đầu tiên mà không bị ràng buộc bởi khoảng cách không gian.
   - **Tính hoán vị tương đương (Permutation Invariance):** Bản thân self-attention hoàn toàn không biết thứ tự không gian của các patch; toàn bộ thông tin vị trí phụ thuộc vào $\mathbf{E}_{\text{pos}}$.
   - *Hệ quả:* ViT có ít inductive bias cứng hơn, cho phép mô hình linh hoạt hơn nhưng đòi hỏi lượng dữ liệu lớn hơn nhiều để tự học được mối quan hệ không gian.

---

## 4. Thiết kế Thí nghiệm Ablation: Positional Embedding

Trong Tuần 3, nhóm sẽ thực hiện khảo sát cô lập (ablation study) bằng cách gỡ bỏ Positional Embedding (`use_pos_embed = False`):
- Khi không có $\mathbf{E}_{\text{pos}}$, tensor $\mathbf{E}_{\text{pos}}$ được thay bằng buffer hằng số $0$, giảm $6.240$ tham số ($268.042$ tham số).
- Thí nghiệm được thực hiện trên cùng tập con $50\%$ dữ liệu qua 3 seeds để đo lường mức độ suy giảm accuracy, từ đó định lượng tầm quan trọng của thông tin vị trí không gian đối với ViT trên ảnh nhỏ $32\times32$.
